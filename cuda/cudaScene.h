#pragma once
#include "../Optimizations.h"
#include "../Scene.h"
#include "../Vec.h"
#include "cudaCamera.h"
#include "cudaLight.h"
#include "cudaObject.h"
#include <fbxsdk.h>

namespace RT
{
    namespace CUDA
    {
        struct SceneTypes
        {
            using CameraType = RT::CUDA::Camera;
            using ObjectType = RT::CUDA::Object;
            using LightType = RT::CUDA::Light;

            struct PrivateData
            {
                Array<CameraData> CameraDeviceMemory;
                Array<ObjectData> ObjectDeviceMemory;
                Array<LightData> LightDeviceMemory;
                Array<Triangle> TriangleDeviceMemory;
            };

            PrivateData Private;
        };

        template<typename SceneTypes = RT::CUDA::SceneTypes>
        class SceneFunctions
        {
        public:
            inline static void CameraCountHint(
                Scene<MakeSceneTraits<SceneTypes, SceneFunctions>>& scene, 
                size_t count )
            {
                scene.Private.CameraDeviceMemory = Array<CameraData>( count );
            }

            inline static void OnCamerasLoaded(
                Scene<MakeSceneTraits<SceneTypes, SceneFunctions>>& scene )
            {
                scene.Private.CameraDeviceMemory.Update();
            }

            inline static typename SceneTypes::CameraType CreateCameraFromFbx(
                Scene<MakeSceneTraits<SceneTypes, SceneFunctions>>& scene,
                size_t index,
                fbxsdk::FbxNode* pCameraNode )
            {
                // Get camera properties
                fbxsdk::FbxCamera* pCamera = static_cast<fbxsdk::FbxCamera*>(pCameraNode->GetNodeAttribute());

                RT::vec4 position = RT::vec4( pCamera->Position.Get() );
                RT::vec4 target = RT::vec4( pCamera->InterestPosition.Get() );
                RT::vec4 direction;
                RT::vec4 up = RT::vec4( pCamera->UpVector.Get() );
                RT::float_t fov = static_cast<RT::float_t>(pCamera->FieldOfViewX.Get());
                RT::float_t aspect = static_cast<RT::float_t>(pCamera->FilmAspectRatio.Get());
                RT::float_t focal = static_cast<RT::float_t>(pCamera->FocalLength.Get());

                // Evaluate direction
                direction = target - position;
                direction.Normalize3();

                // Store camera properties in internal structure
                typename SceneTypes::CameraType cudaCamera( scene.Private.CameraDeviceMemory, index );
                cudaCamera.Memory.Host().Origin = position;
                cudaCamera.Memory.Host().Direction = direction;
                cudaCamera.Memory.Host().Up = up;
                cudaCamera.Memory.Host().HorizontalFOV = RT::Radians( fov );
                cudaCamera.Memory.Host().AspectRatio = aspect;

                return cudaCamera;
            }

            inline static void LightCountHint(
                Scene<MakeSceneTraits<SceneTypes, SceneFunctions>>& scene,
                size_t count )
            {
                scene.Private.LightDeviceMemory = Array<LightData>( count );
            }

            inline static void OnLightsLoaded(
                Scene<MakeSceneTraits<SceneTypes, SceneFunctions>>& scene )
            {
                scene.Private.LightDeviceMemory.Update();
            }

            inline static typename SceneTypes::LightType CreateLightFromFbx(
                Scene<MakeSceneTraits<SceneTypes, SceneFunctions>>& scene,
                size_t index,
                fbxsdk::FbxNode* pLightNode )
            {
                typename SceneTypes::LightType cudaLight( scene.Private.LightDeviceMemory, index );
                cudaLight.Memory.Host().Position = RT::vec4( pLightNode->LclTranslation.Get() );
                cudaLight.Memory.Host().Subdivs = RT_LIGHT_SUBDIVS;
                cudaLight.Memory.Host().Radius = RT_LIGHT_RADIUS;
                cudaLight.Memory.Host().ShadowBias = 0.04f;

                return cudaLight;
            }

            inline static void ObjectCountHint(
                Scene<MakeSceneTraits<SceneTypes, SceneFunctions>>& scene,
                const std::vector<fbxsdk::FbxNode*>& objects )
            {
                scene.Private.ObjectDeviceMemory = Array<ObjectData>( objects.size() );

                // Calculate total number of triangles
                size_t numTriangles = 0;

                for( const auto& pObjectNode : objects )
                {
                    numTriangles += pObjectNode->GetMesh()->GetPolygonCount();
                }

                scene.Private.TriangleDeviceMemory = Array<Triangle>( numTriangles );
            }

            inline static void OnObjectsLoaded(
                Scene<MakeSceneTraits<SceneTypes, SceneFunctions>>& scene )
            {
                scene.Private.ObjectDeviceMemory.Update();
                scene.Private.TriangleDeviceMemory.Update();
            }

            inline static typename SceneTypes::ObjectType CreateObjectFromFbx(
                Scene<MakeSceneTraits<SceneTypes, SceneFunctions>>& scene,
                size_t index,
                fbxsdk::FbxNode* pObjectNode )
            {
                // Ugh
                static size_t numTrianglesProcessed = 0;

                fbxsdk::FbxAMatrix meshTransform = GetMeshTransform( pObjectNode );
                fbxsdk::FbxAMatrix normalTransform = GetNormalTransform( pObjectNode );

                // Get mesh
                fbxsdk::FbxMesh* pMesh = pObjectNode->GetMesh();

                volatile const char* pName = pMesh->GetName();

                // Get vertex properties
                const int vertexCount = pMesh->GetControlPointsCount();
                const fbxsdk::FbxVector4* pElementVertices = pMesh->GetControlPoints();
                const fbxsdk::FbxLayerElementNormal* pElementNormals = pMesh->GetElementNormal();
                const fbxsdk::FbxLayerElementTangent* pElementTangents = pMesh->GetElementTangent();
                const fbxsdk::FbxLayerElementUV* pElementUVs = pMesh->GetElementUV();
                const fbxsdk::FbxLayerElementVertexColor* pElementColors = pMesh->GetElementVertexColor();

                const int polygonCount = pMesh->GetPolygonCount();

                typename SceneTypes::ObjectType cudaObject( scene.Private.ObjectDeviceMemory, index );

                cudaObject.Triangles = scene.Private.TriangleDeviceMemory;
                cudaObject.Memory.Host().FirstTriangle = static_cast<int>(numTrianglesProcessed);
                cudaObject.Memory.Host().NumTriangles = static_cast<int>(polygonCount);

                // Iterate over all polygons in the mesh
                for( int poly = 0; poly < polygonCount; ++poly )
                {
                    typename SceneTypes::ObjectType::TriangleType tri;
                    tri.A = vec4( meshTransform.MultT( pElementVertices[pMesh->GetPolygonVertex( poly, 0 )] ) );
                    tri.B = vec4( meshTransform.MultT( pElementVertices[pMesh->GetPolygonVertex( poly, 1 )] ) );
                    tri.C = vec4( meshTransform.MultT( pElementVertices[pMesh->GetPolygonVertex( poly, 2 )] ) );

                    // Per-vertex normals
                    tri.An = vec4( normalTransform.MultT( GetElement( pMesh, pElementNormals, poly, 0, 0 ) ) );
                    tri.An.Normalize3();
                    tri.Bn = vec4( normalTransform.MultT( GetElement( pMesh, pElementNormals, poly, 0, 1 ) ) );
                    tri.Bn.Normalize3();
                    tri.Cn = vec4( normalTransform.MultT( GetElement( pMesh, pElementNormals, poly, 0, 2 ) ) );
                    tri.Cn.Normalize3();

                    #if RT_ENABLE_BACKFACE_CULL
                    // Normal of the triangle is average (normalized sum) of normals of each of its vertices
                    tri.Normal = tri.An + tri.Bn + tri.Cn;
                    tri.Normal.Normalize3();
                    #endif

                    #if RT_ENABLE_BOUNDING_BOXES
                    // Update bounding box of the object
                    cudaObject.Memory.Host().BoundingBox.Min.x = std::min( std::min( tri.A.x, tri.B.x ), std::min( tri.C.x, cudaObject.Memory.Host().BoundingBox.Min.x ) );
                    cudaObject.Memory.Host().BoundingBox.Max.x = std::max( std::max( tri.A.x, tri.B.x ), std::max( tri.C.x, cudaObject.Memory.Host().BoundingBox.Max.x ) );
                    cudaObject.Memory.Host().BoundingBox.Min.y = std::min( std::min( tri.A.y, tri.B.y ), std::min( tri.C.y, cudaObject.Memory.Host().BoundingBox.Min.y ) );
                    cudaObject.Memory.Host().BoundingBox.Max.y = std::max( std::max( tri.A.y, tri.B.y ), std::max( tri.C.y, cudaObject.Memory.Host().BoundingBox.Max.y ) );
                    cudaObject.Memory.Host().BoundingBox.Min.z = std::min( std::min( tri.A.z, tri.B.z ), std::min( tri.C.z, cudaObject.Memory.Host().BoundingBox.Min.z ) );
                    cudaObject.Memory.Host().BoundingBox.Max.z = std::max( std::max( tri.A.z, tri.B.z ), std::max( tri.C.z, cudaObject.Memory.Host().BoundingBox.Max.z ) );
                    #endif

                    cudaObject.Triangles.Host( numTrianglesProcessed + poly ) = tri;
                }

                if( auto* pMaterial = (fbxsdk::FbxSurfacePhong*)pObjectNode->GetMaterial( 0 ) )
                {
                    cudaObject.Memory.Host().Color = vec4( pMaterial->Diffuse.Get() ) * 255.f;
                    cudaObject.Memory.Host().Ior = static_cast<RT::float_t>(pMaterial->Specular.Get()[0] * 10.f);
                }
                else
                {
                    cudaObject.Memory.Host().Color.x = static_cast<RT::float_t>(rand() % 256);
                    cudaObject.Memory.Host().Color.y = static_cast<RT::float_t>(rand() % 256);
                    cudaObject.Memory.Host().Color.z = static_cast<RT::float_t>(rand() % 256);
                    cudaObject.Memory.Host().Ior = 3.5f;
                }

                // Ugh #2
                numTrianglesProcessed += polygonCount;

                return cudaObject;
            }

        private:
            inline static fbxsdk::FbxAMatrix GetMeshTransform( fbxsdk::FbxNode* pNode )
            {
                fbxsdk::FbxAMatrix meshTransform;
                meshTransform.SetIdentity();

                if( pNode->GetNodeAttribute() )
                {
                    meshTransform.SetT( pNode->GetGeometricTranslation( fbxsdk::FbxNode::eSourcePivot ) );
                    meshTransform.SetR( pNode->GetGeometricRotation( fbxsdk::FbxNode::eSourcePivot ) );
                    meshTransform.SetS( pNode->GetGeometricScaling( fbxsdk::FbxNode::eSourcePivot ) );
                }

                fbxsdk::FbxNode* pParentNode = pNode->GetParent();
                fbxsdk::FbxAMatrix parentMatrix = pParentNode->EvaluateLocalTransform();

                while( (pParentNode = pParentNode->GetParent()) != nullptr )
                {
                    parentMatrix = pParentNode->EvaluateLocalTransform() * parentMatrix;
                }

                return parentMatrix * pNode->EvaluateLocalTransform() * meshTransform;
            }

            inline static fbxsdk::FbxAMatrix GetNormalTransform( fbxsdk::FbxNode* pNode )
            {
                return fbxsdk::FbxAMatrix( fbxsdk::FbxVector4( 0, 0, 0 ), GetMeshTransform( pNode ).GetR(), fbxsdk::FbxVector4( 1, 1, 1 ) );
            }

            template<typename T>
            inline static T GetElement( const fbxsdk::FbxMesh* pMesh, const fbxsdk::FbxLayerElementTemplate<T>* pElementElements, int poly, int edge, int vert )
            {
                int index = -1;

                switch( pElementElements->GetMappingMode() )
                {
                case fbxsdk::FbxLayerElement::eAllSame:         index = 0; break;
                case fbxsdk::FbxLayerElement::eByControlPoint:  index = pMesh->GetPolygonVertex( poly, vert ); break;
                    //case fbxsdk::FbxLayerElement::eByEdge:          index = edge; break;
                case fbxsdk::FbxLayerElement::eByPolygon:       index = poly; break;
                case fbxsdk::FbxLayerElement::eByPolygonVertex: index = pMesh->GetPolygonVertexIndex( poly ) + vert; break;
                default: throw std::runtime_error( "Unsupported fbxsdk::FbxLayerElement::EMappingMode" );
                }

                switch( pElementElements->GetReferenceMode() )
                {
                case fbxsdk::FbxLayerElement::eDirect: break;
                case fbxsdk::FbxLayerElement::eIndexToDirect:   index = pElementElements->GetIndexArray().GetAt( index ); break;
                    // case fbxsdk::FbxLayerElement::eIndex:
                default: throw std::runtime_error( "Unsupported fbxsdk::FbxLayerElement::EReferenceMode" );
                }

                return pElementElements->GetDirectArray().GetAt( index );
            }
        };
    }
}
