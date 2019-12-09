#pragma once
#include "../Optimizations.h"
#include "../Scene.h"
#include "ompCamera.h"
#include "ompLight.h"
#include "ompObject.h"
#include <algorithm>
#include <fbxsdk.h>

namespace RT::OMP
{
    struct SceneTypes
    {
        using CameraType = RT::OMP::Camera;
        using ObjectType = RT::OMP::Object;
        using LightType = RT::OMP::Light;
    };

    template<typename SceneTypes = RT::OMP::SceneTypes>
    class SceneFunctions
    {
    public:
        inline static typename SceneTypes::CameraType CreateCameraFromFbx( fbxsdk::FbxNode* pCameraNode )
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
            typename SceneTypes::CameraType ompCamera;
            ompCamera.Origin = position;
            ompCamera.Direction = direction;
            ompCamera.Up = up;
            ompCamera.HorizontalFOV = RT::Radians( fov );
            ompCamera.AspectRatio = aspect;

            return ompCamera;
        }

        inline static typename SceneTypes::LightType CreateLightFromFbx( fbxsdk::FbxNode* pLightNode )
        {
            typename SceneTypes::LightType ompLight;
            ompLight.Position = RT::vec4( pLightNode->LclTranslation.Get() );
            ompLight.Subdivs = RT_LIGHT_SUBDIVS;

            return ompLight;
        }

        inline static typename SceneTypes::ObjectType CreateObjectFromFbx( fbxsdk::FbxNode* pObjectNode )
        {
            fbxsdk::FbxAMatrix meshTransform = GetMeshTransform( pObjectNode );
            #if RT_ENABLE_BACKFACE_CULL
            fbxsdk::FbxAMatrix normalTransform = GetNormalTransform( pObjectNode );
            #endif

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

            typename SceneTypes::ObjectType ompObject;

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
                ompObject.BoundingBox.Min.x = std::min( std::min( tri.A.x, tri.B.x ), std::min( tri.C.x, ompObject.BoundingBox.Min.x ) );
                ompObject.BoundingBox.Max.x = std::max( std::max( tri.A.x, tri.B.x ), std::max( tri.C.x, ompObject.BoundingBox.Max.x ) );
                ompObject.BoundingBox.Min.y = std::min( std::min( tri.A.y, tri.B.y ), std::min( tri.C.y, ompObject.BoundingBox.Min.y ) );
                ompObject.BoundingBox.Max.y = std::max( std::max( tri.A.y, tri.B.y ), std::max( tri.C.y, ompObject.BoundingBox.Max.y ) );
                ompObject.BoundingBox.Min.z = std::min( std::min( tri.A.z, tri.B.z ), std::min( tri.C.z, ompObject.BoundingBox.Min.z ) );
                ompObject.BoundingBox.Max.z = std::max( std::max( tri.A.z, tri.B.z ), std::max( tri.C.z, ompObject.BoundingBox.Max.z ) );
                #endif

                ompObject.Triangles.push_back( tri );
            }

            ompObject.Color.x = static_cast<RT::float_t>(rand() % 256);
            ompObject.Color.y = static_cast<RT::float_t>(rand() % 256);
            ompObject.Color.z = static_cast<RT::float_t>(rand() % 256);

            return ompObject;
        }

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
