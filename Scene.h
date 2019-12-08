#pragma once
#include "Fbx.h"
#include "Optimizations.h"
#include <stdexcept>
#include <vector>

namespace RT
{
    template<typename SceneTypes_, typename SceneFunctions_>
    struct MakeSceneTraits
    {
        using SceneFunctions = SceneFunctions_;
        using SceneTypes = SceneTypes_;
    };

    template<typename SceneTraits>
    class Scene : public SceneTraits::SceneFunctions, public SceneTraits::SceneTypes
    {
    protected:
        using TT = Scene<SceneTraits>;
    public:
        std::vector<typename TT::CameraType> Cameras;
        std::vector<typename TT::LightType> Lights;
        std::vector<typename TT::ObjectType> Objects;
    };

    class SceneLoader
    {
    public:
        inline SceneLoader()
        {
            m_pFbxManager = FbxCreate<fbxsdk::FbxManager>();
            m_pFbxIOSettings = FbxCreate<fbxsdk::FbxIOSettings>( m_pFbxManager.get(), "RT_FBX_IOSETTINGS" );
            m_pFbxManager->SetIOSettings( m_pFbxIOSettings.get() );
        }

        template<typename SceneTraits>
        inline Scene<SceneTraits> LoadScene( std::string path )
        {
            using Scene = Scene<SceneTraits>;

            // Create importer for the scene
            auto pImporter = FbxCreate<fbxsdk::FbxImporter>( m_pFbxManager.get(), "RT_FBX_IMPORTER" );

            if( !pImporter->Initialize( path.c_str(), -1, m_pFbxIOSettings.get() ) )
            {
                // Initialization of the importer failed
                throw std::runtime_error( "Failed to initialize FbxImporter" );
            }

            // Create the scene
            auto pScene = FbxCreate<fbxsdk::FbxScene>( m_pFbxManager.get(), "RT_FBX_SCENE" );

            if( !pImporter->Import( pScene.get() ) )
            {
                // Import failed
                throw std::runtime_error( "Failed to import FbxScene" );
            }

            fbxsdk::FbxNode* pRootNode = pScene->GetRootNode();

            // Create scene object
            Scene scene;

            // Get all available cameras
            std::vector<fbxsdk::FbxNode*> pCameraNodes;
            FindCameras( pRootNode, pCameraNodes );

            for( fbxsdk::FbxNode* pCamera : pCameraNodes )
            {
                scene.Cameras.push_back( Scene::CreateCameraFromFbx( pCamera ) );
            }

            // Get all lights
            std::vector<fbxsdk::FbxNode*> pLightNodes;
            FindLights( pRootNode, pLightNodes );

            for( fbxsdk::FbxNode* pLight : pLightNodes )
            {
                scene.Lights.push_back( Scene::CreateLightFromFbx( pLight ) );
            }

            // Get all objects
            std::vector<fbxsdk::FbxNode*> pObjectNodes;
            FindObjects( pRootNode, pObjectNodes );

            for( fbxsdk::FbxNode* pObject : pObjectNodes )
            {
                scene.Objects.push_back( Scene::CreateObjectFromFbx( pObject ) );
            }

            return scene;
        }

    protected:
        RT::FbxHandle<fbxsdk::FbxManager> m_pFbxManager;
        RT::FbxHandle<fbxsdk::FbxIOSettings> m_pFbxIOSettings;

        inline void FindCameras( fbxsdk::FbxNode* pNode, std::vector<fbxsdk::FbxNode*>& pCameraNodes )
        {
            if( pNode )
            {
                fbxsdk::FbxNodeAttribute* pNodeAttribute = pNode->GetNodeAttribute();

                if( pNodeAttribute )
                {
                    // Camera is special kind of node
                    if( pNodeAttribute->GetAttributeType() == fbxsdk::FbxNodeAttribute::eCamera )
                    {
                        pCameraNodes.push_back( pNode );
                    }
                }

                const int childCount = pNode->GetChildCount();

                for( int i = 0; i < childCount; ++i )
                {
                    // Find cameras recursively
                    FindCameras( pNode->GetChild( i ), pCameraNodes );
                }
            }
        }

        inline void FindLights( fbxsdk::FbxNode* pNode, std::vector<fbxsdk::FbxNode*>& pLightNodes )
        {
            if( pNode )
            {
                fbxsdk::FbxNodeAttribute* pNodeAttribute = pNode->GetNodeAttribute();

                if( pNodeAttribute )
                {
                    // Light is special kind of node
                    if( pNodeAttribute->GetAttributeType() == fbxsdk::FbxNodeAttribute::eLight )
                    {
                        pLightNodes.push_back( pNode );
                    }
                }

                const int childCount = pNode->GetChildCount();

                for( int i = 0; i < childCount; ++i )
                {
                    // Find lights recursively
                    FindLights( pNode->GetChild( i ), pLightNodes );
                }
            }
        }

        inline void FindObjects( fbxsdk::FbxNode* pNode, std::vector<fbxsdk::FbxNode*>& pObjectNodes )
        {
            if( pNode )
            {
                fbxsdk::FbxNodeAttribute* pNodeAttribute = pNode->GetNodeAttribute();

                if( pNodeAttribute )
                {
                    if( pNodeAttribute->GetAttributeType() == fbxsdk::FbxNodeAttribute::eMesh )
                    {
                        pObjectNodes.push_back( pNode );
                    }
                }

                const int childCount = pNode->GetChildCount();

                for( int i = 0; i < childCount; ++i )
                {
                    // Find objects recursively
                    FindObjects( pNode->GetChild( i ), pObjectNodes );
                }
            }
        }
    };
}
