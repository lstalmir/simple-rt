#include "SceneLoader.h"

/***************************************************************************************\

Function:
    SceneLoader::SceneLoader

Description:
    Constructor

\***************************************************************************************/
RT::SceneLoader::SceneLoader()
    : m_pManager( nullptr )
{
    // Create FBX manager instance
    m_pManager = FbxCreate<fbxsdk::FbxManager>();

    // Create IO settings
    m_pIOSettings = FbxCreate<fbxsdk::FbxIOSettings>( m_pManager.get(), "RT_FBX_IOSETTINGS" );

    m_pManager->SetIOSettings( m_pIOSettings.get() );
}

/***************************************************************************************\

Function:
    SceneLoader::LoadScene

Description:
    Loads scene from FBX file.

\***************************************************************************************/
RT::Scene RT::SceneLoader::LoadScene( std::string path )
{
    // Create importer for the scene
    auto pImporter = FbxCreate<fbxsdk::FbxImporter>( m_pManager.get(), "RT_FBX_IMPORTER" );

    if( !pImporter->Initialize( path.c_str(), -1, m_pIOSettings.get() ) )
    {
        // Initialization of the importer failed
        throw std::runtime_error( "Failed to initialize FbxImporter" );
    }

    // Create the scene
    auto pScene = FbxCreate<fbxsdk::FbxScene>( m_pManager.get(), "RT_FBX_SCENE" );

    if( !pImporter->Import( pScene.get() ) )
    {
        // Import failed
        throw std::runtime_error( "Failed to import FbxScene" );
    }

    fbxsdk::FbxNode* pRootNode = pScene->GetRootNode();

    
}
