#pragma once
#include "Fbx.h"
#include "Scene.h"
#include <string>

namespace RT
{
    class SceneLoader
    {
    public:
        SceneLoader();

        Scene LoadScene( std::string path );

    protected:
        FbxHandle<fbxsdk::FbxManager> m_pManager;
        FbxHandle<fbxsdk::FbxIOSettings> m_pIOSettings;
    };
}
