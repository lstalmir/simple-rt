#pragma once
#include "../Optimizations.h"
#include "../Application.h"
#include "../File.h"
#include "cudaScene.h"

namespace RT::CUDA
{
    class Application : public RT::Application
    {
        using SceneTypes = RT::CUDA::SceneTypes;
        using SceneTraits = RT::MakeSceneTraits<SceneTypes, RT::CUDA::SceneFunctions<SceneTypes>>;

    public:
        Application( const RT::CommandLineArguments& cmdargs )
            : RT::Application( cmdargs )
        {
        }

        inline virtual int Run() override
        {
            return -1;
        }

    protected:
        RT::Scene<SceneTraits> m_Scene;

    };
}
