#pragma once
#include "Arguments.h"
#include "Scene.h"

namespace RT
{
    class Application
    {
    public:
        Application( const CommandLineArguments& cmdargs );
        virtual ~Application();

        virtual int Run() = 0;

    protected:
        CommandLineArguments m_CommandLineArguments;
        SceneLoader m_SceneLoader;
    };
}
