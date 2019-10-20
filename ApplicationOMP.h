#pragma once
#include "Application.h"
#include "omp/ompScene.h"

namespace RT
{
    class ApplicationOMP : public Application
    {
    public:
        ApplicationOMP( const CommandLineArguments& cmdargs );

        virtual int Run() override final;

    protected:
        Scene<OMP::SceneTraits> m_Scene;
    };
}
