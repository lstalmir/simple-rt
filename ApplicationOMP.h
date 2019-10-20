#pragma once
#include "Application.h"
#include "omp/ompScene.h"
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_priority_queue.h>

namespace RT
{
    class ApplicationOMP : public Application
    {
    public:
        ApplicationOMP( const CommandLineArguments& cmdargs );

        virtual int Run() override final;

    protected:
        Scene<RT::OMP::SceneTraits> m_Scene;

        struct RayTriangleIntersection
        {
            RT::OMP::Triangle* pTriangle;
            RT::vec4 ColorDistance;
        };
    };
}
