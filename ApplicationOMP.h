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

        struct RayTriangleIntersectionTask
        {
            int i;
            const RT::OMP::Ray* pRay;
            const RT::OMP::Triangle* pTriangle;
        };

        tbb::concurrent_queue<RayTriangleIntersectionTask> m_Tasks;
    };
}
