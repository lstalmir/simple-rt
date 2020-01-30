#pragma once
#include "Application.h"

namespace RT
{
    class ApplicationBench : public Application
    {
    public:
        ApplicationBench( const CommandLineArguments& cmdargs );

        virtual int Run() override final;

    private:
        void Bench_OMP_Ray_Intersect_Plane();
        void Bench_OMP_Ray2x2_Intersect_Plane();
    };
}
