#pragma once
#include <CL/cl2.hpp>

namespace RT::ocl
{
    struct RayKernels
    {
        RayKernels();
        RayKernels( cl::Context context );

        cl::Kernel IntersectPlane;

    private:
        cl::Program m_clRayProgram;

        static const char* const s_pKernels;
    };
}
