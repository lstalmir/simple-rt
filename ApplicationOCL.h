#pragma once
#include "Application.h"
#include <CL/cl2.hpp>
#include <vector>

namespace RT
{
    class ApplicationOCL : public Application
    {
    public:
        ApplicationOCL( const CommandLineArguments& cmdargs );

        virtual int Run() override final;

    protected:
        cl::Platform m_clPlatform;
        cl::Device m_clDevice;
        cl::Context m_clContext;
    };
}
