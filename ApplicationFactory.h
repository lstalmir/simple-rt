#pragma once
#include "Application.h"
#include <memory>

namespace RT
{
    class ApplicationFactory
    {
    public:
        std::unique_ptr<Application> CreateApplication( const CommandLineArguments& cmdargs );
    };
}
