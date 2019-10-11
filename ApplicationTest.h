#pragma once
#include "Application.h"
#include <string>
#include <unordered_map>

namespace RT
{
    class ApplicationTest : public Application
    {
    public:
        ApplicationTest( const CommandLineArguments& cmdargs );

        virtual int Run() override final;
    };
}
