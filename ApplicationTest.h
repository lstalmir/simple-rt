#pragma once
#include "Application.h"

namespace RT
{
    class ApplicationTest : public Application
    {
    public:
        ApplicationTest( const CommandLineArguments& cmdargs );

        virtual int Run() override final;
    };
}
