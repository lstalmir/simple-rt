#pragma once
#include "Application.h"

namespace RT
{
    class ApplicationOMP : public Application
    {
    public:
        ApplicationOMP( const CommandLineArguments& cmdargs );

        virtual int Run() override final;
    };
}
