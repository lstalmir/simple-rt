#pragma once
#include "Arguments.h"

namespace RT
{
    class Application
    {
    public:
        Application( const CommandLineArguments& cmdargs );
        virtual ~Application();

        virtual int Run() = 0;

    protected:
        CommandLineArguments m_CommandLineArguments;
    };
}
