#pragma once
#include "Arguments.h"
#include "Scene.h"
#include <chrono>

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
        SceneLoader m_SceneLoader;

        std::chrono::high_resolution_clock::time_point m_BenchmarkBeginTimePoint;
        std::chrono::high_resolution_clock::time_point m_BenchmarkEndTimePoint;

        void BenchmarkBegin();
        void BenchmarkEnd();

        void ReportBenchmarkTime( std::ostream& out ) const;
    };
}
