#include "Application.h"
#include <ostream>

/***************************************************************************************\

Function:
    Application::Application

Description:
    Constructor

\***************************************************************************************/
RT::Application::Application( const RT::CommandLineArguments& cmdargs )
    : m_CommandLineArguments( cmdargs )
    , m_SceneLoader()
{
}

/***************************************************************************************\

Function:
    Application::~Application

Description:
    Destructor

\***************************************************************************************/
RT::Application::~Application()
{
}

/***************************************************************************************\

Function:
    Application::BenchmarkBegin

Description:
    Start application benchmark

\***************************************************************************************/
void RT::Application::BenchmarkBegin()
{
    m_BenchmarkBeginTimePoint = std::chrono::high_resolution_clock::now();
}

/***************************************************************************************\

Function:
    Application::BenchmarkEnd

Description:
    Stop application benchmarking

\***************************************************************************************/
void RT::Application::BenchmarkEnd()
{
    m_BenchmarkEndTimePoint = std::chrono::high_resolution_clock::now();
}

/***************************************************************************************\

Function:
    Application::ReportBenchmarkTime

Description:

\***************************************************************************************/
void RT::Application::ReportBenchmarkTime( std::ostream& out ) const
{
    auto dt = m_BenchmarkEndTimePoint - m_BenchmarkBeginTimePoint;

    out << std::chrono::duration_cast<std::chrono::hours>(dt).count() << "h "
        << std::chrono::duration_cast<std::chrono::minutes>(dt).count() % 60 << "min "
        << std::chrono::duration_cast<std::chrono::seconds>(dt).count() % 60 << "s "
        << std::chrono::duration_cast<std::chrono::milliseconds>(dt).count() % 1000 << "ms\n";
}
