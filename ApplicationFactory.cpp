#include "ApplicationFactory.h"
#include "omp/ompApplication.h"
#include "cuda/cudaApplication.h"
#include "ApplicationTest.h"
#include "ApplicationBench.h"

/***************************************************************************************\

Function:
    ApplicationFactory::CreateApplication

Description:
    Creates Application instance.

\***************************************************************************************/
std::unique_ptr<RT::Application> RT::ApplicationFactory::CreateApplication( const RT::CommandLineArguments& cmdargs )
{
    switch( cmdargs.appMode )
    {
    case ApplicationMode::eTest:
    {
        return std::make_unique<ApplicationTest>( cmdargs );
    }
    case ApplicationMode::eBenchmark:
    {
        return std::make_unique<ApplicationBench>( cmdargs );
    }
    case ApplicationMode::eOpenMP:
    {
        return std::make_unique<RT::OMP::Application>( cmdargs );
    }
    case ApplicationMode::eCUDA:
    {
        return std::make_unique<RT::CUDA::Application>( cmdargs );
    }
    }
    return nullptr;
}
