#include "ApplicationFactory.h"
#include "omp/ompApplication.h"
#include "ApplicationTest.h"

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
    case ApplicationMode::eOpenMP:
    {
        return std::make_unique<RT::OMP::Application>( cmdargs );
    }
    //case ApplicationMode::eOpenCL:
    //{
    //    return std::make_unique<ApplicationOCL>( cmdargs );
    //}
    }
    return nullptr;
}
