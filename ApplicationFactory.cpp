#include "ApplicationFactory.h"
#include "ApplicationOMP.h"
#include "ApplicationTest.h"

/***************************************************************************************\

Function:
    ApplicationFactory::CreateApplication

Description:
    Creates Application instance.

\***************************************************************************************/
std::unique_ptr<RT::Application> RT::ApplicationFactory::CreateApplication( const RT::CommandLineArguments& cmdargs )
{
    if( !cmdargs.Test.empty() )
    {
        // Test application
        return std::make_unique<ApplicationTest>( cmdargs );
    }

    // TODO: Add CUDA application
    return std::make_unique<ApplicationOMP>( cmdargs );
}
