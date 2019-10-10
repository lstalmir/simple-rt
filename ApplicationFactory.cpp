#include "ApplicationFactory.h"
#include "ApplicationOMP.h"

/***************************************************************************************\

Function:
    ApplicationFactory::CreateApplication

Description:
    Creates Application instance.

\***************************************************************************************/
std::unique_ptr<RT::Application> RT::ApplicationFactory::CreateApplication( const RT::CommandLineArguments& cmdargs )
{
    // TODO: Add CUDA application
    return std::make_unique<ApplicationOMP>( cmdargs );
}
