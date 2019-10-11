#include "Arguments.h"
#include "ApplicationFactory.h"
#include <errno.h>
#include <iostream>
#include <memory>

/***************************************************************************************\

Function:
    main

Description:
    Entry-point to the application.

\***************************************************************************************/
int main( int argc, char* argv[] )
{
    // Command-line arguments for the RT application
    RT::CommandLineArguments cmdargs = RT::CommandLineArguments::Parse( argc, argv );

    if( !cmdargs.Validate() )
    {
        cmdargs.ReportMissingOptions( std::cout );
        cmdargs.Help( std::cout );
        return EINVAL;
    }

    RT::ApplicationFactory applicationFactory;

    std::unique_ptr<RT::Application> pApplication = applicationFactory.CreateApplication( cmdargs );

    // Run the application
    return pApplication->Run();
}
