#include "ApplicationFactory.h"
#include "ApplicationOCL.h"
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
        switch( cmdargs.appIntrinMode )
        {
        case ApplicationIntrinMode::eDisabled:
            return std::make_unique<RT::OMP::Application<ApplicationIntrinMode::eDisabled>>( cmdargs );

        case ApplicationIntrinMode::eEnabled:
            return std::make_unique<RT::OMP::Application<ApplicationIntrinMode::eEnabled>>( cmdargs );
        }
    }
    case ApplicationMode::eOpenCL:
    {
        return std::make_unique<ApplicationOCL>( cmdargs );
    }
    }
    return nullptr;
}
