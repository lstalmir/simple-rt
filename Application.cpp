#include "Application.h"

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
