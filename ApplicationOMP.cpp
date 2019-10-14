#include "ApplicationOMP.h"
#include <omp.h>

/***************************************************************************************\

Function:
    ApplicationOMP::ApplicationOMP

Description:
    Constructor

\***************************************************************************************/
RT::ApplicationOMP::ApplicationOMP( const RT::CommandLineArguments& cmdargs )
    : Application( cmdargs )
{
}

/***************************************************************************************\

Function:
    ApplicationOMP::Run

Description:
    Runs OpenMP implementation of the application.

\***************************************************************************************/
int RT::ApplicationOMP::Run()
{
    return 0;
}
