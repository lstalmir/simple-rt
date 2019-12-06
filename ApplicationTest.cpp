#include "ApplicationTest.h"
#include <gtest/gtest.h>

/***************************************************************************************\

Function:
    ApplicationTest::ApplicationTest

Description:
    Constructor

\***************************************************************************************/
RT::ApplicationTest::ApplicationTest( const RT::CommandLineArguments& cmdargs )
    : Application( cmdargs )
{
    testing::InitGoogleTest( 0, static_cast<char**>(nullptr) );
}

/***************************************************************************************\

Function:
    ApplicationTest::Run

Description:
    Run selected tests.

\***************************************************************************************/
int RT::ApplicationTest::Run()
{
    return RUN_ALL_TESTS();
}
