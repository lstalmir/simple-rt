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
    testing::InitGoogleTest();
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
