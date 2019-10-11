#include "ApplicationTest.h"
#include "Plane.h"
#include "Ray.h"
#include "Vector.h"
#include <gtest/gtest.h>
#include <iostream>

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
