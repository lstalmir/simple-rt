#include <gtest/gtest.h>
#include "../ompScene.h"

/***************************************************************************************\

Test:
    LoadScene

Description:

\***************************************************************************************/
TEST( ompSceneTests, LoadScene )
{
    RT::SceneLoader loader;
    auto scene = loader.LoadScene<RT::OMP::SceneTraits>( "multiple-cameras.fbx" );

    EXPECT_EQ( scene.Cameras.size(), 2 );
}
