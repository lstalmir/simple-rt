#include <gtest/gtest.h>
#include "../ompScene.h"

/***************************************************************************************\

Test:
    LoadScene

Description:

\***************************************************************************************/
TEST( ompSceneTests, LoadScene )
{
    using SceneTypes = RT::OMP::SceneTypes;
    using SceneTraits = RT::MakeSceneTraits<SceneTypes, RT::OMP::SceneFunctions<SceneTypes>>;

    RT::SceneLoader loader;
    auto scene = loader.LoadScene<SceneTraits>( "multiple-cameras.fbx" );

    EXPECT_EQ( scene.Cameras.size(), 2 );
}
