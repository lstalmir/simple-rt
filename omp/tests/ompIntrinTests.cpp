#include <gtest/gtest.h>
#include "../../Optimizations.h"
#include "../../Intrin.h"
#include "../../Vec.h"

#if RT_ENABLE_AVX
TEST( ompIntrinTests, Dot_256 )
{
    const RT::vec4 As[ 2 ] = {
        RT::vec4( 0, 1, 2, 3 ),
        RT::vec4( 5, 6, 7, 8 ) };

    const RT::vec4 Bs[ 2 ] = {
        RT::vec4( 1, 2, 3, 4 ),
        RT::vec4( 10, 11, 12, 13 ) };

    const RT::vec4 Expected[ 2 ] = {
        RT::vec4( 20 ),
        RT::vec4( 304 ) };

    RT::vec4 Actual[ 2 ];

    __m256 A = _mm256_load_ps( &As->data );
    __m256 B = _mm256_load_ps( &Bs->data );

    __m256 DOT = RT::Dot( A, B );
    _mm256_store_ps( &Actual->data, DOT );

    EXPECT_NEAR( Expected[ 0 ].x, Actual[ 0 ].x, 0.001f );
    EXPECT_NEAR( Expected[ 0 ].y, Actual[ 0 ].y, 0.001f );
    EXPECT_NEAR( Expected[ 0 ].z, Actual[ 0 ].z, 0.001f );
    EXPECT_NEAR( Expected[ 0 ].w, Actual[ 0 ].w, 0.001f );

    EXPECT_NEAR( Expected[ 1 ].x, Actual[ 1 ].x, 0.001f );
    EXPECT_NEAR( Expected[ 1 ].y, Actual[ 1 ].y, 0.001f );
    EXPECT_NEAR( Expected[ 1 ].z, Actual[ 1 ].z, 0.001f );
    EXPECT_NEAR( Expected[ 1 ].w, Actual[ 1 ].w, 0.001f );
}
#endif
