#pragma once
#include <fbxsdk.h>
#include <limits>

namespace RT
{
#ifdef RT_ENABLE_DOUBLE_MATH
    typedef double float_t;
#else
    typedef float float_t;
#endif

    struct alignas(16) vec4
    {
        union // must point to the same memory
        {
            float_t data;
            float_t x = 0;
        };
        float_t y = 0;
        float_t z = 0;
        float_t w = 0;

        inline vec4() = default;

        // Initialize vector components with one value
        inline explicit vec4( float_t x_ )
            : x( x_ ), y( x_ ), z( x_ ), w( x_ )
        {
        }

        // Initialize vector with components
        inline explicit vec4( float_t x_, float_t y_, float_t z_ = 0, float_t w_ = 0 )
            : x( x_ ), y( y_ ), z( z_ ), w( w_ )
        {
        }

        // Initialize from FBXSDK vector
        inline explicit vec4( const fbxsdk::FbxVector4& v_ )
            : x( static_cast<float_t>(v_[0]) ), y( static_cast<float_t>(v_[1]) ), z( static_cast<float_t>(v_[2]) ), w( static_cast<float_t>(v_[3]) )
        {
        }

        // Initialize from FBXSDK vector
        inline explicit vec4( const fbxsdk::FbxVector2& v_ )
            : x( static_cast<float_t>(v_[0]) ), y( static_cast<float_t>(v_[1]) ), z( 0 ), w( 0 )
        {
        }

        // Initialize from FBXSDK vector
        inline explicit vec4( const fbxsdk::FbxDouble4& v_ )
            : x( static_cast<float_t>(v_[0]) ), y( static_cast<float_t>(v_[1]) ), z( static_cast<float_t>(v_[2]) ), w( static_cast<float_t>(v_[3]) )
        {
        }

        // Initialize from FBXSDK vector
        inline explicit vec4( const fbxsdk::FbxDouble3& v_ )
            : x( static_cast<float_t>(v_[0]) ), y( static_cast<float_t>(v_[1]) ), z( static_cast<float_t>(v_[2]) ), w( 0 )
        {
        }

        // Initialize from FBXSDK vector
        inline explicit vec4( const fbxsdk::FbxDouble2& v_ )
            : x( static_cast<float_t>(v_[0]) ), y( static_cast<float_t>(v_[1]) ), z( 0 ), w( 0 )
        {
        }
    };

    struct alignas(32) vec4_2
    {
        union // must point to the same memory
        {
            float_t data;
            vec4 a;
        };
        vec4 b;

        inline vec4_2() = default;

        // Initialize vectors with one value
        inline explicit vec4_2( float_t x_ )
            : a( x_ ), b( x_ )
        {
        }

        // Initialize each vector with one value
        inline explicit vec4_2( float_t x_1, float_t x_2 )
            : a( x_1 ), b( x_2 )
        {
        }

        // Initialize vector with components
        inline explicit vec4_2( float_t x_1, float_t y_1, float_t z_1, float_t w_1, float_t x_2, float_t y_2, float_t z_2, float_t w_2 )
            : a( x_1, y_1, z_1, w_1 ), b( x_2, y_2, z_2, w_2 )
        {
        }

        // Initialize vectors
        inline explicit vec4_2( vec4 a_, vec4 b_ )
            : a( a_ ), b( b_ )
        {
        }
    };
}
