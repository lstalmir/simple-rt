#pragma once
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
    };
}
