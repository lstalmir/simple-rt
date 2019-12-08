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

        // Add two vectors component-wise
        inline vec4 operator+( const vec4& r ) const
        {
            return vec4( x + r.x, y + r.y, z + r.z, w + r.w );
        }

        // Add values to the vector component-wise
        inline vec4& operator+=( const vec4& r )
        {
            x += r.x; y += r.y; z += r.z; w += r.w;
            return *this;
        }

        // Subtract two vectors component-wise
        inline vec4 operator-( const vec4& r ) const
        {
            return vec4( x - r.x, y - r.y, z - r.z, w - r.w );
        }

        // Add values to the vector component-wise
        inline vec4& operator-=( const vec4& r )
        {
            x -= r.x; y -= r.y; z -= r.z; w -= r.w;
            return *this;
        }

        // Negate vector
        inline vec4 operator-() const
        {
            return vec4( -x, -y, -z, -w );
        }

        // Multiply two vectors component-wise
        inline vec4 operator*( const vec4& r ) const
        {
            return vec4( x * r.x, y * r.y, z * r.z, w * r.w );
        }

        // Scale the vector
        inline vec4 operator*( float s ) const
        {
            return vec4( x * s, y * s, z * s, w * s );
        }

        // Divide two vectors component-wise
        inline vec4 operator/( const vec4& r ) const
        {
            return vec4( x / r.x, y / r.y, z / r.z, w / r.w );
        }

        // Divide vector by scalar
        inline vec4 operator/( float s ) const
        {
            return vec4( x / s, y / s, z / s, w / s );
        }

        // Compute dot product of two vectors
        inline float Dot( const vec4& r ) const
        {
            return x * r.x + y * r.y + z * r.z + w * r.w;
        }

        // Compute cross product of two vectors
        inline vec4 Cross( const vec4& r ) const
        {
            return vec4( y * r.z - z * r.y, z * r.x - x * r.z, x * r.y - y * r.x );
        }

        // Compute length of the vector taking into account all 4 components
        inline float Length4() const
        {
            return sqrtf( x * x + y * y + z * z + w * w );
        }

        // Compute length of the vector taking into account first 3 components
        inline float Length3() const
        {
            return sqrtf( x * x + y * y + z * z );
        }

        // Compute length of the vector taking into account first 2 components
        inline float Length2() const
        {
            return sqrtf( x * x + y * y );
        }

        // Compute length of the vector taking into account first component
        inline float Length1() const
        {
            return abs( x );
        }

        // Normalize the vector taking into account all 4 components
        inline void Normalize4()
        {
            const float length = Length4();
            x /= length;
            y /= length;
            z /= length;
            w /= length;
        }

        // Normalize the vector taking into account first 3 components
        inline void Normalize3()
        {
            const float length = Length3();
            x /= length;
            y /= length;
            z /= length;
            w = 0;
        }

        // Normalize the vector taking into account first 2 components
        inline void Normalize2()
        {
            const float length = Length2();
            x /= length;
            y /= length;
            z = 0;
            w = 0;
        }

        // Normalize the vector taking into account first component
        inline void Normalize1()
        {
            const uint32_t xInt = *reinterpret_cast<uint32_t*>(&x) & 0x80000000 | 0x3F800000;
            x = *reinterpret_cast<const float*>(&xInt);
            y = 0;
            z = 0;
            w = 0;
        }
    };

    inline vec4 operator*( float s, const vec4& v )
    {
        return v * s;
    }

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
