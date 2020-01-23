#pragma once
#include <algorithm>
#include <assert.h>
#include <cuda_runtime.h>
#include <system_error>

#ifdef __INTELLISENSE__

// Functions available to nvcc implicitly
#include <math_functions.h>

// Helper declarations
extern const dim3 blockDim;
extern const uint3 blockIdx;
extern const uint3 threadIdx;

#endif // __INTELLISENSE__

namespace RT
{
    namespace CUDA
    {
        class CudaError : public std::system_error
        {
        public:
            inline CudaError( cudaError_t error )
                : std::system_error( error, CudaErrorCategory() )
            {
            }

            inline static void ThrowIfFailed( cudaError_t error )
            {
                // Throw CudaError if value is error
                if( error != cudaError::cudaSuccess )
                {
                    throw CudaError( error );
                }
            }

            inline static void Assert( cudaError_t error )
            {
                assert( error == cudaError::cudaSuccess );
            }

        private:
            inline static const std::error_category& CudaErrorCategory()
            {
                // Anonymous error_category class for cudaError_t
                static const class : public std::error_category
                {
                public:
                    const char* name() const noexcept override
                    {
                        return "cudaError_t";
                    }

                    std::string message( int error ) const override
                    {
                        return cudaGetErrorString( static_cast<cudaError_t>(error) );
                    }

                } cudaErrorCategory;

                return cudaErrorCategory;
            }
        };

        typedef char kilobyte[1024];
        typedef kilobyte megabyte[1024];

        inline constexpr unsigned long long operator""_B( unsigned long long value )
        {
            return value;
        }

        inline constexpr unsigned long long operator""_kB( unsigned long long value )
        {
            return value * sizeof( kilobyte );
        }

        inline constexpr unsigned long long operator""_MB( unsigned long long value )
        {
            return value * sizeof( megabyte );
        }

        struct DispatchParameters
        {
            unsigned int NumThreadsPerBlock;
            unsigned int NumBlocksPerGrid;
            unsigned int SharedMemorySize;

            inline DispatchParameters( unsigned int numInvocationsRequired )
                : _NumInvocationsRequiredHint( numInvocationsRequired )
            {
                UpdateInvocationParameters();
            }

            inline DispatchParameters& MaxThreadsPerBlock( unsigned int maxThreadsPerBlock )
            {
                _MaxThreadsPerBlockHint = maxThreadsPerBlock;
                UpdateInvocationParameters();
                return *this;
            }

            inline DispatchParameters& SharedMemoryPerThread( unsigned int sharedMemoryPerThread )
            {
                _SharedMemoryPerThreadHint = sharedMemoryPerThread;
                UpdateInvocationParameters();
                return *this;
            }

            inline DispatchParameters& LocalMemoryPerThread( unsigned int localMemoryPerThread )
            {
                _LocalMemoryPerThreadHint = localMemoryPerThread;
                UpdateInvocationParameters();
                return *this;
            }

        private:
            unsigned int _NumInvocationsRequiredHint;

            unsigned int _MaxThreadsPerBlockHint = 256;
            unsigned int _SharedMemoryPerThreadHint = 0;
            unsigned int _LocalMemoryPerThreadHint = 0;

            static constexpr unsigned int _MaxSharedMemorySize = 48_kB;
            static constexpr unsigned int _MaxLocalMemorySize = 12_kB;

            inline void UpdateInvocationParameters()
            {
                unsigned int maxThreadsPerBlock = _MaxThreadsPerBlockHint;

                if( _SharedMemoryPerThreadHint > 0 )
                {
                    // Take shared memory size into account
                    // Each thread must get at least _SharedMemoryPerThreadHint bytes
                    maxThreadsPerBlock = std::min( _MaxSharedMemorySize / _SharedMemoryPerThreadHint, maxThreadsPerBlock );
                }

                if( _LocalMemoryPerThreadHint > 0 )
                {
                    // Take local memory size into account
                    // Each thread must get at least _LocalMemoryPerThreadHint bytes
                    maxThreadsPerBlock = std::min( _MaxLocalMemorySize / _LocalMemoryPerThreadHint, maxThreadsPerBlock );
                }

                NumThreadsPerBlock = std::min( _NumInvocationsRequiredHint, maxThreadsPerBlock );
                NumBlocksPerGrid = ((_NumInvocationsRequiredHint - 1) / NumThreadsPerBlock) + 1;
                SharedMemorySize = 0;
            }
        };
    }
}
