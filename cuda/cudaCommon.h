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

        struct DispatchParameters
        {
            unsigned int NumThreadsPerBlock;
            unsigned int NumBlocksPerGrid;
            unsigned int SharedMemorySize;

            inline DispatchParameters( unsigned int numInvocationsRequired,
                unsigned int maxThreadsPerBlock = 32 )
            {
                NumThreadsPerBlock = std::min( numInvocationsRequired, maxThreadsPerBlock );
                NumBlocksPerGrid = ((numInvocationsRequired - 1) / NumThreadsPerBlock) + 1;
                SharedMemorySize = 0;
            }
        };
    }
}
