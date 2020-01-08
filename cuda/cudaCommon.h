#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <system_error>

// WA for C++11 nvcc
#define NAMESPACE_RT_CUDA   namespace RT { namespace CUDA

#define DISPATCH( PARAMS )  <<<(PARAMS).NumThreadsPerGroup,(PARAMS).NumGroupsPerBlock,(PARAMS).SharedMemorySize>>>

NAMESPACE_RT_CUDA
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
        unsigned int NumThreadsPerGroup;
        unsigned int NumGroupsPerBlock;
        unsigned int SharedMemorySize;

        inline DispatchParameters( unsigned int numInvocationsRequired,
            unsigned int maxThreadsPerGroup = 32 )
        {
            NumThreadsPerGroup = std::min( numInvocationsRequired, maxThreadsPerGroup );
            NumGroupsPerBlock = ((numInvocationsRequired - 1) / NumThreadsPerGroup) + 1;
        }
    };
}}
