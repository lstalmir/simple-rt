#pragma once
#include "cudaCommon.h"
#include <cuda_runtime.h>

namespace RT
{
    namespace CUDA
    {
        struct Mutex
        {
            inline __device__ static void EnterCriticalSection( int& mutex )
            {
                #if __CUDA_ARCH__ >= 200
                while( atomicCAS( &mutex, 0, 1 ) != 0 );
                #endif
            }

            inline __device__ static void LeaveCriticalSection( int& mutex )
            {
                #if __CUDA_ARCH__ >= 200
                atomicExch( &mutex, 0 );
                #endif
            }
        };
    }
}
