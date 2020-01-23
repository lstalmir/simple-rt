#pragma once
#include "cudaCommon.h"
#include <memory>
#include <vector>

namespace RT
{
    namespace CUDA
    {
        class DeviceMemoryDeleter
        {
        public:
            inline void operator()( void*& pDeviceMemory )
            {
                cudaFree( pDeviceMemory );

                pDeviceMemory = nullptr;
            }
        };

        class HostMemoryDeleter
        {
        public:
            inline void operator()( void* pHostMemory )
            {
                free( pHostMemory );
            }
        };

        template<typename T>
        class Array
        {
        public:
            inline Array()
            {
            }

            inline Array( size_t size )
            {
                void* pDeviceMemory = nullptr;

                CudaError::ThrowIfFailed
                ( cudaMalloc( &pDeviceMemory, sizeof( T ) * size ) );

                // Construct shared_ptr with custom deleter to call cudaFree on delete
                m_pDeviceMemory = std::shared_ptr<void>( pDeviceMemory, DeviceMemoryDeleter() );

                void* pHostMemory = malloc( sizeof( T ) * size );

                // Reserve shadow host memory
                m_pHostMemory = std::shared_ptr<void>( pHostMemory, HostMemoryDeleter() );

                m_Size = size;
            }

            inline void Update()
            {
                CudaError::ThrowIfFailed
                ( cudaMemcpyAsync( m_pDeviceMemory.get(), m_pHostMemory.get(), sizeof( T ) * (m_Size),
                    cudaMemcpyKind::cudaMemcpyHostToDevice ) );
            }

            inline void Update( size_t firstElement, size_t numElements = 1 )
            {
                const void* pSourceMemory = reinterpret_cast<const T*>(m_pHostMemory.get()) + firstElement;

                void* pDestMemory = reinterpret_cast<T*>(m_pDeviceMemory.get()) + firstElement;

                CudaError::ThrowIfFailed
                ( cudaMemcpyAsync( pDestMemory, pSourceMemory, sizeof( T ) * (numElements - firstElement),
                    cudaMemcpyKind::cudaMemcpyHostToDevice ) );
            }

            inline void Sync()
            {
                CudaError::ThrowIfFailed
                ( cudaMemcpy( m_pHostMemory.get(), m_pDeviceMemory.get(), sizeof( T ) * m_Size,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost ) );
            }

            inline void HostCopyTo( void* pDestMemory )
            {
                std::memcpy( pDestMemory, m_pHostMemory.get(), sizeof( T ) * m_Size );
            }

            inline void HostCopyTo( void* pDestMemory, size_t firstElement, size_t numElements = 1 )
            {
                const void* pSourceMemory = reinterpret_cast<const T*>(m_pHostMemory.get()) + firstElement;

                std::memcpy( pDestMemory, pSourceMemory, sizeof( T ) * (numElements - firstElement) );
            }

            inline size_t Size() const
            {
                return m_Size;
            }

            inline T* Device( int firstElement = 0 )
            {
                return reinterpret_cast<T*>(m_pDeviceMemory.get()) + firstElement;
            }

            inline const T* Device( int firstElement = 0 ) const
            {
                return reinterpret_cast<const T*>(m_pDeviceMemory.get()) + firstElement;
            }

            inline T* Host()
            {
                return reinterpret_cast<T*>(m_pHostMemory.get());
            }

            inline const T* Host() const
            {
                return reinterpret_cast<const T*>(m_pHostMemory.get());
            }

            inline T& Host( int firstElement )
            {
                return Host()[firstElement];
            }

            inline const T& Host( int firstElement ) const
            {
                return Host()[firstElement];
            }

        private:
            size_t m_Size;

            std::shared_ptr<void> m_pHostMemory;
            std::shared_ptr<void> m_pDeviceMemory;
        };

        template<typename T>
        struct ArrayView
        {
        public:
            inline ArrayView()
            {
            }

            inline ArrayView( const Array<T>& array, int index )
                : m_Array( array )
                , m_Index( index )
            {
            }

            inline void Update()
            {
                m_Array.Update( m_Index );
            }

            inline void Sync()
            {
                m_Array.Sync();
            }

            inline T* Device()
            {
                return m_Array.Device( m_Index );
            }

            inline const T* Device() const
            {
                return m_Array.Device( m_Index );
            }

            inline T& Host()
            {
                return m_Array.Host( m_Index );
            }

            inline const T& Host() const
            {
                return m_Array.Host( m_Index );
            }

        protected:
            Array<T> m_Array;
            int m_Index;
        };

        template<typename T>
        struct DataWrapper
        {
            using DataType = T;

            ArrayView<DataType> Memory;

            DataWrapper() = default;

            inline DataWrapper( const Array<DataType>& array, int index )
                : Memory( array, index )
            {
            }
        };
    }
}
