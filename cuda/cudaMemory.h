#pragma once
#include "cudaCommon.h"
#include <memory>
#include <vector>

namespace RT
{
    namespace CUDA
    {
        class MemoryDeleter
        {
        public:
            inline void operator()( void* pDeviceMemory )
            {
                CudaError::Assert
                ( cudaFree( pDeviceMemory ) );
            }
        };

        template<typename T>
        class Memory
        {
        public:
            inline Memory() : Memory( true )
            {
            }

            inline virtual void UpdateDeviceMemory( const T* source )
            {
                const void* pSourceMemory = reinterpret_cast<const void*>(source);

                CudaError::ThrowIfFailed
                ( cudaMemcpy( m_pDeviceMemory.get(), pSourceMemory, sizeof( T ),
                    cudaMemcpyKind::cudaMemcpyHostToDevice ) );
            }

            inline virtual void GetDeviceMemory( T* dest )
            {
                void* pDestMemory = reinterpret_cast<void*>(dest);

                CudaError::ThrowIfFailed
                ( cudaMemcpy( pDestMemory, m_pDeviceMemory.get(), sizeof( T ),
                    cudaMemcpyKind::cudaMemcpyDeviceToHost ) );
            }

            inline virtual T* Data()
            {
                return reinterpret_cast<T*>(m_pDeviceMemory.get());
            }

            inline virtual const T* Data() const
            {
                return reinterpret_cast<const T*>(m_pDeviceMemory.get());
            }

        protected:
            std::shared_ptr<void> m_pDeviceMemory;

            // Custom deleter calling cudaFree
            static const MemoryDeleter& DeviceMemoryDeleter()
            {
                static const MemoryDeleter deleter;
                return deleter;
            }

            inline Memory( bool alloc )
            {
                if( alloc )
                {
                    // Allocate device memory
                    AllocDeviceMemory( sizeof( T ) );
                }
            }

            inline void AllocDeviceMemory( size_t byteSize )
            {
                void* pDeviceMemory = nullptr;

                CudaError::ThrowIfFailed
                ( cudaMalloc( &pDeviceMemory, byteSize ) );

                // Construct shared_ptr with custom deleter to call cudaFree on delete
                m_pDeviceMemory = std::shared_ptr<void>( pDeviceMemory,
                    DeviceMemoryDeleter() );
            }
        };

        template<typename T>
        class Array : public Memory<T>
        {
        public:
            inline Array() : Memory<T>( false )
                , m_Size( 0 )
            {
            }

            inline Array( size_t size ) : Memory<T>( false )
                , m_Size( size )
            {
                AllocDeviceMemory( sizeof( T ) * size );
            }

            inline virtual void UpdateDeviceMemory( const T* source ) override
            {
                // Update all elements
                UpdateDeviceMemory( source, m_Size );
            }

            inline virtual void UpdateDeviceMemory( const T* source, size_t elements )
            {
                const void* pSourceMemory = reinterpret_cast<const void*>(source);

                CudaError::ThrowIfFailed
                ( cudaMemcpy( m_pDeviceMemory.get(), pSourceMemory, sizeof( T ) * elements,
                    cudaMemcpyKind::cudaMemcpyHostToDevice ) );
            }

            inline virtual void UpdateDeviceMemory( const T* source, size_t offset, size_t elements )
            {
                const void* pSourceMemory = reinterpret_cast<const void*>(source);

                void* pDestMemory = reinterpret_cast<char*>(m_pDeviceMemory.get())
                    + sizeof( T ) * offset;

                CudaError::ThrowIfFailed
                ( cudaMemcpy( pDestMemory, pSourceMemory, sizeof( T ) * elements,
                    cudaMemcpyKind::cudaMemcpyHostToDevice ) );
            }

            inline virtual void GetDeviceMemory( T* dest ) override
            {
                // Get all elements
                GetDeviceMemory( dest, m_Size );
            }

            inline virtual void GetDeviceMemory( T* dest, size_t elements )
            {
                void* pDestMemory = reinterpret_cast<void*>(dest);

                CudaError::ThrowIfFailed
                ( cudaMemcpy( pDestMemory, m_pDeviceMemory.get(), sizeof( T ) * elements,
                    cudaMemcpyKind::cudaMemcpyDeviceToHost ) );
            }

            inline size_t Size() const
            {
                return m_Size;
            }

            inline virtual T* Data() override
            {
                return Data( 0 );
            }

            inline virtual T* Data( size_t firstElement )
            {
                return reinterpret_cast<T*>(m_pDeviceMemory.get()) + firstElement;
            }

            inline virtual const T* Data() const override
            {
                return Data( 0 );
            }

            inline virtual const T* Data( size_t firstElement ) const
            {
                return reinterpret_cast<const T*>(m_pDeviceMemory.get()) + firstElement;
            }

            inline virtual std::vector<T> GetDeviceMemory()
            {
                std::vector<T> data( m_Size );

                GetDeviceMemory( data.data(), m_Size );

                return data;
            }

        protected:
            size_t m_Size;
        };

        template<typename T>
        struct ArrayView
        {
        public:
            inline ArrayView()
            {
            }

            inline ArrayView( const Array<T>& array, int index )
                : DeviceMemory( array )
                , Index( index )
            {
            }

            inline virtual void UpdateDeviceMemory( const T* source )
            {
                DeviceMemory.UpdateDeviceMemory( source, Index, 1 );
            }

            inline virtual T* Data()
            {
                return DeviceMemory.Data( Index );
            }

            inline virtual const T* Data() const
            {
                return DeviceMemory.Data( Index );
            }

        protected:
            Array<T> DeviceMemory;
            int Index;
        };

        template<typename T>
        struct DataWrapper
        {
            using DataType = T;

            ArrayView<DataType> DeviceMemory;
            DataType HostMemory;

            DataWrapper() = default;

            inline DataWrapper( const Array<DataType>& array, int index )
                : DeviceMemory( array, index )
            {
            }

            inline void UpdateDeviceMemory()
            {
                DeviceMemory.UpdateDeviceMemory( &HostMemory );
            }
        };
    }
}
