#pragma once
#include <stdio.h>
#include <png.h>
#include <system_error>
#include <vector>
#include <memory>
#include <string.h>

#if defined _WIN32 || defined WIN32 || defined _WIN64
#define FOPEN( handle, filename, mode ) fopen_s( &handle, filename, mode )
#else
#define FOPEN( handle, filename, mode ) handle = fopen( filename, mode )
#endif

namespace RT
{
    class File
    {
    public:
        inline File( const char* filename, const char* mode = "rb" )
            : m_pHandle( nullptr )
        {
            FOPEN( m_pHandle, filename, mode );

            if( m_pHandle == nullptr )
            {
                throw std::runtime_error( "Failed to open file " + std::string( filename ) );
            }
        }

        inline File( FILE* file )
            : m_pHandle( file )
        {
        }

        inline virtual ~File()
        {
            fclose( m_pHandle );
        }

        operator FILE* () const
        {
            return m_pHandle;
        }

    protected:
        FILE* m_pHandle;
    };

    class PngReader
    {
    public:
        inline PngReader( FILE* file )
        {
            m_pPng = png_create_read_struct( PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr );

            if( m_pPng == nullptr )
            {
                throw std::runtime_error( "Failed to initialize PNG reader" );
            }

            m_pInfo = png_create_info_struct( m_pPng );

            if( m_pInfo == nullptr )
            {
                png_destroy_read_struct( &m_pPng, nullptr, nullptr );
                throw std::runtime_error( "Failed to initialize PNG reader info" );
            }

            png_init_io( m_pPng, file );
            png_set_sig_bytes( m_pPng, 0 );

            png_read_info( m_pPng, m_pInfo );

            m_Width = png_get_image_width( m_pPng, m_pInfo );
            m_Height = png_get_image_height( m_pPng, m_pInfo );
            m_BitDepth = png_get_bit_depth( m_pPng, m_pInfo );
            m_Channels = png_get_channels( m_pPng, m_pInfo );
            m_ColorType = png_get_color_type( m_pPng, m_pInfo );

            std::vector<png_bytep> rowPointers( m_Height );
            size_t rowbytes = png_get_rowbytes( m_pPng, m_pInfo );

            for( auto& pointer : rowPointers )
                pointer = new png_byte[rowbytes];

            png_read_image( m_pPng, rowPointers.data() );

            m_pData = new png_byte[png_get_rowbytes( m_pPng, m_pInfo ) * rowPointers.size()];

            for( size_t i = 0; i < rowPointers.size(); ++i )
                memcpy( m_pData + i * rowbytes, rowPointers[i], rowbytes );

            for( auto& pointer : rowPointers )
                delete[] pointer;
        }

        inline ~PngReader()
        {
            png_destroy_read_struct( &m_pPng, &m_pInfo, nullptr );
            delete[] m_pData;
        }

        inline unsigned width() const { return m_Width; }
        inline unsigned height() const { return m_Height; }
        inline unsigned bits_per_pixel() const { return m_BitDepth * m_Channels; }
        inline unsigned bit_depth() const { return m_BitDepth; }
        inline unsigned channels() const { return m_Channels; }
        inline unsigned color_type() const { return m_ColorType; }
        inline size_t size() const { return m_DataSize; }

        template<typename T = void>
        inline const T* data() const { return reinterpret_cast<const T*>(m_pData); }

        template<typename T = void>
        inline T* data() { return reinterpret_cast<T*>(m_pData); }

    protected:
        png_structp m_pPng;
        png_infop   m_pInfo;
        uint32_t    m_Width;
        uint32_t    m_Height;
        uint32_t    m_BitDepth;
        uint32_t    m_Channels;
        uint32_t    m_ColorType;
        size_t      m_DataSize;
        png_bytep   m_pData;
    };

    class PngWriter
    {
    public:
        inline PngWriter( FILE* file )
        {
            m_pPng = png_create_write_struct( PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr );

            if( m_pPng == nullptr )
            {
                throw std::runtime_error( "Failed to initialize PNG writer" );
            }

            m_pInfo = png_create_info_struct( m_pPng );

            if( m_pInfo == nullptr )
            {
                png_destroy_write_struct( &m_pPng, nullptr );
                throw std::runtime_error( "Failed to initialize PNG writer info" );
            }

            png_init_io( m_pPng, file );
        }

        inline ~PngWriter()
        {
            png_destroy_write_struct( &m_pPng, &m_pInfo );
        }

        inline void set_width( unsigned width ) { m_Width = width; }
        inline void set_height( unsigned height ) { m_Height = height; }
        inline void set_bit_depth( unsigned depth ) { m_BitDepth = depth; }
        inline void set_channels( unsigned channels ) { m_Channels = channels; }
        inline void set_color_type( unsigned type ) { m_ColorType = type; }

        inline void write_data( png_byte* data )
        {
            png_set_IHDR(
                m_pPng,
                m_pInfo,
                m_Width,
                m_Height,
                m_BitDepth,
                m_ColorType,
                PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_BASE,
                PNG_FILTER_TYPE_BASE );

            png_write_info( m_pPng, m_pInfo );

            std::vector<png_bytep> rowPointers( m_Height );
            size_t rowbytes = png_get_rowbytes( m_pPng, m_pInfo );

            for( size_t i = 0; i < rowPointers.size(); ++i )
            {
                rowPointers[i] = new png_byte[rowbytes];
                memcpy( rowPointers[i], data + i * rowbytes, rowbytes );
            }

            png_write_image( m_pPng, rowPointers.data() );
            png_write_end( m_pPng, m_pInfo );

            for( auto& pointer : rowPointers )
                delete[] pointer;
        }

    protected:
        png_structp m_pPng;
        png_infop   m_pInfo;
        uint32_t    m_Width;
        uint32_t    m_Height;
        uint32_t    m_BitDepth;
        uint32_t    m_Channels;
        uint32_t    m_ColorType;
    };
}
