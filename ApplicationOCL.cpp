#include "ApplicationOCL.h"
#include <iostream>

RT::ApplicationOCL::ApplicationOCL( const RT::CommandLineArguments& cmdargs )
    : Application( cmdargs )
    , m_clPlatform( cl::Platform::get() )
    , m_clDevice()
    , m_clContext( nullptr )
{
    std::vector<cl::Device> allDevices;
    m_clPlatform.getDevices( CL_DEVICE_TYPE_ALL, &allDevices );

    uint32_t i = 0;

    // Helper functions
    auto cl_device_type_to_string = []( cl_device_type type )
    {
        switch( type )
        {
        case CL_DEVICE_TYPE_CPU: return "CPU";
        case CL_DEVICE_TYPE_GPU: return "GPU";
        case CL_DEVICE_TYPE_ACCELERATOR: return "ACC";
        case CL_DEVICE_TYPE_CUSTOM: return "CUS";
        }
        return "???";
    };
    auto cl_device_available_to_string = []( cl_bool available )
    {
        return !available ? "N/A" : "   ";
    };

    // Enumerate to console
    std::cout << "Enumerate OpenCL devices:\n";

    for( cl::Device device : allDevices )
    {
        std::cout << "    (" << i << ")  "
            << cl_device_type_to_string( device.getInfo<CL_DEVICE_TYPE>() ) << "  "
            << cl_device_available_to_string( device.getInfo<CL_DEVICE_AVAILABLE>() ) << "  "
            << device.getInfo<CL_DEVICE_NAME>() << "\n";
        i++;
    }

    // Select device

}

int RT::ApplicationOCL::Run()
{
    return 0;
}
