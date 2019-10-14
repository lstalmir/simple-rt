#pragma once
#include <string>

// From getopt.h
struct option;

namespace RT
{
    enum class ApplicationMode
    {
        eUndefined,
        eTest,
        eOpenMP,
        eOpenCL
    };

    enum class OpenCLDeviceType
    {
        eUndefined,
        eCPU,
        eGPU,
        eAccelerator
    };

    class CommandLineArguments
    {
    public:
        CommandLineArguments();

        static void Help( std::ostream& out );

        static CommandLineArguments Parse( int argc, char** argv, std::ostream& err );

        bool Validate( std::ostream& err ) const;

        int                 argc;
        char**              argv;
        std::string         appInputFilename;
        std::string         appOutputFilename;
        ApplicationMode     appMode;
        OpenCLDeviceType    oclDeviceType;

    private:
        static const char s_pShortOptions[];
        static const option s_pLongOptions[];
    };
}
