#pragma once
#include <string>

// From getopt.h
struct option;

namespace RT
{
    enum class CommandLineOption
    {
        eInput,
        eOutput,
        eTest
    };

    class CommandLineArguments
    {
    public:
        CommandLineArguments();

        static void Help( std::ostream& out );

        static CommandLineArguments Parse( int argc, char** argv );

        static int FindOption( CommandLineOption opt, int argc, char** argv );

        bool Validate() const;
        void ReportMissingOptions( std::ostream& out ) const;

        int argc;
        char** argv;
        std::string InputFilename;
        std::string OutputFilename;
        bool Test;

    private:
        static const char s_pShortOptions[];
        static const option s_pLongOptions[];
    };
}
