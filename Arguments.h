#pragma once
#include <string>

// From getopt.h
struct option;

namespace RT
{
    class CommandLineArguments
    {
    public:
        CommandLineArguments();

        static void Help( std::ostream& out );

        static CommandLineArguments Parse( int argc, char* const* argv );

        bool Validate() const;
        void ReportMissingOptions( std::ostream& out ) const;

        std::string InputFilename;
        std::string OutputFilename;
        std::string Test;

    private:
        static const char s_pShortOptions[];
        static const option s_pLongOptions[];
    };
}
