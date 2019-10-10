#include "Arguments.h"
#include <getopt.h>
#include <ostream>

const char RT::CommandLineArguments::s_pShortOptions[] = "i:o:";
const option RT::CommandLineArguments::s_pLongOptions[] = {
    { "input", required_argument, 0, 'i' },
    { "output", required_argument, 0, 'o' },
    { 0, 0, 0, 0 } };

/***************************************************************************************\

Function:
    CommandLineArguments::CommandLineArguments

Description:
    Constructor

\***************************************************************************************/
RT::CommandLineArguments::CommandLineArguments()
    : InputFilename()
    , OutputFilename()
{
}

/***************************************************************************************\

Function:
    CommandLineArguments::Help

Description:
    Prints help message to the output stream.

\***************************************************************************************/
void RT::CommandLineArguments::Help( std::ostream& out )
{
    out << "Available options:\n";

    const option* pOption = s_pLongOptions;

    while( pOption->name != nullptr )
    {
        out << "-" << pOption->name << ",-" << static_cast<char>( pOption->val );

        switch( pOption->has_arg )
        {
            case required_argument:
            {
                out << " <arg>";
                break;
            }

            case optional_argument:
            {
                out << " [<arg>]";
                break;
            }
        }

        out << "\n";

        // Move to the next option
        pOption++;
    }
}

/***************************************************************************************\

Function:
    CommandLineArguments::Parse

Description:
    Parses command-line arguments into CommandLineArguments structure.

\***************************************************************************************/
RT::CommandLineArguments RT::CommandLineArguments::Parse( int argc, char* const* argv )
{
    // Reset structure values
    RT::CommandLineArguments cmdargs;

    // Reset internal getopt counter to start scanning from the beginning of argv
    optind = 1;

    int opt = 0;
    while( (opt = getopt_long_only( argc, argv, s_pShortOptions, s_pLongOptions, 0 )) != -1 )
    {
        switch( opt )
        {
            // Input filename
            case 'i':
            {
                cmdargs.InputFilename = optarg;
                break;
            }

            // Output filename
            case 'o':
            {
                cmdargs.OutputFilename = optarg;
                break;
            }
        }
    }

    return cmdargs;
}

/***************************************************************************************\

Function:
    CommandLineArguments::Validate

Description:
    Validates values currently stored in the structure.

\***************************************************************************************/
bool RT::CommandLineArguments::Validate() const
{
    return !InputFilename.empty()
        && !OutputFilename.empty();
}

/***************************************************************************************\

Function:
    CommandLineArguments::ReportMissingOptions

Description:
    Report command-line options which were required, but not provided.

\***************************************************************************************/
void RT::CommandLineArguments::ReportMissingOptions( std::ostream& out ) const
{
    if( InputFilename.empty() )
    {
        out << "ERROR: Missing input filename\n";
    }

    if( OutputFilename.empty() )
    {
        out << "ERROR: Missing output filename\n";
    }
}
