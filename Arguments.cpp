#include "Arguments.h"
#include <ostream>

#if __has_include(<getopt.h>)
#include <getopt.h>
#else
#include <string.h>

#define no_argument 0
#define required_argument 1
#define optional_argument 2

struct option
{
    const char* name;
    int has_arg;
    int* flag;
    int val;
};

namespace
{
    static char* optarg = nullptr;
    static int optind = 1;
    static int opterr = 0;
    static int optopt = 0;

    /***********************************************************************************\

    Function:
        _internal_getopt_init

    Description:
        Initialize global getopt state.

    \***********************************************************************************/
    inline void _internal_getopt_init()
    {
        optarg = nullptr;
        optind = 1;
        opterr = 0;
        optopt = 0;
    }

    /***********************************************************************************\

    Function:
        _internal_getopt_missing_argument

    Description:
        Report a missing argument for the option and move to the next option.

    \***********************************************************************************/
    inline int _internal_getopt_missing_argument( const char* opt, const char* shortopts )
    {
        // Missing argument for the option
        fprintf( stderr, "option '%s' requires an argument\n", opt );
        if( shortopts && *shortopts == ':' ) return ':';
        else return '?';
    }

    /***********************************************************************************\

    Function:
        _internal_getopt_unrecognized

    Description:
        Report not recognized option.

    \***********************************************************************************/
    inline int _internal_getopt_unrecognized( const char* opt )
    {
        // Move to the next option
        optind++;
        fprintf( stderr, "unrecognized option '%s'\n", opt );
        return '?';
    }

    /***********************************************************************************\

    Function:
        getopt_long_only

    Description:
        Parse next command-line argument.

    \***********************************************************************************/
    inline int getopt_long_only( int argc, char* const* argv, const char* shortopts,
        const option* longopts, int* longind )
    {
        if( optind == 0 ) _internal_getopt_init();

        int retval = -1;

        // Check if there are any options to parse
        if( optind >= argc ) return retval;

        const char* opt = argv[optind];

        // Option must start with '-' character
        if( opt[0] != '-' ) return retval;
        if( opt[1] == '-' ) opt += 2;
        else opt++;

        // Single '-' and '--' strings are not options
        if( opt[0] == 0 ) return retval;

        optarg = nullptr;
        opterr = 0;
        optopt = 0;

        retval = 0;

        // Search long options first
        for( const option* longopt = longopts; longopt && longopt->name; ++longopt )
        {
            if( strcmp( longopt->name, opt ) == 0 )
            {
                // Valid option found, advance the pointer
                optind++;

                // If longopt->flag is not NULL, store val at that address instead of returning it
                if( !longopt->flag ) retval = longopt->val;
                else *longopt->flag = longopt->val;

                if( longopt->has_arg != no_argument )
                {
                    if( optind == argc && longopt->has_arg == required_argument )
                    {
                        // Commandline ends unexpectedly
                        return _internal_getopt_missing_argument( argv[optind], shortopts );
                    }

                    if( optind < argc )
                    {
                        // Store pointer to the option argument in the optarg and advance the pointer
                        optarg = argv[optind++];
                    }
                }
                return retval;
            }
        }

        // Check if option is unrecognized long option
        if( opt[1] != 0 ) return _internal_getopt_unrecognized( argv[optind] );

        // Search short options
        for( const char* shortopt = shortopts; shortopt && *shortopt; ++shortopt )
        {
            // Current character in shortopts may be extension to the previous option
            if( *shortopt == ':' ) continue;

            if( *shortopt == *opt )
            {
                // Valid option found, advance the pointer
                optind++;

                if( shortopt[1] == ':' )
                {
                    if( optind == argc )
                    {
                        // Commandline ends unexpectedly
                        return _internal_getopt_missing_argument( argv[optind], shortopts );
                    }

                    // Store pointer to the option argument in the optarg and advance the pointer
                    optarg = argv[optind++];
                }

                return *opt;
            }
        }

        // Option not found
        return _internal_getopt_unrecognized( argv[optind] );
    }
}

#endif

const char RT::CommandLineArguments::s_pShortOptions[] = "i:o:t";
const option RT::CommandLineArguments::s_pLongOptions[] = {
    { "input", required_argument, 0, 'i' },
    { "output", required_argument, 0, 'o' },
    { "test", no_argument, 0, 't' },
    { 0, 0, 0, 0 } };

/***************************************************************************************\

Function:
    CommandLineArguments::CommandLineArguments

Description:
    Constructor

\***************************************************************************************/
RT::CommandLineArguments::CommandLineArguments()
    : argc( 0 )
    , argv( nullptr )
    , InputFilename()
    , OutputFilename()
    , Test( false )
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
RT::CommandLineArguments RT::CommandLineArguments::Parse( int argc, char** argv )
{
    // Reset structure values
    RT::CommandLineArguments cmdargs;

    cmdargs.argc = argc;
    cmdargs.argv = argv;

    // Reset internal getopt counter to start scanning from the beginning of argv
    optind = 1;

    int opt = 0;
    while( (opt = getopt_long_only( argc, argv, s_pShortOptions, s_pLongOptions, nullptr )) != -1 )
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

            // Test
            case 't':
            {
                cmdargs.Test = true;
                break;
            }
        }
    }

    return cmdargs;
}

/***************************************************************************************\

Function:
    CommandLineArguments::FindOption

Description:
    Find index of the option in the command-line array.

\***************************************************************************************/
int RT::CommandLineArguments::FindOption( RT::CommandLineOption opt, int argc, char** argv )
{
    // Reset internal getopt counter to start scanning from the beginning of argv
    optind = 1;

    int i = 0;
    while( (i = getopt_long_only( argc, argv, s_pShortOptions, s_pLongOptions, nullptr )) != -1 )
    {
        if( (i == 'i' && opt == CommandLineOption::eInput) ||
            (i == 'o' && opt == CommandLineOption::eOutput) ||
            (i == 't' && opt == CommandLineOption::eTest) )
        {
            return optind;
        }
    }

    return -1;
}

/***************************************************************************************\

Function:
    CommandLineArguments::Validate

Description:
    Validates values currently stored in the structure.

\***************************************************************************************/
bool RT::CommandLineArguments::Validate() const
{
    return (Test)
        || (!InputFilename.empty() && !OutputFilename.empty());
}

/***************************************************************************************\

Function:
    CommandLineArguments::ReportMissingOptions

Description:
    Report command-line options which were required, but not provided.

\***************************************************************************************/
void RT::CommandLineArguments::ReportMissingOptions( std::ostream& out ) const
{
    if( !Test )
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
}
