#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <conduit_blueprint.hpp>
#include <mpi.h>
#include <string.h>
#include "bridge.h"

inline void DisplayUsage( char *argv[] )
{
    fprintf( stderr, "\nUsage:\n" );
    fprintf( stderr, "%s -i <xml filename> -d <data name> -h \n"
            "  sensei xml filename = -i ConduitTestBraid.xml \n"
            "  data name           = -d braid \n"
            "  basic type          = -b uniform \n"
            "  help                = -h \n"
            "\n\n",
             argv[0] );
    exit( 1 );
}

void ParseThreeValues( int values[3], const char *optarg )
{
    char *tk, delimiters[] = ",", opt[100];

    strncpy( opt, optarg, sizeof(opt) );
    tk = strtok( opt, delimiters );
    values[0] = atoi( tk );
    tk = strtok( NULL, delimiters );
    values[1] = atoi( tk );
    tk = strtok( NULL, delimiters );
    values[2] = atoi( tk );
}

typedef struct
{
    const char *xmlFilename;   // Input sensei xml filename.
    const char *data;          // data name to create.
    const char *basic;         // basic data set has a coords type name.
    int         dims[3];       // dimentions x,y,z.
} commandLineArgs;

commandLineArgs cla;
const char *commandLineOptions = "i:d:b:x:h";

void ParseCommandLine( int argc, char* argv[] )
{
    int c;  // char return by getopt.

    // clear command line args struct.
    memset( &cla, 0, sizeof(cla) );

    // Process command line arguments.
    while( (c = getopt(argc, argv, commandLineOptions)) != -1 )
    {
        switch( c )
        {
            case 'b':
                cla.basic = optarg;
                break;
            case 'd':
                cla.data = optarg;
                break;
            case 'i':
                cla.xmlFilename = optarg;
                break;
            case 'x':
                ParseThreeValues( cla.dims, optarg );
                break;
            case 'h':
                DisplayUsage( argv );
                break;
        }
    }

    // check that we have the arguments required.
    if( cla.xmlFilename == NULL || cla.data == NULL )
        DisplayUsage( argv );

    // Check if we can read the file.
    if( access(cla.xmlFilename, R_OK) )
    {
        fprintf( stderr, "\nCould not read the input xlm file: '%s'.\n\n", cla.xmlFilename );
        exit( -1 );
    }

    // If data is Basic, that the basic and dims are set.
    if( (cla.data[0] == 'B' || cla.data[0] == 'b') && (cla.data[0] == 'A' || cla.data[0] == 'a'))
        if( cla.basic == NULL || cla.dims[0] == 0 )
            DisplayUsage( argv );
}

int main( int argc, char* argv[] )
{
  int rank, size;

  ParseCommandLine( argc, argv );

  MPI_Init( &argc, &argv );
  MPI_Comm_rank( MPI_COMM_WORLD, &rank );
  MPI_Comm_size( MPI_COMM_WORLD, &size );

  SenseiInitialize( cla.xmlFilename );

  // This is where you would have the data loop.
    // But for this sample we will just create a sample data set.
    conduit::Node data;

    switch( cla.data[0] )
    {
        case 'b':
        case 'B':
            switch( cla.data[1] )
            {
                case 'a':
                case 'A':
                    conduit::blueprint::mesh::examples::basic( cla.basic, cla.dims[0], cla.dims[1], cla.dims[2], data );
                    break;
                case 'r':
                case 'R':
                    conduit::blueprint::mesh::examples::braid( "rectilinear", 20, 20, 20, data );
                    break;
            }
            break;

        case 'j':
        case 'J':
            conduit::blueprint::mesh::examples::julia( 10, 10, 0, 8, 0, 8, 0.285, 0.01, data );
            break;

        case 's':
        case 'S':
            conduit::blueprint::mesh::examples::spiral( 7, data );
            break;

        case 'p':
        case 'P':
            conduit::blueprint::mesh::examples::polytess( 24, data );
            break;

        default:
            fprintf( stderr, "\nData set not supported: '%s'.\n\n", cla.data );
            DisplayUsage( argv );
            return( -1 );
    }
    // Debug print data set.
    //data.print();

    conduit::Node verify_info;
    if( !conduit::blueprint::mesh::verify(data, verify_info) )
    {
        fprintf( stderr, "\nConduit Blueprint Mesh Verify Failed.\n\n" );
        verify_info.print();
    }

    // Call this to Analyze or Visualize the data.
    SenseiAnalyze( data );
  // End of the data loop.

  // Clean up.
  SenseiFinalize();
  MPI_Finalize();

  return( 0 );
}

