/*
DATA GENERATOR SCRIPT
[ function:  R = sqrt((x-xc)^2 + (y-yc)^2) ]

Creates computational domain over specified dimenstions, adds white noise & sends the data
over SENSEI to fft analysis endpoint.

$ mpirun -np p ./data_generator x y w t xml
    p = no. of processors for parallel execution
    x = x_dimension
    y = y_dimension
    w = change the original value by what %? (0-100)
    t = what % of total values to change? (0-100)
    xml = path to xml configuration file for fft

// Contributions:
Sudhanshu Kulkarni, E. Wes Bethel, B. Loring
2022
*/

#include <iostream>
#include <vector>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <algorithm>
#include <random>
#include <omp.h>
#include <chrono>
#include <string>
#include <set>
#include <memory>

#include <svtkDataArray.h>
#include <svtkFieldData.h>
#include <svtkDataObject.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkSmartPointer.h>

#include <svtkDoubleArray.h>
#include <svtkImageData.h>
#include <svtkPointData.h>

#include "ConfigurableAnalysis.h"
#include "MeshMetadata.h"
#include "MeshMetadataMap.h"
#include "AnalysisAdaptor.h"
#include "SVTKDataAdaptor.h"
#include "Error.h"


using namespace std;

/* Writing output instead of SENSEI Endpoint 
NOT USED ANY MORE, included for debugging. */
void
writeStuff(long int data_size, std::string outputFileName, std::vector<double> output_data_floats = {}, std::vector<int> output_labels = {})
{   
    printf("\nWriting output -> %s \t(size %ld)\n", outputFileName.c_str(), data_size);
    
    // open the input file
    FILE *f = fopen(outputFileName.c_str(), "w");
    if (f == NULL)
    {
        perror(" Problem opening output file ");
        return ;
    }

    // this code writes out the output as floats rather than converting to ubyte
    if (!output_data_floats.empty()){
        fwrite((const void *)output_data_floats.data(), sizeof(double), data_size, f);
    }
    else if (!output_labels.empty()){
        fwrite((const void *)output_labels.data(), sizeof(int), data_size, f);
    }
        
    fclose(f);
}


/* Reading input instead of SENSEI Endpoint */
std::vector <double>
readStuff(long int data_size, std::string inputFileName)
{   
    printf("\nReading input from -> %s \t(size %ld)\n", inputFileName.c_str(), data_size);
    
    // vector to hold input data
    std::vector<double> input_data_floats(data_size);

    // open the input file
    FILE *f = fopen(inputFileName.c_str(), "r");
    if (f == NULL)
    {
        perror(" Problem opening output file ");
        return input_data_floats;
    }

    // this code reads the input as floats rather than converting to ubyte
    fread((void *)input_data_floats.data(), sizeof(double), data_size, f);
        
    fclose(f);

    // DEBUG:
    // printf("\nINPUT DATA:\n");
    // for (int y = 0; y < sqrt(data_size); ++y) {
    //     for (int x = 0; x < sqrt(data_size); ++x)
    //     {
    //         printf("%f\t", input_data_floats[y*sqrt(data_size) + x]);
    //     }
    //     printf("\n");
    // }

    return input_data_floats;
}

/* Add w % noise to x */
double
noisy(double x, int w)
{
    static random_device rd;
    static mt19937 gen(rd());
    double range_min = x-(w*x/100), range_max = x+(w*x/100);
    uniform_real_distribution<double> dis(range_min, range_max);

    return dis(gen);
}

/* Execute SENSEI */
void
send_with_sensei(vector <double> data, double xDim, double yDim, string const& xmlFileName)
{
    // Initialize XML
    sensei::ConfigurableAnalysis *fft_endpoint = sensei::ConfigurableAnalysis::New();
    fft_endpoint->Initialize(xmlFileName);

    // Setting up double array
    unsigned int nVals = data.size();
    
    svtkDoubleArray *da = svtkDoubleArray::New();
    da->SetNumberOfTuples(nVals);
    da->SetName("dataArray");
    for (unsigned int i = 0; i < nVals; ++i)
        *da->GetPointer(i) = data[i];

    // DEBUG:
    // printf("\n-> testFFT :: Setting up data in svtkDoubleArray");

    // Setting up Image Data
    svtkImageData *im = svtkImageData::New();
    im->SetDimensions(xDim, yDim, 1);
    im->GetPointData()->AddArray(da);
    // im->SetOrigin(0, -90, 0);
    // im->SetSpacing(0.5, 0.25, 0);
    da->Delete();

    // DEBUG:
    // printf("\n-> testFFT :: Setting up data in svtkImageData");

    // Setting up MultiBlockDataSet    
    svtkMultiBlockDataSet* mb = svtkMultiBlockDataSet::New();
    mb->SetNumberOfBlocks(1);
    mb->SetBlock(0, im);

    // DEBUG:
    // printf("\n-> testFFT :: Setting up data in svtkMultiBlockDataSet");

    sensei::SVTKDataAdaptor *dataAdaptor = sensei::SVTKDataAdaptor::New();
    dataAdaptor->SetDataObject("mesh", mb);

    // sensei::MeshMetadataFlags mmdFlags = sensei::MeshMetadataFlags::New();
    sensei::MeshMetadataPtr mmd = sensei::MeshMetadata::New();
    mmd->Flags.SetBlockDecomp();
    mmd->Flags.SetBlockBounds();
    mmd->Flags.SetBlockExtents();

    dataAdaptor->GetMeshMetadata(0, mmd);

    // printf("MESHMETADATA SIM: (valid- %d) \n", mmd->Validate(MPI_COMM_WORLD, mmd->Flags));
    // mmd->ToStream(std::cout);

    im->Delete();
    mb->Delete();

    // Executing FFT via configurable analysis
    fft_endpoint->Execute(dataAdaptor, nullptr);
}


int 
main(int argc, char *argv[])
{
    // Initializing MPI 
    MPI_Init(&argc, &argv);
    
    int myrank, nranks; 
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Initializing some required variables
    if (argc != 6){
        printf("Exactly 5 arguments required: 'y_dim, x_dim, w, t, xml_file_path' and in the same order.\n");
        printf("y_dim: Y-dimension of the data\nx_dim: X-dimension of the data\nw = change the original value by what percentage? (0-100)\n");
        printf("t: what percentage of total values to change? (0-100)\nxml_file_path: XML file for FFT analysis endpoint.");
        exit(0);
    }

    // Hold dimensions + noise related variables 
    const int N0 = atoi(argv[1]), N1 = atoi(argv[2]), N0c = N0 / 2, N1c = N1 / 2;
    const int w = atoi(argv[3]);
    std::string data_file;
    int t = 0;
    
    if(w < 0){
        data_file = argv[4];
    }
    else{
        t = atoi(argv[4]);
    }
    string xml_file_path = argv[5];
    
    // If w = -1, we are reading input data for inverse FFT:
    if(w < 0){
        std::vector<double> denoised_data = readStuff(N0*N1, data_file);

        send_with_sensei(denoised_data, N0, N1, xml_file_path);
        return 0;
    }

    int local_n0, local_n0_start, y_diff, x_diff, y, x;
    
    // Timing stuff
    // chrono::time_point<chrono::high_resolution_clock> start_time, end_time;
    // chrono::duration<double> elapsed_computation_time;

    /* Row Decomposition */
    //      Maximum work that can be allocated to maximum ranks while ensuring all ranks gets some work
    int max_local_n0 = (N0 % nranks) ? N0 / nranks + 1 : N0 / nranks;
    int max_working_ranks = nranks - (max_local_n0*nranks - N0);

    if (myrank == 0){
        printf(":: test_FFT ::\n-> %d x %d domain with %d%% noise over %d%% data points with %d way parallel execution.\n", N0, N1, w, t, nranks);

        local_n0_start = 0;
        local_n0 = max_local_n0;
    }
    else if (myrank+1 > max_working_ranks){
        local_n0 = max_local_n0-1;
        local_n0_start = max_working_ranks*max_local_n0 + (myrank-max_working_ranks)*(max_local_n0 - 1);
    }
    else{
        local_n0 = max_local_n0;
        local_n0_start = myrank*max_local_n0;
    }

    // all_data_containers (to store data aggregated from all ranks) 
    vector <double> all_pure_data(N0*N1);
    fill(all_pure_data.begin(), all_pure_data.end(), -1.0);

    vector <double> all_noisy_data(N0*N1);
    fill(all_noisy_data.begin(), all_noisy_data.end(), -1.0);

    /* Creating a source buffer of all values of computation to send */
    vector <double> data(N1*local_n0);    
    fill(data.begin(), data.end(), -1.0);

    // ---------------------------------------------------
    // start_time = chrono::high_resolution_clock::now();

    // initialize data using computation formula
    for (y = 0; y < local_n0; ++y) 
        for (x = 0; x < N1; ++x){   
            y_diff = y + local_n0_start - N0c;
            x_diff = x - N1c;

            data[y*N1 + x] = sqrt(x_diff*x_diff + y_diff*y_diff);
        }

    // DEBUG:
    // printf("\n-> myrank: %d out of %d <- N0: %d, N1: %d, C[%d, %d],  local_N0: %d, local_0_start: %d\n", myrank, nranks, N0, N1, N0c, N1c, local_n0, local_n0_start);
    // printf("\nPURE: (rank %d) [%d x %d] \n", myrank, local_n0, N1);
    // for (y = 0; y < local_n0; ++y) {
    //     for (x = 0; x < N1; ++x)
    //     {
    //         printf("`%f\t", data[y*N1 + x]);
    //     }
    //     printf("\n");
    // }

    // ---------------------------------------------------
    /* SENDING PURE DATA (tag=0) */
    // if (myrank != 0){   
    //     // DEBUG:
    //     // printf("\nSENDING FROM %d, <PURE DATA> size: %ld\n", myrank, data.size());
    //     MPI_Send(data.data(), local_n0*N1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    // }
    // // Adding pure data of rank 0 to all_data_container
    // else{
    //     for(size_t k = 0; k < data.size(); ++k)
    //         all_pure_data.at(k) = data.at(k);
    // }

    // calculating noise indices:
    set <int> noisy_indices;
    int noise_count = t*local_n0*N1/100;

    /* generating new random numbers for every run */ 
    static random_device rd;
    srand(rd());
    
    /* storing indices based on t where we need to apply noise */
    for (int k = 0; k < noise_count; ++k){
        noisy_indices.insert(rand() % (local_n0*N1));
    }

    for(size_t k = 0; k < data.size(); ++k){
        if (noisy_indices.count(k))
            data.at(k) = noisy(data.at(k), w);
        else
            data.at(k) = data.at(k);
    }

    // DEBUG:
    // printf("\nNOISY: (rank %d) [%d x %d] \n", myrank, local_n0, N1);
    // for (y = 0; y < local_n0; ++y) {
    //     for (x = 0; x < N1; ++x)
    //     {   
    //         if (noisy_indices.count(y*N1 + x)){
    //             printf("!%f\t", data[y*N1 + x]);
    //         }
    //         else{
    //             printf("-\t");
    //         }
    //     }
    //     printf("\n");
    // }

    send_with_sensei(data, static_cast<double>(local_n0), static_cast<double>(N1), xml_file_path);

    /* In case we need to measure time and write to file */

    //     // ---------------------------------------------------
    //     /* final Time Measurements */
    //     end_time = chrono::high_resolution_clock::now();
    //     elapsed_computation_time = end_time - start_time;
    //     long elapsed_time_int = elapsed_computation_time.count();

    //     // printf("\n____ TIMING RESULTS ____");
    //     // printf("\nElapsed Computation Time: %6.4ld (ms)\n", elapsed_time_int);

    //     /* Writing output in data folder */
    //     std::string output_file_name;
    //     output_file_name = "pure_";
    //     output_file_name += argv[1];
    //     output_file_name += "x";
    //     output_file_name += argv[2];
    //     output_file_name += "_";
    //     // output_file_name += argv[3];
    //     // output_file_name += "-";
    //     // output_file_name += argv[4];
    //     // output_file_name += "n_";
    //     output_file_name += to_string(myrank);
    //     output_file_name += ".dat";
    //     // writeStuff(N0*N1, output_file_name, all_pure_data, {});

    //     output_file_name = "noisy_";
    //     output_file_name += argv[1];
    //     output_file_name += "x";
    //     output_file_name += argv[2];
    //     output_file_name += "_";
    //     // output_file_name += argv[3];
    //     // output_file_name += "-";
    //     // output_file_name += argv[4];
    //     // output_file_name += "n_";
    //     output_file_name += to_string(myrank);
    //     output_file_name += ".dat";
    //     // writeStuff(N0*N1, output_file_name, all_noisy_data, {});

    //     // output_file_name = "data/labels_";
    //     // output_file_name += argv[1];
    //     // output_file_name += "x";
    //     // output_file_name += argv[2];
    //     // output_file_name += "_";
    //     // output_file_name += argv[3];
    //     // output_file_name += "-";
    //     // output_file_name += argv[4];
    //     // output_file_name += "n_";
    //     // output_file_name += to_string(nranks);
    //     // output_file_name += "r.dat";
    //     // writeStuff(N0*N1, output_file_name, {}, labels);

    //     send_with_sensei(all_noisy_data, N0, N1, xml_file_path);
    // }
    MPI_Finalize();
    return 0;
}
