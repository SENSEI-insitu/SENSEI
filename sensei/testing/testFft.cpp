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
Sudhanshu Kulkarni, E. Wes Bethel
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

// #include <where-mpi.h>
#include <svtkDataArray.h>
#include <svtkFieldData.h>
#include <svtkDataObject.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkSmartPointer.h>

#include <svtkDoubleArray.h>
#include <svtkImageData.h>
#include <svtkPointData.h>

//#include <fftw3.h>
//#include <fftw3-mpi.h>

#include "ConfigurableAnalysis.h"
#include "MeshMetadata.h"
#include "AnalysisAdaptor.h"
#include "SVTKDataAdaptor.h"
#include "Error.h"


using namespace std;

/* Writing output instead of SENSEI Endpoint 
NOT USED ANY MORE, included for debugging. */
void
writeStuff(long int data_size, string outputFileName, vector<double> output_data_floats = {}, vector<int> output_labels = {})
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
send_with_sensei(vector <double> data, ptrdiff_t xDim, ptrdiff_t yDim, string const& xmlFileName)
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

    im->Delete();
    mb->Delete();

    // DEBUG:
    // printf("\n-> testFFT :: Setting up data in svtkDataAdaptor");

    // Executing FFT via configurable analysis
    fft_endpoint->Execute(dataAdaptor, nullptr);
}


int 
main(int argc, char *argv[])
{
    // Initializing some required variables
    if (argc != 6){
        printf("Exactly 5 arguments required: 'y_dim, x_dim, w, t, xml_file_path' and in the same order.\n");
        printf("y_dim: Y-dimension of the data\nx_dim: X-dimension of the data\nw = change the original value by what percentage? (0-100)\n");
        printf("t: what percentage of total values to change? (0-100)\nxml_file_path: XML file for FFT analysis endpoint.");
        exit(0);
    }

    // Hold dimensions + noise related variables 
    const int N0 = atoi(argv[1]), N1 = atoi(argv[2]), N0c = N0 / 2, N1c = N1 / 2;
    const int w = atoi(argv[3]), t = atoi(argv[4]);
    string xml_file_path = argv[5];

    int local_n0, local_n0_start, y_diff, x_diff, y, x;
    
    // Timing stuff
    chrono::time_point<chrono::high_resolution_clock> start_time, end_time;
    chrono::duration<double> elapsed_computation_time;

    // Initializing MPI 
    MPI_Init(&argc, &argv);
    int myrank, nranks; 
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // fftw_mpi_init();

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
    start_time = chrono::high_resolution_clock::now();

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
    if (myrank != 0){   
        // DEBUG:
        // printf("\nSENDING FROM %d, <PURE DATA> size: %ld\n", myrank, data.size());
        MPI_Send(data.data(), local_n0*N1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    // Adding pure data of rank 0 to all_data_container
    else{
        for(size_t k = 0; k < data.size(); ++k)
            all_pure_data.at(k) = data.at(k);
    }

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

    /* SENDING NOISY DATA (tag=1) */
    if (myrank != 0){   
        // DEBUG:
        // printf("\nSENDING FROM %d, <NOISY DATA> size: %ld\n", myrank, data.size());
        MPI_Send(data.data(), local_n0*N1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
    }
    /* GATHERING DATA */
    else{
        // Adding noisy data of rank 0 to all_data_container
        for(size_t k = 0; k < data.size(); ++k)
            all_noisy_data.at(k) = data.at(k);
        
        // Maintaining rank labels and portions of the domain they worked on
        vector <int> labels(N1*N0);
        for (int k = 0; k < local_n0; ++k)
            labels.at(k) = 0;

        /* Getting data from all other ranks */
        for (int k = 1; k < nranks; ++k){
            // Calculate position info for each rank
            if (k+1 > max_working_ranks){
                local_n0 = max_local_n0-1;
                local_n0_start = max_working_ranks*max_local_n0 + (k-max_working_ranks)*(max_local_n0 - 1);
            }
            else{
                local_n0 = max_local_n0;
                local_n0_start = k*max_local_n0;
            }

            // Buffer to recieve into
            vector <double> rbuf(local_n0*N1);
            int all_offset = local_n0_start*N1;
            int rbuf_index = 0;
            
            int rcount;
            MPI_Status stat;

            // Receiving pure data
            MPI_Recv(rbuf.data(), local_n0*N1, MPI_DOUBLE, k, 0,  MPI_COMM_WORLD, &stat);
            MPI_Get_count(&stat, MPI_DOUBLE, &rcount);

            if(rcount != local_n0*N1) {
                printf("\n[Rank %d] error: we expected %d items from Rank %d, but received only %d items. \n", 0, local_n0*N1, k, rcount);
            }
            else{
                // DEBUG
                // printf("\nRECEiVING FROM %d, <PURE DATA> size: %d at y=%d\n", k, rcount, local_n0_start);
                for(x = all_offset; x < all_offset + rbuf.size(); ++x, ++rbuf_index)
                    all_pure_data.at(x) = rbuf.at(rbuf_index);
            }

            // Receiving noisy data
            MPI_Recv(rbuf.data(), local_n0*N1, MPI_DOUBLE, k, 1,  MPI_COMM_WORLD, &stat);
            MPI_Get_count(&stat, MPI_DOUBLE, &rcount);

            if(rcount != local_n0*N1) {
                printf("\n[Rank %d] error: we expected %d items from Rank %d, but received only %d items. \n", 0, local_n0*N1, k, rcount);
            }
            else{
                // DEBUG
                // printf("\nRECEiVING FROM %d, <NOISY DATA> size: %d at y=%d\n", k, rcount, local_n0_start);
                for(x = all_offset, rbuf_index = 0; x < all_offset + rbuf.size(); ++x, ++rbuf_index)
                    all_noisy_data.at(x) = rbuf.at(rbuf_index);
            }

            // Adding rank labels
            for(x = all_offset; x < all_offset + (local_n0*N1); ++x)
                labels.at(x) = k;
        }

        // DEBUG:
        // printf("\n=== ALL pure DATA === [%d, %d]\n", N0, N1);
        // for (y = 0; y < N0; ++y) {
        //     for (x = 0; x < N1; ++x){
        //         printf("%f\t", all_pure_data[y*N1 + x]);
        //     }
        //     printf("\n");
        // }

        // DEBUG:
        // printf("\n=== ALL noisy DATA === [%d, %d]\n", N0, N1);
        // for (y = 0; y < N0; ++y) {
        //     for (x = 0; x < N1; ++x){
        //         printf("%f\t", all_noisy_data[y*N1 + x]);
        //     }
        //     printf("\n");
        // }

        // ---------------------------------------------------
        /* final Time Measurements */
        end_time = chrono::high_resolution_clock::now();
        elapsed_computation_time = end_time - start_time;
        long elapsed_time_int = elapsed_computation_time.count();

        // printf("\n____ TIMING RESULTS ____");
        // printf("\nElapsed Computation Time: %6.4ld (ms)\n", elapsed_time_int);

        // Invoke FFT:
        send_with_sensei(all_noisy_data, N0, N1, xml_file_path);

        // /* Writing output in data folder */
        // string output_file_name;
        // output_file_name = "data/pure_";
        // output_file_name += argv[1];
        // output_file_name += "x";
        // output_file_name += argv[2];
        // output_file_name += "_";
        // output_file_name += argv[3];
        // output_file_name += "-";
        // output_file_name += argv[4];
        // output_file_name += "n_";
        // output_file_name += to_string(nranks);
        // output_file_name += "r.dat";
        // writeStuff(N0*N1, output_file_name, all_pure_data, {});

        // output_file_name = "data/noisy_";
        // output_file_name += argv[1];
        // output_file_name += "x";
        // output_file_name += argv[2];
        // output_file_name += "_";
        // output_file_name += argv[3];
        // output_file_name += "-";
        // output_file_name += argv[4];
        // output_file_name += "n_";
        // output_file_name += to_string(nranks);
        // output_file_name += "r.dat";
        // writeStuff(N0*N1, output_file_name, all_noisy_data, {});

        // output_file_name = "data/labels_";
        // output_file_name += argv[1];
        // output_file_name += "x";
        // output_file_name += argv[2];
        // output_file_name += "_";
        // output_file_name += argv[3];
        // output_file_name += "-";
        // output_file_name += argv[4];
        // output_file_name += "n_";
        // output_file_name += to_string(nranks);
        // output_file_name += "r.dat";
        // writeStuff(N0*N1, output_file_name, {}, labels);
    }
    MPI_Finalize();
    return 0;
}
