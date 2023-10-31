/*
FFT ENDPOINT

Perform FORWARD & INVERSE FFT on data ingested by simulation. Requires fftw library.
Needs an XML configuration file for data ingesion, with following parameters:
- mesh: mesh name
- direction: FFTW_FORWARD / FFTW_INVERSE
- array: array name
- python_xml: XML file denoting python_endpoint config for displaying the image of FFT output

Project:
Sensei - FFTW - SFSU

Contributions:
S. Kulkarni, E. Wes Bethel, B. Loring
*/

#include "Fft.h"
#include "DataAdaptor.h"
#include "ConfigurableAnalysis.h"
#include "InTransitDataAdaptor.h"
#include "MeshMetadata.h"
#include "MeshMetadataMap.h"
#include "SVTKUtils.h"

#include <svtkDoubleArray.h>
#include <svtkImageData.h>
#include <svtkPointData.h>
#include <svtkFieldData.h>
#include <SVTKDataAdaptor.h>
#include <svtkSmartPointer.h>
#include <svtkCompositeDataIterator.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkCompositeDataSet.h>
#include <svtkDataObject.h>

#include <iostream>
#include <vector>
#include <complex>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <typeinfo>

#include <fftw3-mpi.h>

/* Writing output instead of SENSEI Endpoint 
NOT USED ANY MORE, included for debugging. */
void
write_to_file(long int data_size, std::string outputFileName, std::vector<double> output_data_floats = {}, std::vector<int> output_labels = {})
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

/* Send over to Python */ 
static void
send_with_sensei(std::vector <double> data, ptrdiff_t xDim, ptrdiff_t yDim, std::string const& xmlFileName)
{
    // initialize xml for python
    sensei::ConfigurableAnalysis *aa = sensei::ConfigurableAnalysis::New();
    aa->Initialize(xmlFileName);     
    
    // Setting up double array
    svtkDoubleArray *da = svtkDoubleArray::New();
    unsigned int nVals = data.size();

    da->SetNumberOfTuples(nVals);
    da->SetName("data");
    for (unsigned int i = 0; i < nVals; ++i)
        *da->GetPointer(i) = data.at(i);
    
    // DEBUG:
    // printf("\n-> FFT to Python :: Setting up data in svtkDoubleArray of size %d", nVals);

    // Setting up image data
    svtkImageData *im = svtkImageData::New(); 
    im->SetDimensions(xDim, yDim, 1);
    im->GetPointData()->AddArray(da);
    da->Delete();
    
    // DEBUG:
    // printf("\n-> FFT to Python :: Setting up data in svtkImageData");

    // Setting up MultiBlockDataSet    
    svtkMultiBlockDataSet* mb = svtkMultiBlockDataSet::New();
    mb->SetNumberOfBlocks(1);
    mb->SetBlock(0, im);

    // Add to Data Adaptor
    sensei::SVTKDataAdaptor *dataAdaptor = sensei::SVTKDataAdaptor::New();
    dataAdaptor->SetDataObject("mesh", mb); 
    
    // Setup MeshMetaData flags
    sensei::MeshMetadataPtr mmd = sensei::MeshMetadata::New();
    mmd->Flags.SetBlockDecomp();
    mmd->Flags.SetBlockBounds();
    mmd->Flags.SetBlockExtents();

    im->Delete();
    mb->Delete();

    // DEBUG:
    // printf("\n-> FFT to Python :: Setting up data in svtkDataAdaptor");

    // printf("MESHMETADATA CON-PY: (valid- %d) \n", mmd->Validate(MPI_COMM_WORLD, mmd->Flags));
    // mmd->ToStream(std::cout);

    // Send via configurable analysis adaptor
    aa->Execute(dataAdaptor, nullptr);
}


/* Perform FFTW */
std::vector<double>
fftw(ptrdiff_t N0, ptrdiff_t N1, std::string direction, std::vector<double> input_data)
{
    /* FFTW MPI Stuff */
    ptrdiff_t alloc_local, local_n0, local_n0_start, y, x;

    fftw_plan plan;
    fftw_complex *fftw_buffer;

    fftw_mpi_init();

    /* get local data size and allocate */
    alloc_local = fftw_mpi_local_size_2d(N0, N1, MPI_COMM_WORLD, &local_n0, &local_n0_start);
    fftw_buffer = fftw_alloc_complex(alloc_local);

    // TODO: Multi dimension:
    // alloc_local = fftw_mpi_local_size_many(dimension, &n, 1, FFTW_MPI_DEFAULT_BLOCK(0),
    //                                MPI_COMM_WORLD, &local_n0, &local_0_start);

    /* create plan for in-place DFT */
    if (direction == "FFTW_FORWARD")
        plan = fftw_mpi_plan_dft_2d(N0, N1, fftw_buffer, fftw_buffer, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);  
        // plan = fftw_mpi_plan_dft(dimension, &n, fftw_buffer, fftw_buffer, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);  
        
    else
        plan = fftw_mpi_plan_dft_2d(N0, N1, fftw_buffer, fftw_buffer, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);  
        // plan = fftw_mpi_plan_dft(dimension, &n, fftw_buffer, fftw_buffer, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);  


    /* Adding input data to FFTW data buffer */
    for (y = 0; y < local_n0; ++y) 
        for (x = 0; x < N1; ++x){
            fftw_buffer[y*N1 + x][0] = input_data[y*N1 + x];
            fftw_buffer[y*N1 + x][1] = 0;

            //TODO: Find ways to move n-dim data in fftw_buffer
        }

    // DEBUG:
    // printf("\nINPUT DATA: (rank %d) [%ld x %ld] at offet %ld\n", myrank, local_n0, N1, local_n0_start);
    // for (y = 0; y < local_n0; ++y) {
    //     for (x = 0; x < N1; ++x)
    //     {
    //         printf("`(%d)%f\t", myrank, data[y*N1 + x][0]);
    //     }
    //     printf("\n");
    // }
    
    /* compute transforms, in-place, as many times as desired */
    fftw_execute(plan);

    // DEBUG:
    // printf("\nAFTER Forward FFT: (rank %d) [%ld x %ld]\n", myrank, local_n0, N1);
    // for (y = 0; y < local_n0; ++y) {
    //     for (x = 0; x < N1; ++x){
    //         printf("~(%d)%f\t", myrank, data[y*N1 + x][0]);
    //     }
    //     printf("\n");
    // }

    std::vector<double> fftw_out;
    fftw_out.reserve(local_n0*N1);

    // Adding FFT output data to source buffer
    if (direction == "FFTW_FORWARD"){
        for(x = 0; x < local_n0*N1; ++x)
            fftw_out.push_back(fftw_buffer[x][0]);
    }
    else{
        // FFTW_BACKWARD data is scaled: https://www.fftw.org/faq/section3.html#whyscaled
        for(x = 0; x < local_n0*N1; ++x)
            fftw_out.push_back(fftw_buffer[x][0]/(N0*N1));
    }

    // auto getMinMaxRow  = std::minmax_element(std::begin(fftw_out), std::end(fftw_out));
    // std::cout << "min = " << *getMinMaxRow.first << ", max = " << *getMinMaxRow.second << '\n';

    fftw_mpi_cleanup();
    
    return fftw_out;
}

/* ANALYSIS ENDPOINT */
namespace sensei
{
struct Fft::InternalsType
{
  InternalsType() : N0(12), N1(12), direction("FFTW_FORWARD"), python_xml(""), mesh_name(""), array_name("") {}
  ~InternalsType() {}

  ptrdiff_t N0, N1;
  std::string direction;
  std::vector <double> data;
  std::string python_xml;
  std::string mesh_name;
  std::string array_name;
  
// TODO
//   int dimension;
//   std::vector <ptrdiff_t> n;

};

//----------------------------------------------------------------------------
Fft::Fft()
{
  MPI_Comm_dup(MPI_COMM_WORLD, &this->Comm);
  this->Internals = new InternalsType;
}

//----------------------------------------------------------------------------
Fft::~Fft()
{
  MPI_Comm_free(&this->Comm);
  delete this->Internals;
}

//----------------------------------------------------------------------------
// Setup internal struct
void Fft::Initialize(std::string const& direction, std::string const& python_xml, std::string const& mesh_name, std::string const& array_name){
    this->Internals->direction = direction;
    this->Internals->python_xml = python_xml;
    this->Internals->mesh_name = mesh_name;
    this->Internals->array_name = array_name;
}

//----------------------------------------------------------------------------
bool Fft::Execute(sensei::DataAdaptor* dataIn, sensei::DataAdaptor** dataOut)
{
    SENSEI_STATUS("Executing FFT ENDPOINT.");

    // MPI_Comm comm = this->GetCommunicator();

    int myrank, nranks; 
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    
    // we do not return anything yet
    if (dataOut)
    {
        *dataOut = nullptr;
    }

    // TODO
    // Check if we can cast the data adaptor to an in-transit data adaptor.
    // InTransitDataAdaptor *itDataAdaptor = dynamic_cast<InTransitDataAdaptor*>(dataIn); 

    MeshMetadataFlags flags;
    flags.SetBlockDecomp();
    flags.SetBlockBounds();

    // see what the simulation is providing
    MeshMetadataMap mdMap;
    if (mdMap.Initialize(dataIn, flags)){
        SENSEI_ERROR("Failed to get metadata")
        return false;
    }

    // get the mesh metadata object
    MeshMetadataPtr mmd;
    if (mdMap.GetMeshMetadata(this->Internals->mesh_name, mmd)){
        SENSEI_ERROR("Failed to get metadata for mesh \"" << this->Internals->mesh_name << "\"")
        return false;
    }

    // TODO
    // printf("MESHMETADATA CON: (valid- %d) \n", mmd->Validate(this->Comm, mmd->Flags));
    // mmd->ToStream(std::cout);

    // get the mesh object
    svtkDataObject *dobj = nullptr;

    if (dataIn->GetMesh(this->Internals->mesh_name, false, dobj)){
        SENSEI_ERROR("Failed to get mesh: \t " + this->Internals->mesh_name << "\"")
        
        MPI_Abort(this->Comm, -1);
        return false;
    }

     // fetch the array that the FFT will be computed on
    if (dataIn->AddArray(dobj, this->Internals->mesh_name, svtkDataObject::POINT, this->Internals->array_name)){
        SENSEI_ERROR(<< dataIn->GetClassName() << " failed to add point"
        << " data array \""  <<  this->Internals->array_name << "\"")
        
        MPI_Abort(this->Comm, -1);
        return false;
    }

    // convert to multiblock and process the blocks
    svtkCompositeDataSetPtr mesh = SVTKUtils::AsCompositeData(Comm, dobj, true);
    svtkSmartPointer<svtkCompositeDataIterator> iter;
    iter.TakeReference(mesh->NewIterator());
    
    for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
    {   
        // get the block
        svtkDataObject *curObj = iter->GetCurrentDataObject();
        
        // Get Data from simulation
        svtkDataArray* da = this->GetArray(curObj, this->Internals->array_name);
        if (!da)
        {
            SENSEI_WARNING("Data block " << iter->GetCurrentFlatIndex()
                << " of mesh \"" << this->Internals->mesh_name  << "\" has no array named \""
                << this->Internals->array_name << "\"")
            continue;
        }

        // Check if the input data object is actually an image data and fetch dimensions from it
        svtkImageData* imageData = svtkImageData::SafeDownCast(curObj);
        if (imageData){
            int dims[3];
            imageData->GetDimensions(dims);	

            this->Internals->N0 = dims[0];
            this->Internals->N1 = dims[1];
        }
        else{
            SENSEI_WARNING("Image Data has no dimensions.");
            continue;
        }
        
        // Copy data into Internals
        for (int i = 0; i < (this->Internals->N0 * this->Internals->N1); ++i){
            this->Internals->data.push_back(*da->GetTuple(i));
        }  
    }
    
    // DEBUG:
    // printf("\n-> FFT :: ALL input DATA === [%ld x %ld]\n", this->Internals->N0, this->Internals->N1);
    // for (ptrdiff_t y = 0; y < this->Internals->N0; ++y) {
    //     for (ptrdiff_t x = 0; x < this->Internals->N1; ++x){
    //         printf("%f\t", this->Internals->data.at(y*this->Internals->N1 + x));
    //     }
    //     printf("\n");
    // }

    if (myrank == 0){
        printf(":: FFTW ::\n-> %ld x %ld domain in %s direction with %d way parallel execution.\n", this->Internals->N0, this->Internals->N1, this->Internals->direction.c_str(), nranks);
    }
    // Perform FFT
    std::vector<double> fftw_data = fftw(this->Internals->N0, this->Internals->N1, this->Internals->direction, this->Internals->data);

    // DEBUG:
    // printf("\n-> FFT :: ALL fftw DATA === [%ld x %ld] on rank {%d}\n", this->Internals->N0, this->Internals->N1, myrank);
    // for (ptrdiff_t y = 0; y < this->Internals->N0; ++y) {
    //     for (ptrdiff_t x = 0; x < this->Internals->N1; ++x){
    //         printf("%f\t", fftw_data.at(y*this->Internals->N1 + x));
    //     }
    //     printf("\n");
    // }

    // std::string output_file_name;
    // output_file_name = "spectral_";
    // output_file_name += this->Internals->direction;
    // output_file_name += "_";
    // output_file_name += std::to_string(this->Internals->N0);
    // output_file_name += "x";
    // output_file_name += std::to_string(this->Internals->N1);
    // output_file_name += "_";
    // output_file_name += std::to_string(myrank);
    // output_file_name += ".dat";
    // write_to_file(this->Internals->N0*this->Internals->N1, output_file_name, fftw_data, {});

    // TODO: Only if python-xml provided
    // Send to python
    send_with_sensei(fftw_data, this->Internals->N1, this->Internals->N0, this->Internals->python_xml);
    return true;
}

//-----------------------------------------------------------------------------
// Get array from svtkDataObject
svtkDataArray* Fft::GetArray(svtkDataObject* dobj, const std::string& arrayname)
{
    if (svtkFieldData* fd = dobj->GetAttributesAsFieldData(0)){
        return fd->GetArray(arrayname.c_str());
    }
    return nullptr;
}

//-----------------------------------------------------------------------------
int Fft::Finalize()
{
  return 0;
}

//----------------------------------------------------------------------------
Fft* Fft::New(){
    return new sensei::Fft();
}

}
