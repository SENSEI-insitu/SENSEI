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

    // Add to Data Adaptor
    sensei::SVTKDataAdaptor *dataAdaptor = sensei::SVTKDataAdaptor::New();
    dataAdaptor->SetDataObject("mesh", im); 
    
    im->Delete();

    // DEBUG:
    // printf("\n-> FFT to Python :: Setting up data in svtkDataAdaptor");

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
    fftw_complex *data;

    fftw_mpi_init();

    /* get local data size and allocate */
    alloc_local = fftw_mpi_local_size_2d(N0, N1, MPI_COMM_WORLD, &local_n0, &local_n0_start);
    data = fftw_alloc_complex(alloc_local);

    /* create plan for in-place DFT */
    if (direction == "FFTW_FORWARD")
        plan = fftw_mpi_plan_dft_2d(N0, N1, data, data, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);  
    else
        plan = fftw_mpi_plan_dft_2d(N0, N1, data, data, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);  

    /* Read the input data and store the appropriate portion of it, in rank's data container */
    // std::vector <double> input_data = readStuff(N0*N1, input_file_name);
    ptrdiff_t data_index = 0;

    /* Adding input data to FFTW data buffer */
    for (y = local_n0_start; y < local_n0_start + local_n0; ++y) 
        for (x = 0; x < N1; ++x){
            data[data_index][0] = input_data[y*N1 + x];
            data[data_index][1] = 0;
            ++data_index;
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

    std::vector<double> sbuf;
    sbuf.reserve(local_n0*N1);

    // Adding FFT output data to source buffer
    if (direction == "FFTW_FORWARD"){
        for(x = 0; x < local_n0*N1; ++x)
            sbuf.emplace_back(data[x][0]);
    }
    else{
        for(x = 0; x < local_n0*N1; ++x)
            sbuf.emplace_back(data[x][0]/(N0*N1));
    }

    return sbuf;
}


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

  // we do not return anything yet
  if (dataOut)
    {
      *dataOut = nullptr;
    }
    
    // TODO
    // see what the simulation is providing
    MeshMetadataMap mdMap;
    if (mdMap.Initialize(dataIn)){
        SENSEI_ERROR("Failed to get metadata")
        return false;
    }

    // get the mesh metadata object
    MeshMetadataPtr mmd;
    if (mdMap.GetMeshMetadata(this->Internals->mesh_name, mmd)){
        SENSEI_ERROR("Failed to get metadata for mesh \"" << this->Internals->mesh_name << "\"")
        return false;
    }

    // get the mesh object
    svtkDataObject *dobj = nullptr;

    if (dataIn->GetMesh(this->Internals->mesh_name, false, dobj)){
        SENSEI_ERROR("Failed to get mesh: \t " + this->Internals->mesh_name << "\"")
        return false;
    }

     // fetch the array that the FFT will be computed on
    if (dataIn->AddArray(dobj, this->Internals->mesh_name, svtkDataObject::POINT, this->Internals->array_name)){
        SENSEI_ERROR(<< dataIn->GetClassName() << " failed to add point"
        << " data array \""  <<  this->Internals->array_name << "\"")

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

    // Perform FFT
    std::vector<double> fftw_data = fftw(this->Internals->N0, this->Internals->N1, this->Internals->direction, this->Internals->data);

    // DEBUG:
    // printf("\n-> FFT :: ALL fftw DATA === [%ld x %ld]\n", this->Internals->N0, this->Internals->N1);
    // for (ptrdiff_t y = 0; y < this->Internals->N0; ++y) {
    //     for (ptrdiff_t x = 0; x < this->Internals->N1; ++x){
    //         printf("%f\t", fftw_data.at(y*this->Internals->N1 + x));
    //     }
    //     printf("\n");
    // }

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
