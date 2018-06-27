#include "ProgrammableDataAdaptor.h"
#include "PythonAnalysis.h"
#include "ConfigurableAnalysis.h"
#include "Error.h"

#include <vtkMultiBlockDataSet.h>
#include <vtkImageData.h>
#include <vtkCellData.h>
#include <vtkDoubleArray.h>
#include <vtkDataObject.h>

#include <iostream>
#include <sstream>
#include <random>

using std::cerr;
using std::endl;

// x - y mesh size, the mesh gets one
// layer in z for each rank
int gnx = 100;
int gny = 100;


// generate random Gaussian
template<typename n_t>
int getSequence(n_t *vals, long nVals)
{
  std::random_device dev;
  std::mt19937 gen(dev());
  std::normal_distribution<n_t> dist(0.0,0.5);

  for (long i = 0; i < nVals; ++i)
    vals[i] = dist(gen);

  return 0;
}

// data adaptor
int getNumMeshes(unsigned int &n)
{
    n = 1;
    return 0;
}

int getMeshName(unsigned int i, std::string &name)
{
  if (i == 0)
    {
    name = "mesh";
    return 0;
    }
  return -1;
}

int getNumberOfArrays(const std::string &meshName, int assoc, unsigned int &num)
{
  num = 0;
  if (meshName == "mesh")
    {
    if (assoc == vtkDataObject::CELL)
      num = 1;
    return 0;
    }

  SENSEI_ERROR("Invalid mesh \"" << meshName << "\"")
  return -1;
};

int getArrayName(const std::string &meshName, int assoc, unsigned int id, std::string &arrayName)
{
  if ((meshName == "mesh") && (assoc == vtkDataObject::CELL) && (id == 0))
    {
    arrayName = "values";
    return 0;
    }
  return -1;
}

int getMesh(const std::string &meshName, bool, vtkDataObject *&mesh)
{
  if (meshName == "mesh")
    {
    int rank = 0;
    int nRanks = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    int ext[] = {0, gnx, 0, gny, rank, rank+1};

    vtkImageData *im = vtkImageData::New();
    im->SetExtent(ext);

    vtkMultiBlockDataSet *mb = vtkMultiBlockDataSet::New();
    mb->SetNumberOfBlocks(nRanks);
    mb->SetBlock(rank, im);
    im->Delete();

    mesh = mb;

    return 0;
    }
  return -1;
}

int addArray(vtkDataObject *mesh, const std::string &meshName,
  int assoc, const std::string &name)
{
  if ((meshName == "mesh") && (assoc == vtkDataObject::CELL) && (name == "values"))
    {
    int rank = 0;
    int nRanks = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    vtkMultiBlockDataSet *mb = dynamic_cast<vtkMultiBlockDataSet*>(mesh);

    if (!mb)
      return -1;

    vtkImageData *ds = dynamic_cast<vtkImageData*>(mb->GetBlock(rank));

    if (!ds)
      return -1;

    long nVals = gnx*gny;

    vtkDoubleArray *da = vtkDoubleArray::New();
    da->SetName("values");
    da->SetNumberOfTuples(nVals);
    double *vals = da->GetPointer(0);

    getSequence(vals, nVals);

    ds->GetCellData()->AddArray(da);
    da->Delete();

    return 0;
    }
  return -1;
}



int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  if (argc != 2)
    {
    SENSEI_ERROR("need an xml config on the command line")
    MPI_Abort(MPI_COMM_WORLD, -1);
    return -1;
    }

  sensei::ProgrammableDataAdaptor *da = sensei::ProgrammableDataAdaptor::New();
  da->SetGetNumberOfMeshesCallback(getNumMeshes);
  da->SetGetMeshNameCallback(getMeshName);
  da->SetGetNumberOfArraysCallback(getNumberOfArrays);
  da->SetGetArrayNameCallback(getArrayName);
  da->SetGetMeshCallback(getMesh);
  da->SetAddArrayCallback(addArray);

  sensei::ConfigurableAnalysis *aa = sensei::ConfigurableAnalysis::New();
  if (aa->Initialize(argv[1]))
    {
    SENSEI_ERROR("Failed to intialize the analysis")
    MPI_Abort(MPI_COMM_WORLD, -1);
    return -1;
    }

  /*sensei::PythonAnalysis *aa = sensei::PythonAnalysis::New();
  //aa->SetModuleName("anal");
  aa->SetScriptFile("anal.py");
  aa->SetInitializeSource("numBins=10; meshName='mesh'; arrayName='values'; arrayCen=1");
  aa->Initialize();*/

  for (int i = 0; i < 5; ++i)
    {
    da->SetDataTimeStep(i);
    da->SetDataTime(i);
    aa->Execute(da);
    }

  aa->Finalize();

  da->Delete();
  aa->Delete();

  MPI_Finalize();

  return 0;
}
