#include "ProgrammableDataAdaptor.h"
#include "PythonAnalysis.h"
#include "ConfigurableAnalysis.h"
#include "MeshMetadata.h"
#include "Error.h"

#include <svtkMultiBlockDataSet.h>
#include <svtkImageData.h>
#include <svtkCellData.h>
#include <svtkDoubleArray.h>
#include <svtkDataObject.h>

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

int getMeshMetadata(unsigned int i, sensei::MeshMetadataPtr &mdp)
{
  if (i != 0)
    return -1;

  int rank = 0;
  int nRanks = 1;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

  mdp = sensei::MeshMetadata::New();
  mdp->MeshName = "mesh";
  mdp->MeshType = SVTK_MULTIBLOCK_DATA_SET;
  mdp->BlockType = SVTK_IMAGE_DATA;
  mdp->NumBlocks = nRanks;
  mdp->NumBlocksLocal = {1};
  mdp->NumArrays = 1;

  mdp->ArrayName = {"values"};
  mdp->ArrayCentering = {svtkDataObject::CELL};
  mdp->ArrayComponents = {1};
  mdp->ArrayType = {SVTK_DOUBLE};

  mdp->BlockIds = {0};
  mdp->BlockOwner = {rank};
  mdp->BlockBounds = {{0.0, 1.0, 0.0, 1.0, double(rank), double(rank+1)}};
  mdp->BlockExtents = {{0, gnx, 0, gny, rank, rank+1}};
  mdp->BlockNumCells = {gnx*gny};
  mdp->BlockNumPoints = {2*gnx*gny};

  mdp->BlockArrayRange = {{{std::numeric_limits<double>::lowest(),
    std::numeric_limits<double>::max()}}};

  return 0;
}

int getMesh(const std::string &meshName, bool, svtkDataObject *&mesh)
{
  if (meshName == "mesh")
    {
    int rank = 0;
    int nRanks = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    int ext[] = {0, gnx, 0, gny, rank, rank+1};

    svtkImageData *im = svtkImageData::New();
    im->SetExtent(ext);

    svtkMultiBlockDataSet *mb = svtkMultiBlockDataSet::New();
    mb->SetNumberOfBlocks(nRanks);
    mb->SetBlock(rank, im);
    im->Delete();

    mesh = mb;

    return 0;
    }
  return -1;
}

int addArray(svtkDataObject *mesh, const std::string &meshName,
  int assoc, const std::string &name)
{
  if ((meshName == "mesh") && (assoc == svtkDataObject::CELL) && (name == "values"))
    {
    int rank = 0;
    int nRanks = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    svtkMultiBlockDataSet *mb = dynamic_cast<svtkMultiBlockDataSet*>(mesh);

    if (!mb)
      return -1;

    svtkImageData *ds = dynamic_cast<svtkImageData*>(mb->GetBlock(rank));

    if (!ds)
      return -1;

    long nVals = gnx*gny;

    svtkDoubleArray *da = svtkDoubleArray::New();
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
  da->SetGetMeshMetadataCallback(getMeshMetadata);
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

    sensei::DataAdaptor *ret = nullptr;
    aa->Execute(da, ret);
    }

  aa->Finalize();

  da->Delete();
  aa->Delete();

  MPI_Finalize();

  return 0;
}
