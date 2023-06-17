#include <random>
#include <iostream>
#include <mpi.h>
#include <svtkHAMRDataArray.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkTable.h>
#include "Error.h"
#include "DataBinning.h"
#include "DataAdaptor.h"

template<typename n_t>
int genCoords(long start, long end, n_t *x, n_t a = 5)
{
  static unsigned long q = 0;
  q += start;
  std::random_device dev;
  std::mt19937 gen(dev());
  std::normal_distribution<n_t> dist(0.,1.);
  for (long i = start; i < end; ++i)
  {
    do {
    gen.seed(q);
    x[i] = dist(gen);
    ++q;
    } while (!((x[i] >= -a) && (x[i] <= a)));
  }
  return 0;
}



///  a data adaptor that generates a set of particles
class ParticleSource : public sensei::DataAdaptor
{
public:

  static ParticleSource *New(long n, double bds) { return new ParticleSource(n, bds); }

  int GetNumberOfMeshes(unsigned int &n) override { n = 1; return 0; }

  int GetMeshMetadata(unsigned int, sensei::MeshMetadataPtr &md) override
  {
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &n_ranks);
    md->MeshName = "particles";
    md->MeshType = SVTK_MULTIBLOCK_DATA_SET;
    md->BlockType = SVTK_TABLE;
    md->Bounds = {-this->Bds, this->Bds, -this->Bds, this->Bds, -this->Bds, this->Bds};
    md->BlockBounds = {{-this->Bds, this->Bds, -this->Bds, this->Bds, -this->Bds, this->Bds}};
    md->NumBlocks = n_ranks;
    md->NumBlocksLocal = {1};
    md->NumArrays = 4;
    md->ArrayName = {"xpos","ypos","zpos","mass"};
    md->ArrayCentering = {1, 1, 1, 1};
    md->ArrayComponents = {1, 1, 1, 1};
    md->ArrayType = {SVTK_DOUBLE, SVTK_DOUBLE, SVTK_DOUBLE, SVTK_DOUBLE};
    md->ArrayRange = {{-this->Bds, this->Bds}, {-this->Bds, this->Bds}, {-this->Bds, this->Bds}, {-this->Bds, this->Bds}};
    md->GlobalView = 0;
    return 0;
  }

  int GetMesh(const std::string &, bool, svtkDataObject *&mesh) override
  {
    int rank = 0;
    int n_ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    auto blk = svtkTable::New();

    auto mbds = svtkMultiBlockDataSet::New();
    mbds->SetNumberOfBlocks(n_ranks);
    mbds->SetBlock(rank, blk);
    blk->Delete();

    mesh = mbds;

    return 0;
  }

  int AddArray(svtkDataObject* mesh, const std::string &, int, const std::string &arrayName) override
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto mbds = dynamic_cast<svtkMultiBlockDataSet*>(mesh);
    auto blk = dynamic_cast<svtkTable*>(mbds->GetBlock(rank));

    svtkStream strm;
    auto alloc = svtkAllocator::malloc;
#if defined(SENSEI_ENABLE_CUDA)
    alloc = svtkAllocator::cuda_host;
    strm = this->Stream[this->StreamId % 3];
    ++this->StreamId;
#endif

    svtkHAMRDoubleArray *arr;
    if (arrayName == "mass")
    {
      arr = svtkHAMRDoubleArray::New(arrayName, this->NPart,
              1, alloc, strm, svtkStreamMode::async, 1.);
    }
    else
    {
      arr = svtkHAMRDoubleArray::New(arrayName, this->NPart,
              1, alloc, strm, svtkStreamMode::async);

      genCoords(0, this->NPart, arr->GetData(), this->Bds/2.);
    }

    blk->AddColumn(arr);
    arr->Delete();

    return 0;
  }

  ~ParticleSource()
#if defined(SENSEI_ENABLE_CUDA)
  {
    for (int i = 0; i < 3; ++i)
      cudaStreamDestroy(this->Stream[i]);
  }
#else
  {}
#endif

protected:
  ParticleSource(long nPart, double bds) : NPart(nPart), Bds(bds)
#if defined(SENSEI_ENABLE_CUDA)
  , Stream{}, StreamId(0)
  {
    for (int i = 0; i < 3; ++i)
      cudaStreamCreate(&(this->Stream[i]));
  }
#else
  {}
#endif

  long NPart; ///< total number of particles on all ranks
  double Bds; ///< data is defined on a cube +/- Bds in all directions
#if defined(SENSEI_ENABLE_CUDA)
  unsigned StreamId;
  cudaStream_t Stream[3];
#endif

};



int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  if (argc < 4)
  {
    std::cerr << "testDataBinning [np] [x/y res] [odir] [op 1] ... [op n]" << std::endl;
    return -1;
  }

  long np = atoi(argv[1]);
  long res = atoi(argv[2]);
  const char *odir = argv[3];

  std::vector<std::string> array;
  std::vector<std::string> op;

  for (int i = 4; i < argc; ++i)
  {
    array.push_back("mass");
    op.push_back(argv[i]);
  }

  // generate some particle data
  auto da = ParticleSource::New(np, 10.);

  // process
  auto aa = sensei::DataBinning::New();
  aa->Initialize("particles","xpos","ypos",array,op,res,res,odir,0);

  aa->Execute(da, nullptr);

  aa->Finalize();

  // clean up
  aa->Delete();
  da->Delete();

  MPI_Finalize();

  return 0;
}
