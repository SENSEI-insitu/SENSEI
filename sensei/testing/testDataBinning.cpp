#include <random>
#include <iostream>
#include <mpi.h>
#include <svtkHAMRDataArray.h>
#include <svtkMultiBlockDataSet.h>
#include <svtkTable.h>
#include "Error.h"
#include "DataBinning.h"
#include "DataAdaptor.h"

/// generate a Gaussian distribution of particles
template<typename n_t>
int genCoords(long start, long end, n_t *x, n_t cen, n_t mu, n_t a)
{
  static unsigned long q = 0;
  q += start;
  std::random_device dev;
  std::mt19937 gen(dev());
  gen.seed(q);
  std::normal_distribution<n_t> dist(cen, mu);
  n_t twoA = n_t(2)*a;
  for (long i = start; i < end; ++i)
  {
    double xi = dist(gen);
    if (xi < 0.0)
        x[i] = a - fmod(fabs(xi - a), twoA);
    else
        x[i] = fmod(xi + a, twoA) - a;
    ++q;
  }
  return 0;
}

/// shift the particle position by dx
template<typename n_t>
int updateCoords(long start, long end, n_t *x, n_t dx, n_t a)
{
  n_t twoA = n_t(2)*a;
  for (long i = start; i < end; ++i)
  {
    n_t xi = x[i] + dx;
    if (xi < 0.0)
        x[i] = a - fmod(fabs(xi - a), twoA);
    else
        x[i] = fmod(xi + a, twoA) - a;
  }
  return 0;
}



///  a data adaptor that generates a set of particles
class ParticleSource : public sensei::DataAdaptor
{
public:

  static ParticleSource *New(long n, double bds, double dt) { return new ParticleSource(n, bds, dt); }

  void Increment()
  {
    this->SetDataTimeStep(this->GetDataTimeStep() + 1);
    this->SetDataTime(this->GetDataTimeStep() + this->Dt);

    updateCoords(0, this->NPart, this->XPos->GetData(), this->Dt, this->Bds);
    updateCoords(0, this->NPart, this->YPos->GetData(), this->Dt, this->Bds);
    updateCoords(0, this->NPart, this->ZPos->GetData(), this->Dt, this->Bds);
  }

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
    md->BlockArrayRange = {{{-this->Bds, this->Bds}, {-this->Bds, this->Bds}, {-this->Bds, this->Bds}, {-this->Bds, this->Bds}}};
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

    if (arrayName == "mass")
    {
      blk->AddColumn(this->Mass);
    }
    else if (arrayName == "xpos")
    {
      blk->AddColumn(this->XPos);
    }
    else if (arrayName == "ypos")
    {
      blk->AddColumn(this->YPos);
    }
    else if (arrayName == "zpos")
    {
      blk->AddColumn(this->ZPos);
    }
    else
    {
        SENSEI_ERROR("No array named \"" << arrayName << "\"")
        return -1;
    }

    return 0;
  }

  void GenerateData()
  {
  }

  ~ParticleSource()
  {
    if (this->XPos) this->XPos->Delete();
    if (this->YPos) this->YPos->Delete();
    if (this->ZPos) this->ZPos->Delete();
    if (this->Mass) this->Mass->Delete();

#if defined(SENSEI_ENABLE_CUDA)
    for (int i = 0; i < 4; ++i)
      cudaStreamDestroy(this->Stream[i]);
#endif
  }

protected:
  ParticleSource(long nPart, double bds, double dt) : NPart(nPart), Bds(bds),
    Dt(dt), XPos(nullptr), YPos(nullptr), ZPos(nullptr), Mass(nullptr)
  {
#if defined(SENSEI_ENABLE_CUDA)
    for (int i = 0; i < 4; ++i)
    {
      cudaStream_t strm;
      cudaStreamCreate(&strm);
      this->Stream[i] = strm;
    }
#endif
    auto hostAlloc = svtkAllocator::malloc;
#if defined(SENSEI_ENABLE_CUDA)
    hostAlloc = svtkAllocator::cuda_host;
    cudaSetDevice(0);
#endif
    auto xpos = svtkHAMRDoubleArray::New("xpos", this->NPart,
             1, hostAlloc, this->Stream[0], svtkStreamMode::async, 1.);
    this->XPos = xpos;

    auto ypos = svtkHAMRDoubleArray::New("ypos", this->NPart,
             1, hostAlloc, this->Stream[0], svtkStreamMode::async, 1.);
    this->YPos = ypos;

    auto zpos = svtkHAMRDoubleArray::New("zpos", this->NPart,
             1, hostAlloc, this->Stream[0], svtkStreamMode::async, 1.);
    this->ZPos = zpos;

    this->Mass = svtkHAMRDoubleArray::New("mass", this->NPart,
                   1, hostAlloc, this->Stream[3], svtkStreamMode::async, 1.);

    genCoords(0, this->NPart, this->XPos->GetData(), 0.0, this->Bds/3., this->Bds);
    genCoords(0, this->NPart, this->YPos->GetData(), 0.0, this->Bds/3., this->Bds);
    genCoords(0, this->NPart, this->ZPos->GetData(), 0.0, this->Bds/3., this->Bds);
  }

  long NPart; ///< total number of particles on all ranks
  double Bds; ///< data is defined on a cube +/- Bds in all directions
  double Dt;  ///< time step

  svtkStream Stream[4];

  svtkHAMRDoubleArray *XPos; ///< particle position
  svtkHAMRDoubleArray *YPos;
  svtkHAMRDoubleArray *ZPos;
  svtkHAMRDoubleArray *Mass; ///< particle mass
};



int main(int argc, char **argv)
{
  int threadLevel;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadLevel);
  if (threadLevel != MPI_THREAD_MULTIPLE)
  {
    std::cerr << "ERROR: MPI_THREAD_MULTIPLE is required for this program" << std::endl;
    return -1;
  }

  if (argc < 9)
  {
    std::cerr << "testDataBinning [nit] [np] [dt] [x/y res] [odir] [dev] [async] "
                 "[x axis] [y axis] [op 1] ... [op n]" << std::endl;
    return -1;
  }

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  long nit = atoi(argv[1]);
  long np = atoi(argv[2]);
  long res = atoi(argv[3]);
  const char *odir = argv[4];
  int device = atoi(argv[5]);
  int async = atoi(argv[6]);
  const char *xAxis = argv[7];
  const char *yAxis = argv[8];

  double a = 10.; // box goes from -a to a
  double dt = cos( 3.141592 / 4. ) * 2. * sqrt(2.) * a / (nit - 1); // finish at the start

  std::vector<std::string> array;
  std::vector<std::string> op;

  for (int i = 9; i < argc; ++i)
  {
    array.push_back(xAxis);
    array.push_back(yAxis);
    array.push_back("mass");

    op.push_back(argv[i]);
    op.push_back(argv[i]);
    op.push_back(argv[i]);
  }

#if defined(SENSEI_ENABLE_CUDA)
  cudaSetDevice(0);
#endif

  // generate some particle data
  auto da = ParticleSource::New(np, a, dt);

  // process
  auto aa = sensei::DataBinning::New();
  aa->SetDeviceId(device);
  aa->SetAsynchronous(async);
  aa->SetVerbose(1);
  aa->Initialize("particles", xAxis, yAxis, array, op, res, res, odir, 0, 8);

  for (long i = 0; i < nit; ++i)
  {
    da->GenerateData();
    aa->Execute(da, nullptr);
    da->Increment();
  }

  aa->Finalize();

  // clean up
  aa->Delete();
  da->Delete();

  MPI_Finalize();

  return 0;
}
