/*!
 * This program is designed to be an endpoint component in a scientific
 * workflow. It can read a data-stream using ADIOS-FLEXPATH. When enabled, this end point
 * supports histogram and catalyst-slice analysis via the Sensei infrastructure.
 *
 * Usage:
 *  <exec> input-stream-name
 */
#include <opts/opts.h>
#include <mpi.h>
#include <iostream>
#include <adios.h>
#include <adios_read.h>
#include <vtkADIOSDataAdaptor.h>
#include <vtkInsituAnalysisAdaptor.h>
#include <vtkNew.h>
#include <vtkDataSet.h>

#ifdef ENABLE_HISTOGRAM
#include "HistogramAnalysisAdaptor.h"
#endif
#ifdef ENABLE_CATALYST
#include <vtkCatalystAnalysisAdaptor.h>
# ifdef ENABLE_CATALYST_SLICE
#include <vtkCatalystSlicePipeline.h>
# endif
#endif

using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char** argv)
{
  int rank, size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Init (&argc, &argv);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::string input;
  std::string readmethod("bp");

  opts::Options ops(argc, argv);
  ops >> opts::Option('r', "readmethod", readmethod, "specify read method: bp, bp_aggregate, dataspaces, dimes, or flexpath ");
#ifdef ENABLE_HISTOGRAM
  std::string histogram;
  ops >> opts::Option('H', "histogram", histogram, "cell array to histogram with");
  int bins=10;
  ops >> opts::Option('b', "bins", bins, "number of bins");
#endif
#ifdef ENABLE_CATALYST_SLICE
  std::string colorarray;
  ops >> opts::Option('c', "color", colorarray, "cell array to color the slice with");
#endif
  if (ops >> opts::Present('h', "help", "show help") ||
    !(ops >> opts::PosOption(input)))
    {
    if (rank == 0)
      {
      cout << "Usage: " << argv[0] << "[OPTIONS] input-stream-name\n\n" << ops;
      }
    MPI_Barrier(comm);
    return 1;
    }

  std::map<std::string, ADIOS_READ_METHOD> readmethods;
  readmethods["bp"] = ADIOS_READ_METHOD_BP;
  readmethods["bp_aggregate"] = ADIOS_READ_METHOD_BP_AGGREGATE;
  readmethods["dataspaces"] = ADIOS_READ_METHOD_DATASPACES;
  readmethods["dimes"] = ADIOS_READ_METHOD_DIMES;
  readmethods["flexpath"] = ADIOS_READ_METHOD_FLEXPATH;

  std::vector<vtkSmartPointer<vtkInsituAnalysisAdaptor> > analyses;
#ifdef ENABLE_HISTOGRAM
  if (!histogram.empty())
    {
    vtkNew<HistogramAnalysisAdaptor> histAA;
    histAA->Initialize(comm, bins, vtkDataObject::FIELD_ASSOCIATION_CELLS, histogram);
    analyses.push_back(histAA.GetPointer());
    }
#endif
#ifdef ENABLE_CATALYST
  vtkNew<vtkCatalystAnalysisAdaptor> catalyst;
  analyses.push_back(catalyst.GetPointer());
# ifdef ENABLE_CATALYST_SLICE
  vtkNew<vtkCatalystSlicePipeline> slicePipeline;
  slicePipeline->SetSliceNormal(0, 0, 1);
  if (!colorarray.empty())
    {
    slicePipeline->ColorBy(vtkDataObject::FIELD_ASSOCIATION_CELLS, colorarray.c_str());
    }
  catalyst->AddPipeline(slicePipeline.GetPointer());
# endif
#endif

  vtkNew<vtkADIOSDataAdaptor> dataAdaptor;
  dataAdaptor->Open(comm, readmethods[readmethod], input);
  do
    {
    // request reading of meta-data for this step.
    dataAdaptor->ReadStep();
    if (rank == 0)
      {
      cout << "TimeStep: " << dataAdaptor->GetDataTimeStep()
        << " Time: " << dataAdaptor->GetDataTime() << endl;
      }
    if (analyses.size() == 0)
      {
      dataAdaptor->GetCompleteMesh()->Print(cout);
      }
    else
      {
#ifdef ENABLE_CATALYST_SLICE
      // set slice origin.
      if (vtkDataSet* ds = vtkDataSet::SafeDownCast(dataAdaptor->GetMesh(true)))
        {
        double bounds[6];
        ds->GetBounds(bounds);
        slicePipeline->SetSliceOrigin(
          (bounds[0] + bounds[1]) / 2.0,
          (bounds[2] + bounds[3]) / 2.0,
          (bounds[4] + bounds[5]) / 2.0);
        }
#endif
      for (size_t cc=0, max=analyses.size(); cc < max; ++cc)
        {
        analyses[cc]->Execute(dataAdaptor.GetPointer());
        }
      }
    dataAdaptor->ReleaseData();
    }
  while (dataAdaptor->Advance());
  return 0;
}
