#include "ADIOS1DataAdaptor.h"
#include "ConfigurableAnalysis.h"
#include "Timer.h"
#include "Error.h"

#include <opts/opts.h>

#include <mpi.h>
#include <iostream>
#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkDataSet.h>

using DataAdaptorPtr = vtkSmartPointer<sensei::ADIOS1DataAdaptor>;
using AnalysisAdaptorPtr = vtkSmartPointer<sensei::ConfigurableAnalysis>;


/*!
 * This program is designed to be an endpoint component in a scientific
 * workflow. It can read a data-stream using ADIOS1-FLEXPATH. When enabled, this end point
 * supports histogram and catalyst-slice analysis via the Sensei infrastructure.
 *
 * Usage:
 *  <exec> fileName
 */

using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char **argv)
{
  int rank, size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Init (&argc, &argv);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::string fileName;
  std::string readmethod("bp");
  std::string config_file;

  opts::Options ops(argc, argv);
  ops >> opts::Option('r', "readmethod", readmethod, "specify read method: bp, bp_aggregate, dataspaces, dimes, or flexpath ")
      >> opts::Option('f', "config", config_file, "Sensei analysis configuration xml (required)");

  bool log = ops >> opts::Present("log", "generate time and memory usage log");
  bool shortlog = ops >> opts::Present("shortlog", "generate a summary time and memory usage log");
  bool showHelp = ops >> opts::Present('h', "help", "show help");
  bool haveInput = ops >> opts::PosOption(fileName);

  if (!showHelp && !haveInput && (rank == 0))
    SENSEI_ERROR("Missing ADIOS1 fileName stream")

  if (!showHelp && config_file.empty() && (rank == 0))
    SENSEI_ERROR("Missing XML analysis configuration")

  if (showHelp || !haveInput || config_file.empty())
    {
    if (rank == 0)
      {
      cerr << "Usage: " << argv[0] << "[OPTIONS] fileName\n\n" << ops << endl;
      }
    MPI_Finalize();
    return showHelp ? 0 : 1;
    }

  if (log | shortlog)
    sensei::Timer::Enable(shortlog);

  sensei::Timer::Initialize();

  SENSEI_STATUS("Opening: \"" << fileName.c_str() << "\" using method \""
    << readmethod.c_str() << "\"")

  // open the ADIOS1 stream using the ADIOS1 adaptor
  DataAdaptorPtr dataAdaptor = DataAdaptorPtr::New();

  dataAdaptor->SetCommunicator(comm);
  dataAdaptor->SetReadMethod(readmethod);
  dataAdaptor->SetFileName(fileName);

  if (dataAdaptor->OpenStream())
    {
    SENSEI_ERROR("Failed to open \"" << fileName << "\"")
    MPI_Abort(comm, 1);
    }

  // initlaize the analysis using the XML configurable adaptor
  SENSEI_STATUS("Loading configurable analysis \"" << config_file << "\"")

  AnalysisAdaptorPtr analysisAdaptor = AnalysisAdaptorPtr::New();
  analysisAdaptor->SetCommunicator(comm);
  if (analysisAdaptor->Initialize(config_file))
    {
    SENSEI_ERROR("Failed to initialize analysis")
    MPI_Abort(comm, 1);
    }

  // read from the ADIOS1 stream until all steps have been
  // processed
  unsigned int nSteps = 0;
  do
    {
    // gte the current simulation time and time step
    long timeStep = dataAdaptor->GetDataTimeStep();
    double time = dataAdaptor->GetDataTime();
    nSteps += 1;

    sensei::Timer::MarkStartTimeStep(timeStep, time);

    SENSEI_STATUS("Processing time step " << timeStep << " time " << time)

    // execute the analysis
    sensei::Timer::MarkStartEvent("AnalysisAdaptor::Execute");
    if (!analysisAdaptor->Execute(dataAdaptor.Get()))
      {
      SENSEI_ERROR("Execute failed")
      MPI_Abort(comm, 1);
      }
    sensei::Timer::MarkEndEvent("AnalysisAdaptor::Execute");

    // let the data adaptor release the mesh and data from this
    // time step
    dataAdaptor->ReleaseData();

    sensei::Timer::MarkEndTimeStep();
    }
  while (!dataAdaptor->AdvanceStream());

  SENSEI_STATUS("Finished processing " << nSteps << " time steps")

  // close the ADIOS1 stream
  dataAdaptor->CloseStream();
  dataAdaptor->Finalize();

  analysisAdaptor->Finalize();

  // we must force these to be destroyed before mpi finalize
  // some of the adaptors make MPI calls in the destructor
  // noteabley Catalyst
  dataAdaptor = nullptr;
  analysisAdaptor = nullptr;

  sensei::Timer::Finalize();

  MPI_Finalize();

  return 0;
}
