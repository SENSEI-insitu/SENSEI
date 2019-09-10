#include "libISDataAdaptor.h"
#include "ConfigurableAnalysis.h"
#include "Timer.h"
#include "Error.h"
#include "is_client.h"
#include <opts/opts.h>
#include <mpi.h>
#include <iostream>
#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkDataSet.h>

using DataAdaptorPtr = vtkSmartPointer<sensei::libISDataAdaptor>;
using AnalysisAdaptorPtr = vtkSmartPointer<sensei::ConfigurableAnalysis>;


/*!
 * This program is designed to be an endpoint component in a scientific
 * workflow. It can read a data-stream using libIS. When enabled, this end point
 * supports analysis via the Sensei infrastructure.
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
  std::string port("29374");
  std::string config_file;

  opts::Options ops(argc, argv);
  ops >> opts::Option('p', "port", port, "specify TCP port for communication ")
      >> opts::Option('f', "config", config_file, "Sensei analysis configuration xml (required)");

  bool log = ops >> opts::Present("log", "generate time and memory usage log");
  bool shortlog = ops >> opts::Present("shortlog", "generate a summary time and memory usage log");
  bool showHelp = ops >> opts::Present('h', "help", "show help");
  //bool haveInput = ops >> opts::PosOption(fileName);

  //if (!showHelp && !haveInput && (rank == 0))
  //  SENSEI_ERROR("Missing ADIOS1 fileName stream")

  if (!showHelp && config_file.empty() && (rank == 0))
    SENSEI_ERROR("Missing XML analysis configuration")

  if (showHelp || config_file.empty())
    {
    if (rank == 0)
      {
      cerr << "Usage: " << argv[0] << "[OPTIONS] \n\n" << ops << endl;
      }
    MPI_Finalize();
    return showHelp ? 0 : 1;
    }

  if (log | shortlog)
    timer::Enable(shortlog);

  timer::Initialize();

  //SENSEI_STATUS("Opening: \"" << fileName.c_str() << "\" using method \""
  //  << readmethod.c_str() << "\"")

  // open the stream using the libIS adaptor
  DataAdaptorPtr dataAdaptor = DataAdaptorPtr::New();

  dataAdaptor->SetCommunicator(comm);
  //dataAdaptor->SetReadMethod(readmethod);
  //dataAdaptor->SetFileName(fileName);

  if (dataAdaptor->OpenStream())
    {
    SENSEI_ERROR("Failed to open libIS stream")
    MPI_Abort(comm, 1);
    }

  // initialize the analysis using the XML configurable adaptor
  SENSEI_STATUS("Loading configurable analysis \"" << config_file << "\"")

  AnalysisAdaptorPtr analysisAdaptor = AnalysisAdaptorPtr::New();
  analysisAdaptor->SetCommunicator(comm);
  if (analysisAdaptor->Initialize(config_file))
    {
    SENSEI_ERROR("Failed to initialize analysis")
    MPI_Abort(comm, 1);
    }

  // read from the libIS stream until all steps have been
  // processed
  unsigned int nSteps = 0;
  do
    {
    // get the current simulation time and time step
    long timeStep = dataAdaptor->GetDataTimeStep();
    double time = dataAdaptor->GetDataTime();
    nSteps += 1;

    timer::MarkStartTimeStep(timeStep, time);

    SENSEI_STATUS("Processing time step " << timeStep << " time " << time)

    // execute the analysis
    timer::MarkStartEvent("AnalysisAdaptor::Execute");
    if (!analysisAdaptor->Execute(dataAdaptor.Get()))
      {
      SENSEI_ERROR("Execute failed")
      MPI_Abort(comm, 1);
      }
    timer::MarkEndEvent("AnalysisAdaptor::Execute");

    // let the data adaptor release the mesh and data from this
    // time step
    dataAdaptor->ReleaseData();

    timer::MarkEndTimeStep();
    }
  while (!dataAdaptor->AdvanceStream());

  SENSEI_STATUS("Finished processing " << nSteps << " time steps")

  // close the libIS stream
  dataAdaptor->CloseStream();
  dataAdaptor->Finalize();

  analysisAdaptor->Finalize();

  // we must force these to be destroyed before mpi finalize
  // some of the adaptors make MPI calls in the destructor
  // notably Catalyst
  dataAdaptor = nullptr;
  analysisAdaptor = nullptr;

  timer::Finalize();

  MPI_Finalize();

  return 0;
}
