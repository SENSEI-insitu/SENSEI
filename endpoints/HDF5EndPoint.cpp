#include "ConfigurableAnalysis.h"
#include "Error.h"
#include "HDF5DataAdaptor.h"
#include "Timer.h"

#include <opts/opts.h>

#include <iostream>
#include <mpi.h>
#include <vtkDataSet.h>
#include <vtkNew.h>
#include <vtkSmartPointer.h>

using DataAdaptorPtr = vtkSmartPointer<sensei::HDF5DataAdaptor>;
using AnalysisAdaptorPtr = vtkSmartPointer<sensei::ConfigurableAnalysis>;

/*!
 * This program is designed to be an endpoint component in a scientific
 * workflow. It can read a data-stream using HDF5. When enabled, this end point
 * supports histogram and catalyst-slice analysis via the Sensei infrastructure.
 *
 * Usage:
 *  <exec> input-stream-name
 */

using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char **argv) {
  int rank, size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::string input;
  std::string readmethod("n"); // no-streaming
  std::string config_file;

  opts::Options ops(argc, argv);
  ops >> opts::Option('r', "readmethod", readmethod,
                      "specify read method: s(=streaming) n(=nostreaming)") >>
      opts::Option('f', "config", config_file,
                   "Sensei analysis configuration xml (required)");

  bool log = ops >> opts::Present("log", "generate time and memory usage log");
  bool shortlog =
      ops >>
      opts::Present("shortlog", "generate a summary time and memory usage log");
  bool showHelp = ops >> opts::Present('h', "help", "show help");
  bool haveInput = ops >> opts::PosOption(input);

  bool doStreaming = ('s' == readmethod[0]);
  bool doCollectiveTxf = ((readmethod.size() > 1) && ('c' == readmethod[1]));

  if (!showHelp && !haveInput && (rank == 0))
    SENSEI_ERROR("Missing HDF5 input stream");

  if (!showHelp && config_file.empty() && (rank == 0))
    SENSEI_ERROR("Missing XML analysis configuration");

  if (showHelp || !haveInput || config_file.empty()) {
    if (rank == 0) {
      cerr << "Usage: " << argv[0] << "[OPTIONS] input-stream-name\n\n"
           << ops << endl;
    }
    MPI_Finalize();
    return showHelp ? 0 : 1;
  }

  if (log | shortlog)
    timer::Enable(shortlog);

  timer::Initialize();

  DataAdaptorPtr dataAdaptor = DataAdaptorPtr::New();
  dataAdaptor->SetCommunicator(comm);
  dataAdaptor->SetStreaming(doStreaming);
  dataAdaptor->SetCollective(doCollectiveTxf);
  dataAdaptor->SetStreamName(input);

  if (dataAdaptor->OpenStream()) {
    SENSEI_ERROR("Failed to open \"" << input << "\"");
    MPI_Abort(comm, 1);
  }

  // initlaize the analysis using the XML configurable adaptor
  SENSEI_STATUS("Loading configurable analysis \"" << config_file << "\"");

  if (rank == 0) {
    SENSEI_STATUS("... streaming? " << doStreaming);
  }

  AnalysisAdaptorPtr analysisAdaptor = AnalysisAdaptorPtr::New();
  analysisAdaptor->SetCommunicator(comm);
  if (analysisAdaptor->Initialize(config_file)) {
    SENSEI_ERROR("Failed to initialize analysis");
    MPI_Abort(comm, 1);
  }

  // read from the HDF5 stream until all steps have been
  // processed
  unsigned int nSteps = 0;
  do {
    // gte the current simulation time and time step
    long timeStep = dataAdaptor->GetDataTimeStep();
    double time = dataAdaptor->GetDataTime();
    nSteps += 1;

    timer::MarkStartTimeStep(timeStep, time);

    SENSEI_STATUS("Processing time step " << timeStep << " time " << time);

    // execute the analysis
    timer::MarkStartEvent("AnalysisAdaptor::Execute");
    if (!analysisAdaptor->Execute(dataAdaptor.Get())) {
      SENSEI_ERROR("Execute failed");
      MPI_Abort(comm, 1);
    }
    timer::MarkEndEvent("AnalysisAdaptor::Execute");

    // let the data adaptor release the mesh and data from this
    // time step
    dataAdaptor->ReleaseData();

    timer::MarkEndTimeStep();
  } while (!dataAdaptor->AdvanceStream());

  SENSEI_STATUS("Finished processing " << nSteps << " time steps")

  // close the HDF5 stream
  dataAdaptor->CloseStream();
  analysisAdaptor->Finalize();

  // we must force these to be destroyed before mpi finalize
  // some of the adaptors make MPI calls in the destructor
  // noteabley Catalyst
  dataAdaptor = nullptr;
  analysisAdaptor = nullptr;

  timer::Finalize();

  MPI_Finalize();

  return 0;
}
