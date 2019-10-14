#include "ConfigurableInTransitDataAdaptor.h"
#include "ConfigurableAnalysis.h"
#include "MPIManager.h"
#include "Profiler.h"
#include "Error.h"

#include <opts/opts.h>

#include <mpi.h>
#include <iostream>
#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkDataSet.h>

using DataAdaptorPtr = vtkSmartPointer<sensei::ConfigurableInTransitDataAdaptor>;
using AnalysisAdaptorPtr = vtkSmartPointer<sensei::ConfigurableAnalysis>;

int main(int argc, char **argv)
{
  sensei::MPIManager mpiMan(argc, argv);
  int rank = mpiMan.GetCommRank();

  std::string transportXml;
  std::string analysisXml;
  std::string connectionInfo;

  opts::Options ops(argc, argv);

  ops >> opts::Option('t', "transport-xml", transportXml,
         "SENSEI transport XML configuration file")

    >> opts::Option('a', "analysis-xml", analysisXml,
      "SENSEI analysis XML configuration file")

    >> opts::Option('c', "connection-info", connectionInfo,
       "transport specific conncetion information");

  if (ops >> opts::Present('h', "help", "show help"))
    {
    if (rank == 0)
      cerr << "Usage: EndPoint [OPTIONS]\n\n" << ops << endl;
    MPI_Abort(MPI_COMM_WORLD, -1);
    }

  if (transportXml.empty() || analysisXml.empty())
    {
    SENSEI_ERROR("Missing " << (transportXml.empty() ?
      (analysisXml.empty() ? "transport and analysis XML" :
      "transport XML") : "analysis XML"))
    MPI_Abort(MPI_COMM_WORLD, 1);
    }

  // create the reead side of the transport
  SENSEI_STATUS("Creating transport data adaptor. transport-xml=\""
    << transportXml << "\"")

  DataAdaptorPtr dataAdaptor = DataAdaptorPtr::New();
  if (dataAdaptor->SetConnectionInfo(connectionInfo) ||
    dataAdaptor->Initialize(transportXml))
    {
    SENSEI_ERROR("Failed to initialize the transport data adaptor")
    MPI_Abort(MPI_COMM_WORLD, -1);
    }

  // connect and open the stream
  if (dataAdaptor->OpenStream())
    {
    SENSEI_ERROR("Failed to open stream. connection-info=\""
      << connectionInfo << "\"")
    MPI_Abort(MPI_COMM_WORLD, -1);
    }

  // initlaize the analysis using the XML configurable adaptor
  SENSEI_STATUS("Creating the analysis adaptor. analysis-xml=\""
    << analysisXml << "\"")

  AnalysisAdaptorPtr analysisAdaptor = AnalysisAdaptorPtr::New();
  if (analysisAdaptor->Initialize(analysisXml))
    {
    SENSEI_ERROR("Failed to initialize analysis adaptor")
    MPI_Abort(MPI_COMM_WORLD, -1);
    }

  // read from the stream until all steps have been
  // processed
  unsigned int nSteps = 0;
  do
    {
    // gte the current simulation time and time step
    long timeStep = dataAdaptor->GetDataTimeStep();
    double time = dataAdaptor->GetDataTime();
    nSteps += 1;

    SENSEI_STATUS("Processing time step " << timeStep << " time " << time)

    // execute the analysis
    if (!analysisAdaptor->Execute(dataAdaptor.Get()))
      {
      SENSEI_ERROR("Execute failed")
      MPI_Abort(MPI_COMM_WORLD, -1);
      }

    // let the data adaptor release the mesh and data from this
    // time step
    dataAdaptor->ReleaseData();
    }
  while (!dataAdaptor->AdvanceStream());

  SENSEI_STATUS("Finished processing " << nSteps << " time steps")

  // close the ADIOS1 stream
  dataAdaptor->CloseStream();
  dataAdaptor->Finalize();

  analysisAdaptor->Finalize();

  // we must force these to be destroyed before mpi finalize some of the analysis
  // adaptors (eg Catalyst) make MPI calls in the destructor
  dataAdaptor = nullptr;
  analysisAdaptor = nullptr;

  return 0;
}
