

#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <pugixml.hpp>

#include <opts/opts.h>

#include <ConfigurableAnalysis.h>
#include <InTransitAdaptorFactory.h>
#include <vtkSmartPointer.h>
#include <XMLUtils.h>
#include <Error.h>


using ConfigAnalysisAdaptorPtr = vtkSmartPointer<sensei::ConfigurableAnalysis>;

using namespace sensei;


/*! Designed to be a configurable endpoint in SENSEI 3.0 that supports different (in-transit) 
 * transport layers and analysis adaptors.
 *
 * Usage: sensei_end_point <xml config> <input stream name>
 */

int main(int argc, char** argv)
{
  int myRank, numRanks;

  MPI_Init(&argc, &argv);
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &myRank);
  MPI_Comm_size(comm, &numRanks);

  std::string configFile;

  opts::Options ops(argc, argv);
  ops >> opts::Option('f', "config", configFile, "Sensei data transport and analysis" 
    "configuration XML (required)");

  bool showHelp = ops >> opts::Present('h', "help", "show help");

  if (!showHelp && configFile.empty() && (myRank == 0))
    SENSEI_ERROR("Missing XML data transport and analysis configuration")

  if (showHelp || configFile.empty())
    {
    if (myRank == 0)
      std::cerr << "Usage: " << argv[0] << " [OPTIONS]\n\n" << ops << std::endl;

    MPI_Finalize();
    return showHelp ? 0 : 1;
    }

  pugi::xml_document doc;
  if (XMLUtils::Parse(comm, configFile, doc))
    {
    if (myRank == 0)
      SENSEI_ERROR("failed to parse configuration")
    MPI_Finalize();
    return 1;
    }

  pugi::xml_node root = doc.child("sensei");

  InTransitDataAdaptor* itda = nullptr;
  if (InTransitAdaptorFactory::Initialize(comm, root, itda))
    {
    if (myRank == 0)
      SENSEI_ERROR("Failed to construct InTransitDataAdaptor")
    MPI_Finalize();
    return 1;
    }

  {
    InTransitDataAdaptorPtr inTranDataAdaptor = itda;
    ConfigAnalysisAdaptorPtr analysisAdaptor = ConfigAnalysisAdaptorPtr::New();

    analysisAdaptor->SetCommunicator(comm);
    if (analysisAdaptor->Initialize(root))
      {
      if (myRank == 0)
        SENSEI_ERROR("Failed to initialize analysis")
      MPI_Finalize();
      return 1;
      }

    inTranDataAdaptor->OpenStream();

    while(inTranDataAdaptor->StreamGood())
      {
      analysisAdaptor->Execute(inTranDataAdaptor.Get());
      inTranDataAdaptor->AdvanceStream();
      }

    inTranDataAdaptor->CloseStream();
    analysisAdaptor->Finalize();
  }
  
  MPI_Finalize();

  return 0;
}
