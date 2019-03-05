

#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <pugixml.hpp>

#include <opts/opts.h>

#include <ConfigurableAnalysis.h>
#include <InTransitDataAdaptorFactory.h>
#include <InTransitAdaptorFactory.h>
#include <vtkSmartPointer.h>
#include <Utils.h>
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
	if(argc < 3)
	{
		std::cout << "Usage: sensei_end_point <xml config> <stream id>" << std::endl;
		std::exit(-1);
	}

	int my_rank, n_ranks;

	MPI_Init(&argc, &argv);
	MPI_Comm comm = MPI_COMM_WORLD;
	MPI_Comm_rank(comm, &my_rank);
	MPI_Comm_size(comm, &n_ranks);

	std::string config_file;
	std::string stream_id;

	opts::Options ops(argc, argv);
	ops >> opts::Option('f', "config", config_file, "Sensei data transport and analysis configuration XML (required)")
	    >> opts::Option('s', "stream", stream_id, "Input stream name (required)");

	bool showHelp = ops >> opts::Present('h', "help", "show help");

	if (!showHelp && stream_id.empty() && (rank == 0))
		SENSEI_ERROR("Missing input stream")

  	if (!showHelp && config_file.empty() && (rank == 0))
  		SENSEI_ERROR("Missing XML data transport and analysis configuration")

	if(showHelp || config_file.empty() || stream_id.empty())
	{
		if(rank == 0)
		{
			std::cerr << "Usage: " << argv[0] << "[OPTIONS]\n\n" << ops << std::endl;
		}
		MPI_Finalize();
    	return showHelp ? 0 : 1;
	}

	pugi::xml_document doc;
	if (Utils::parseXML(comm, my_rank, config_file, doc))
    {
    	if (my_rank == 0)
        	SENSEI_ERROR("failed to parse configuration")
        MPI_Finalize();
   		return 1;
    }

    pugi::xml_node root = doc.child("sensei");

	InTransitDataAdaptor* itda = nullptr;
	InTransitAdaptorFactory::Initialize(root, comm, itda);

	InTransitDataAdaptorPtr inTranDataAdaptor = itda;
	ConfigAnalysisAdaptorPtr analysisAdaptor = ConfigAnalysisAdaptorPtr::New();

  	analysisAdaptor->SetCommunicator(comm);
  	if (analysisAdaptor->Initialize(root))
    {
    	SENSEI_ERROR("Failed to initialize analysis")
    	MPI_Abort(comm, 1);
    }

	inTranDataAdaptor->OpenStream(stream_id);

	// Get metadata
	inTranDataAdaptor->InititalizeMetadata();

	while(inTranDataAdaptor->StreamGood())
	{
		analysisAdaptor->Execute(inTranDataAdaptor.Get());

		inTranDataAdaptor->AdvanceStream();
	}

	inTranDataAdaptor->CloseStream();

	MPI_Finalize():

	return 0;
}
