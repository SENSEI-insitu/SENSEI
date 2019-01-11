

#include <iostream>
#include <cstdlib>
#include <mpi.h>
#include <pugixml.hpp>

#include <ConfigurableAnalysis.h>
#include <InTransitDataAdaptorFactory.h>
#include <InTransitAdaptorFactory.h>


using namespace sensei;


int main(int argc, char** argv)
{
	if(argc < 3)
	{
		std::cout << "Usage: sensei_end_point <xml config> <stream id>" << std::endl;
		std::exit(-1);
	}

	int my_rank, n_ranks;

	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

	std::string xml_config = argv[1];
	std::string stream_id = argv[2];

	ConfigurableAnalysis* aa = nullptr;
	InTransitDataAdaptor* itda = nullptr;

	InTransitAdaptorFactory::Initialize(xml_config, aa, itda);

	itda->OpenStream(stream_id);

	// Get metadata
	itda->InititalizeMetadata();

	while(da->StreamGood())
	{
		aa->Execute(itda);

		itda->AdvanceStream();
	}

	itda->CloseStream();

	MPI_Finalize():

	return 0;
}