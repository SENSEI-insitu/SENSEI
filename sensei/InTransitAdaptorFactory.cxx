#include "InTransitAdaptorFactory.h"
#include "BlockPartitioner.h"
#include "CyclicPartitioner.h"
#include "MappedPartitioner.h"
#include "PlanarPartitioner.h"
#include "Utils.h"
#include "Error.h"


namespace sensei
{

namespace InTransitAdaptorFactory
{
	int Initialize(const std::string &fileName, MPI_Comm comm,
  		AnalysisAdaptor *&analysisAdaptor, InTransitDataAdaptor *&dataAdaptor)
	{
		int rank = 0;
  		MPI_Comm_rank(comm, &rank);

  		pugi::xml_document doc;
  		if (Utils::parseXML(comm, rank, fileName, doc))
    	{
    		if (rank == 0)
        		SENSEI_ERROR("failed to parse configuration")
    		return -1;
    	}

  		pugi::xml_node root = doc.child("sensei");
  		return Initialize(root, analysisAdaptor, dataAdaptor);
	}

	int Initialize(const pugi::xml_node &root, MPI_Comm comm,
  		AnalysisAdaptor *&analysisAdaptor, InTransitDataAdaptor *&dataAdaptor)
	{
		pugi::xml_node node = root.child("data_adaptor");

		std::string type = node.attribute("transport").value();
		if (type == "adios_2")
		{
			// Create Adios InTransitDataAdaptor
			// dataAdaptor = new InTransitAdiosDataAdaptor();
			// dataAdaptor->Initialize(node);
		}
		else if (type == "data_elevators")
		{
			// Create DE InTransitDataAdaptor
		}
		else if (type == "libis")
		{
			// Create LibIS InTransitDataAdaptor
		}
		else
		{
			if (rank == 0)
        		SENSEI_ERROR("Failed to add '" << type << "' data adaptor")
        	return -1;
		}

		pugi::xml_node partitioner_node = node.child("partitioner");
		std::string partitioner_type = partitioner_node.attribute("type").value();
		Partitioner* partitioner = nullptr;
		if (partitioner_type == "block")
		{
			partitioner = new BlockPartitioner();
		}
		else if (partitioner_type == "cyclic")
		{
			partitioner = new CyclicPartitioner();
		}
		else if (partitioner_type == "planar")
		{
			pugi::xml_node size_node = partitioner_node.child("plane_size");
			if (!size_node || !size_node.text())
    		{
    			SENSEI_ERROR("Missing plane_size; failed to initialize InTransitDataAdaptor");
    			return -1;
    		}
			unsigned int plane_size = size_node.text().as_uint();
			partitioner = new PlanarPartitioner(plane_size);
		}
		else if (partitioner_type == "mapped")
		{
			pugi::xml_node blk_owner_node = partitioner_node.child("block_owner");
			pugi::xml_node blk_id_node = partitioner_node.child("block_id");
			if (!blk_owner_node || !blk_owner_node.text() ||
				!blk_id_node || !blk_id_node.text())
    		{
    			SENSEI_ERROR("Missing block_owner and/or block_id; failed to initialize InTransitDataAdaptor");
    			return -1;
    		}
    		std::string blk_owner_line = blk_owner_node.text().as_string();
    		std::string blk_id_line = blk_id_node.text().as_string();
			partitioner = new MappedPartitioner();
		}
		else
		{
			if (rank == 0)
        		SENSEI_ERROR("Failed to add '" << type << "' data adaptor")
        	return -1;
		}

		dataAdaptor->SetPartitioner(partitioner);

		return 0;
	}
}


}
