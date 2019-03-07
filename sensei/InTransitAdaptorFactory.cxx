#include "InTransitAdaptorFactory.h"
//#include "ADIOS1DataAdaptor.h"
#include "BlockPartitioner.h"
#include "CyclicPartitioner.h"
#include "MappedPartitioner.h"
#include "PlanarPartitioner.h"
#include "XMLUtils.h"
#include "Error.h"


namespace sensei
{

namespace InTransitAdaptorFactory
{

int Initialize(MPI_Comm comm, const std::string &fileName, InTransitDataAdaptor *&dataAdaptor)
{
  int myRank = 0;
  MPI_Comm_rank(comm, &myRank);

  pugi::xml_document doc;
  if (XMLUtils::parseXML(comm, myRank, fileName, doc))
    {
    if (myRank == 0)
      SENSEI_ERROR("failed to parse configuration")
    return -1;
    }

  pugi::xml_node root = doc.child("sensei");
  return Initialize(comm, root, dataAdaptor);
}

int Initialize(MPI_Comm comm, const pugi::xml_node &root, InTransitDataAdaptor *&dataAdaptor)
{
  int numRanks = 0, myRank = 0;
  MPI_Comm_rank(comm, &myRank);
  MPI_Comm_size(comm, &numRanks);
  pugi::xml_node node = root.child("data_adaptor");

  std::string type = node.attribute("transport").value();
  if (type == "adios_2")
    {
    // Create ADIOS1DataAdaptor
    // dataAdaptor = ADIOS1DataAdaptor::New();
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
    if (myRank == 0)
      SENSEI_ERROR("Failed to add '" << type << "' data adaptor")
    return -1;
		}

  pugi::xml_node partitionerNode = node.child("partitioner");
  std::string partitionerType = partitionerNode.attribute("type").value();
  Partitioner* partitioner = nullptr;
  if (partitionerType == "block")
    {
    partitioner = new BlockPartitioner();
    }
  else if (partitionerType == "cyclic")
    {
    partitioner = new CyclicPartitioner();
    }
  else if (partitionerType == "planar")
    {
    partitioner = new PlanarPartitioner();
    }
  else if (partitionerType == "mapped")
    {
    partitioner = new MappedPartitioner();
    }
  else
    {
    if (myRank == 0)
      SENSEI_ERROR("Failed to add '" << type << "' data adaptor")
    return -1;
    }

  partitioner->Initialize(partitionerNode);
  dataAdaptor->SetPartitioner(partitioner);
  
  return 0;
}

}   // End of namespace InTransitAdaptorFactory

}   // End of namespace sensei
