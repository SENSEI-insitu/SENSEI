#include "InTransitAdaptorFactory.h"
//#include "ADIOS1DataAdaptor.h"
#include "XMLUtils.h"
#include "Error.h"

#include <pugixml.hpp>

namespace sensei
{

namespace InTransitAdaptorFactory
{

int Initialize(MPI_Comm comm, const std::string &fileName, InTransitDataAdaptor *&dataAdaptor)
{
  int myRank = 0;
  MPI_Comm_rank(comm, &myRank);

  pugi::xml_document doc;
  if (XMLUtils::Parse(comm, fileName, doc))
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
  int myRank = 0;
  MPI_Comm_rank(comm, &myRank);

  pugi::xml_node node = root.child("data_adaptor");
  // TODO -- hande error of no data adaptor node is found

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

  if (dataAdaptor->Initialize(node))
    {
    SENSEI_ERROR("Failed to initialize \"" << type << "\" data adaptor")
    return -1;
    }

  return 0;
}

}   // End of namespace InTransitAdaptorFactory

}   // End of namespace sensei
