#include "InTransitAdaptorFactory.h"

#ifdef ENABLE_ADIOS1
#include "ADIOS1DataAdaptor.h"
#endif

#ifdef ENABLE_HDF5
#include "HDF5DataAdaptor.h"
#endif

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

  if (XMLUtils::RequireChild(root, "data_adaptor"))
    {
    SENSEI_ERROR(
      "Failed to construct an InTransitDataAdaptor. Missing \"data_adaptor\" element");
    return -1;
    }

  pugi::xml_node node = root.child("data_adaptor");
  
  if (XMLUtils::RequireAttribute(node, "transport"))
    {
    SENSEI_ERROR(
      "Failed to construct an InTransitDataAdaptor. Missing \"transport\" attribute");
    return -1;
    }

  std::string type = node.attribute("transport").value();
  if (type == "adios_1")
    {
#ifndef ENABLE_ADIOS1
    SENSEI_ERROR("ADIOS 1 transport requested but is disabled in this build")
    return -1;
#else
    dataAdaptor = ADIOS1DataAdaptor::New();
#endif
    }
  else if (type == "hdf5")
    {
#ifndef ENABLE_HDF5
    SENSEI_ERROR("HDF5 transport requested but is disabled in this build")
    return -1;
#else
    dataAdaptor = HDF5DataAdaptor::New();
#endif
    }
  else if (type == "libis")
    {
    // Create LibIS InTransitDataAdaptor
    }
  else
    {
    SENSEI_ERROR("Failed to add \"" << type << "\" data adaptor")
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
