#ifndef XMLUtils_h
#define XMLUtils_h

#include <mpi.h>
#include <string>
#include <pugixml.hpp>


namespace sensei
{

namespace XMLUtils
{

int requireAttributeXML(pugi::xml_node &node, const char *attributeName);
int parseXML(MPI_Comm comm, int rank, const std::string &filename, pugi::xml_document &doc);

}

}


#endif		// #ifndef Utils_h
