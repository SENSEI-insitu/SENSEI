#ifndef Utils_h
#define Utils_h

#include <mpi.h>
#include <pugixml.hpp>
#include <string>


namespace sensei
{

namespace Utils
{

int requireAttributeXML(pugi::xml_node &node, const char *attributeName);
int parseXML(MPI_Comm comm, int rank, const std::string &filename, pugi::xml_document &doc);

}

}


#endif		// #ifndef Utils_h
