#ifndef sensei_XMLUtils_h
#define sensei_XMLUtils_h

#include <mpi.h>
#include <string>

namespace pugi { class xml_node; }
namespace pugi { class xml_document; }

namespace sensei
{

namespace XMLUtils
{

// check that an attribute of the passed in name exists
// return of 0 indicates that it does. if not non-zero value
// is returned and an error message is sent to stderr.
int RequireAttribute(pugi::xml_node &node, const char *attributeName);

// Parallel collective read, parse, and distribute the XML file. Rank 0 does
// the I/O and parse and will broadcast to the other ranks in the communicator.
// return of 0 indicates success.
int Parse(MPI_Comm comm, const std::string &filename, pugi::xml_document &doc);

}

}


#endif		// #ifndef Utils_h
