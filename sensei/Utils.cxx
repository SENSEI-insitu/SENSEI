#include <cstdio>
#include <cstring>
#include <errno.h>

#include "Utils.h"
#include "Error.h"


namespace sensei
{

namespace Utils
{

int requireAttributeXML(pugi::xml_node &node, const char *attributeName)
{
	if (!node.attribute(attributeName))
	{
    	SENSEI_ERROR(<< node.name()
    				<< " is missing required attribute " << attributeName)
    	return -1;
    }
  	return 0;
}

//----------------------------------------------------------------------------
int parseXML(MPI_Comm comm, int rank, 
             const std::string &filename, pugi::xml_document &doc)
{
	unsigned long nbytes = 0;
  	char *buffer = nullptr;
  	if(rank == 0)
    {
    	FILE *f = fopen(filename.c_str(), "rb");
    	if (f)
      	{
      		setvbuf(f, nullptr, _IONBF, 0);
      		fseek(f, 0, SEEK_END);
      		nbytes = ftell(f);
      		fseek(f, 0, SEEK_SET);
      		buffer = static_cast<char*>(
      			pugi::get_memory_allocation_function()(nbytes));
      		unsigned long nread = fread(buffer, 1, nbytes, f);
      		fclose(f);
      		if (nread == nbytes)
        	{
        		MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, 0, comm);
        		MPI_Bcast(buffer, nbytes, MPI_CHAR, 0, comm);
        	}
      		else
        	{
        		SENSEI_ERROR("read error on \""  << filename << "\"" << endl
          					<< strerror(errno))
        		nbytes = 0;
        		MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, 0, comm);
        		return -1;
        	}
      	}
    	else
      	{
      		SENSEI_ERROR("failed to open \""  << filename << "\"" << endl
        				<< strerror(errno))
      		MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, 0, comm);
      		return -1;
      	}
    }
  	else
    {
    	MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, 0, comm);
    	if (!nbytes)
        	return -1;
    	buffer = static_cast<char*>(
    		pugi::get_memory_allocation_function()(nbytes));
    	MPI_Bcast(buffer, nbytes, MPI_CHAR, 0, comm);
    }
  	pugi::xml_parse_result result = doc.load_buffer_inplace_own(buffer, nbytes);
  	if (!result)
    {
    	SENSEI_ERROR("XML [" << filename << "] parsed with errors, attr value: ["
      			<< doc.child("node").attribute("attr").value() << "]" << endl
      			<< "Error description: " << result.description() << endl
      			<< "Error offset: " << result.offset << endl)
    	return -1;
    }

  	return 0;
}

}

}
