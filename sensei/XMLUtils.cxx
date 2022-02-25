#include "XMLUtils.h"
#include "Error.h"

#include <pugixml.hpp>

#include <cstdio>
#include <cstring>
#include <errno.h>


namespace sensei
{

namespace XMLUtils
{

//----------------------------------------------------------------------------
int RequireAttribute(const pugi::xml_node &node, const char *attributeName)
{
  if (!node.attribute(attributeName))
    {
    SENSEI_ERROR(<< node.name() << " is missing required attribute " << attributeName)
    return -1;
    }
  return 0;
}

//----------------------------------------------------------------------------
int RequireChild(const pugi::xml_node &node, const char *childName)
{
  if (!node.child(childName))
    {
    SENSEI_ERROR(<< node.name() << " is missing required child element " << childName)
    return -1;
    }
  return 0;
}

//----------------------------------------------------------------------------
int Parse(MPI_Comm comm, const std::string &filename, pugi::xml_document &doc)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);

  unsigned long nbytes = 0;
  char *buffer = nullptr;
  if (rank == 0)
    {
    FILE *f = fopen(filename.c_str(), "rb");
    if (f)
      {
      setvbuf(f, nullptr, _IONBF, 0);
      fseek(f, 0, SEEK_END);
      nbytes = ftell(f);
      fseek(f, 0, SEEK_SET);
      buffer = static_cast<char*>(pugi::get_memory_allocation_function()(nbytes));
      unsigned long nread = fread(buffer, 1, nbytes, f);
      fclose(f);
      if (nread == nbytes)
        {
        MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, 0, comm);
        MPI_Bcast(buffer, nbytes, MPI_CHAR, 0, comm);
        }
      else
        {
        SENSEI_ERROR("read error on \""  << filename << "\"" << endl << strerror(errno))
        nbytes = 0;
        MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, 0, comm);
        return -1;
        }
      }
    else
      {
      SENSEI_ERROR("failed to open \""  << filename << "\"" << endl << strerror(errno))
      MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, 0, comm);
      return -1;
      }
    }
  else
    {
    MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, 0, comm);
    if (!nbytes)
      return -1;
    buffer = static_cast<char*>(pugi::get_memory_allocation_function()(nbytes));
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

//----------------------------------------------------------------------------
int ParseNameValuePairs(const pugi::xml_node &node,
  std::vector<std::string> &names, std::vector<std::string> &values)
{
  std::string strData = node.text().as_string();

  std::string delims = " ,\t\n";

  std::size_t curr = strData.find_first_not_of(delims, 0);
  std::size_t eq = std::string::npos;
  std::size_t val = std::string::npos;
  std::size_t valEnd = std::string::npos;

  while (curr != std::string::npos)
    {
    // locate the equal sign
    if ((eq = strData.find_first_of("=", curr + 1)) == std::string::npos)
      {
      SENSEI_ERROR("Failed to locate the \"=\" in \"name = value\" at \""
        << strData.substr(curr, std::string::npos) << "\"")
      break;
      }

    // locate the value
    if ((val = strData.find_first_not_of(delims, eq + 1)) == std::string::npos)
      {
      SENSEI_ERROR("Failed to locate the \"value\" in \"name = value\" at \""
        << strData.substr(curr, std::string::npos) << "\"")
      break;
      }

    // locate the end of the  name value pair
    valEnd = strData.find_first_of(delims, val + 1);

    // extract the name and its value
    std::string tmp = strData.substr(curr,
      valEnd == std::string::npos ? strData.size() - curr : valEnd - curr);

    char name[128];
    char value[128];

    name[127] = '\0';
    value[127] = '\0';

    if (sscanf(tmp.c_str(), "%127s = %127s", name, value) == 2)
      {
      names.push_back(name);
      values.push_back(value);
      }
    else
      {
      SENSEI_WARNING("Failed to parse \"name = value\" in \"" << tmp << "\"")
      }

    // advance to next name value pair
    curr = valEnd == std::string::npos ? valEnd :
      strData.find_first_not_of(delims, valEnd + 1);
    }

  return 0;
}

}
}
