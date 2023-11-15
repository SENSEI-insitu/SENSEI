#ifndef sensei_XMLUtils_h
#define sensei_XMLUtils_h

/// @file

#include "Error.h"

#include <mpi.h>
#include <string>
#include <vector>
#include <array>
#include <pugixml.hpp>

namespace sensei
{

/// a collection of functions for parsing XML.
namespace XMLUtils
{

/** check that an attribute of the passed in name exists return of 0 indicates
 * that it does. if not non-zero value is returned and an error message is sent
 * to stderr.
 */
SENSEI_EXPORT
int RequireAttribute(const pugi::xml_node &node, const char *attributeName);

/** check that a child element of the passed in name exists return of 0
 * indicates that it does. if not non-zero value is returned and an error
 * message is sent to stderr.
 */
SENSEI_EXPORT
int RequireChild(const pugi::xml_node &node, const char *childName);

/** Parallel collective read, parse, and distribute the XML file. Rank 0 does
 * the I/O and parse and will broadcast to the other ranks in the communicator.
 * return of 0 indicates success.
 */
SENSEI_EXPORT
int Parse(MPI_Comm comm, const std::string &filename, pugi::xml_document &doc);


/// @cond

/// helper for string to numeric type conversions
template <typename num_t> struct numeric_traits;

#define declare_numeric_traits(cpp_t, func) \
template <> struct numeric_traits<cpp_t>    \
{                                           \
    static                                  \
    cpp_t convert(const std::string &str)   \
    { return (cpp_t)func(str); }            \
};

declare_numeric_traits(int, std::stol)
declare_numeric_traits(unsigned int, std::stoul)
declare_numeric_traits(long, std::stol)
declare_numeric_traits(unsigned long, std::stoul)
declare_numeric_traits(long long, std::stol)
declare_numeric_traits(unsigned long long, std::stoul)
declare_numeric_traits(float, std::stof)
declare_numeric_traits(double, std::stod)

/// @endcond

/** parse text data in the node of unknown length, return in the vector. return
 * 0 if successful
 */
template <typename num_t>
int ParseNumeric(const pugi::xml_node &node, std::vector<num_t> &numData)
{
  std::string strData = node.text().as_string();
  std::string delims = " ,\t\n";

  std::size_t curr = strData.find_first_not_of(delims, 0);
  std::size_t next = std::string::npos;

  while (curr != std::string::npos)
    {
    next = strData.find_first_of(delims, curr + 1);
    std::string tmp = strData.substr(curr, next - curr);
    numData.push_back(numeric_traits<num_t>::convert(tmp));
    curr = strData.find_first_not_of(delims, next);
    }

  return 0;
}

/// parse text data in the node of specific length
template <typename num_t, unsigned long n>
int ParseNumeric(const pugi::xml_node &node, std::array<num_t,n> &numData)
{
  std::string strData = node.text().as_string();
  std::string delims = " ,\t";

  std::size_t curr = strData.find_first_not_of(delims, 0);
  std::size_t next = std::string::npos;

  unsigned long i = 0;
  while ((curr != std::string::npos) && (i < n))
    {
    next = strData.find_first_of(delims, curr + 1);
    std::string tmp = strData.substr(curr, next - curr);
    numData[i] = numeric_traits<num_t>::convert(tmp);
    curr = strData.find_first_not_of(delims, next);
    ++i;
    }

  if (i != n)
    {
    SENSEI_ERROR(<< "Missmatch in the nuber of values detected. "
      << node.name() << " requires " << n << " values")
    return -1;
    }

  return 0;
}

/// process a sequence of "name = value" pairs in a node's text.
int ParseNameValuePairs(const pugi::xml_node &node,
  std::vector<std::string> &names, std::vector<std::string> &values);

/** process a list in a node's text. List elements should be
 * space or comma delimited. Give the following XML
 *
 * ```XML
 * <some_elem>
 *   list1, list2, ... , listN
 * </some_elem>
 * ```
 *
 * extract list1 ... listN and return as a vector of strings. returns the number
 * of elements found or < 0 on error.
 */
int ParseList(pugi::xml_node node, std::vector<std::string> &listOut);

}
}
#endif
