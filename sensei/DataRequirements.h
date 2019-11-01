#ifndef DataRequirements_h
#define DataRequirements_h

#include <string>
#include <vector>
#include <map>
#include <set>
#include <pugixml.hpp>

namespace sensei
{

class DataAdaptor;
class MeshRequirementsIterator;
class ArrayRequirementsIterator;

/// This is a helper class that handles the common
/// task of specifying the set of meshes and arrays
/// rqeuired to perform a specific analysis. An analysis
/// would typically intialize the data requirement from
/// XML and then probe the requirements during execution
/// to get the minimal set of data required to complete
/// the analysis.
class DataRequirements
{
public:
  DataRequirements();
  ~DataRequirements();

  /// Returns true if the object is empty
  bool Empty() const { return this->MeshArrayMap.empty(); }

  /// initialize from XML. the XML should contain one or more
  /// mesh elements each with zero or more array groups
  ///
  /// <parent>
  ///   <mesh name="mesh_1" structure_only="1">
  ///     <cell_arrays>  array_1, ... array_n </cell_arrays>
  ///     <point_arrays>  array_1, ... array_n </point_arrays>
  ///   </mesh>
  ///   .
  ///   .
  ///   .
  ///   <mesh name="mesh_n" structure_only="0">
  ///     <cell_arrays>  array_1, ... array_n </cell_arrays>
  ///     <point_arrays>  array_1, ... array_n </point_arrays>
  ///   </mesh>
  /// </parent>
  ///
  /// @param[in] parent  XML node which contains mesh elements
  /// @returns the number of mesh elements processed.
  int Initialize(pugi::xml_node parent);

  /// @breif Get a description of what data is available
  ///
  /// This is a convenience method that fills in
  /// all of data that the data adaptor makes avaialable.
  ///
  /// @param[in] adaptor a DataAdaptor instance
  /// @param[in] structureOnly true if mesh geometry
  //             and topology are not needed
  /// @returns zero if successful, non zero if an error occurred
  int Initialize(DataAdaptor *adaptor, bool structureOnly);

  /// Adds a mesh
  /// The requirement consists of a mesh name and weather it is structure only
  /// @param[in] meshName name of the mesh
  /// @param[in] structureOnly flag indicating if mesh geometry is needed
  /// @returns zero if successful
  int AddRequirement(const std::string &meshName, bool structureOnly);

  /// Adds a set of arrays on a specific mesh
  /// The requirement consists of a mesh name and a list of arrays
  /// @param[in] meshName name of the mesh
  /// @param[in] association type of arrays
  /// @param[in] arrays a list of the required arrays
  /// @returns zero if successful
  int AddRequirement(const std::string &meshName, int association,
    const std::vector<std::string> &arrays);

  int AddRequirement(const std::string &meshName, int association,
    const std::string &array);

  /// Get the list of meshes
  /// @param[out] meshes a vector where mesh names will be stored
  /// @returns zero if successful
  int GetRequiredMeshes(std::vector<std::string> &meshes) const;

  unsigned int GetNumberOfRequiredMeshes() const;
  int GetRequiredMesh(unsigned int id, std::string &mesh) const;

  /// For the named mesh, gets the list of required arrays
  /// @param[in] meshName the name of the mesh
  /// @param[in] association vtkDataObject::POINT, vtkDataObject::CELL, etc
  /// @param[out] arrays a vector where the arrays will be stored
  /// @returns zero if successful
  int GetRequiredArrays(const std::string &meshName, int association,
    std::vector<std::string> &arrays) const;

  /// For the named mesh, and association, gets the number of required arrays
  /// @param[in] meshName the name of the mesh
  /// @param[in] association vtkDataObject::POINT, vtkDataObject::CELL, etc
  /// @param[out] number of arrays
  /// @returns zero if successful
  int GetNumberOfRequiredArrays(const std::string &meshName,
    int association, unsigned int &nArrays) const;

  /// Clear the contents of the container
  void Clear();

  /// Get an iterator for the named mesh
  MeshRequirementsIterator GetMeshRequirementsIterator() const;

  /// Get an iterator for the associated arrays
  ArrayRequirementsIterator GetArrayRequirementsIterator(
    const std::string &meshName) const;

public:
  using AssocArrayMapType = std::map<int, std::vector<std::string>>;
  using MeshArrayMapType = std::map<std::string, AssocArrayMapType>;
  using MeshNamesType = std::map<std::string, bool>;

private:
  friend class ArrayRequirementsIterator;
  friend class MeshRequirementsIterator;

  MeshNamesType MeshNames;
  MeshArrayMapType MeshArrayMap;
};

// iterate over the meshes
class MeshRequirementsIterator
{
public:
  MeshRequirementsIterator() : Valid(false) {}

  MeshRequirementsIterator(const DataRequirements::MeshNamesType &meshNames)
    : Valid(true), It(meshNames.cbegin()), End(meshNames.cend()) {}

  /// advance to the next requirement
  MeshRequirementsIterator &operator++(){ ++this->It; return *this; }

  /// Test if the iterator is finished
  operator bool() const { return Valid && (this->It != this->End); }

  /// returns the mesh name
  const std::string &MeshName(){ return this->It->first; }

  bool StructureOnly(){ return this->It->second; }

private:
  bool Valid;
  DataRequirements::MeshNamesType::const_iterator It;
  DataRequirements::MeshNamesType::const_iterator End;
};


/// Iterate over the mesh's arrays. One can iterate either over
/// associations and access all of an associations arrays or
/// one by one over the arrays.
class ArrayRequirementsIterator
{
public:
  enum {MODE_ARRAY, MODE_ASSOCIATION};

  ArrayRequirementsIterator() : Valid(false), Mode(MODE_ARRAY) {}

  ArrayRequirementsIterator(const DataRequirements::AssocArrayMapType &aa)
    : Valid(true), Mode(MODE_ARRAY), It(aa.cbegin()), End(aa.cend())
  { this->UpdateArrayIterator(); }

  /// Set the mode of operator++
  void SetMode(int mode){ this->Mode = mode; }

  /// Test if the iterator is finished
  operator bool() const
  {
    if (Valid)
      {
      if (this->Mode == MODE_ASSOCIATION)
        {
        if (this->It == this->End)
          return false;
        else
          return true;
        }
      else if (this->Mode == MODE_ARRAY)
        {
        if ((this->It == this->End) && (this->AIt == this->AEnd))
          return false;
        else
          return true;
        }
      }
    return false;
  }

  /// returns the type of requirement (point,cell, etc)
  int Association(){ return this->It->first; }

  /// Get the next group of requied arrays
  const std::vector<std::string> &Arrays(){ return this->It->second; }

  /// Get the next requied array
  const std::string &Array(){ return *this->AIt; }

  /// Advance the iterator
  ArrayRequirementsIterator &operator++()
  {
    if (this->Mode == MODE_ARRAY)
      return this->NextArray();
    return this->NextAssociation();
  }

  /// move to the next group od arrays
  ArrayRequirementsIterator &NextAssociation()
  {
    if (this->It != this->End)
      {
      ++this->It;
      this->UpdateArrayIterator();
      // the current association is empty
      if (this->AIt == this->AEnd)
        return this->NextAssociation();
      }
    return *this;
  }

  /// advance to the next array
  ArrayRequirementsIterator &NextArray()
  {
    ++this->AIt;
    // at the end of this association's arrays
    if (this->AIt == this->AEnd)
      this->NextAssociation();
    return *this;
  }

private:

  // move array iterator to the next group of arrays
  void UpdateArrayIterator()
  {
    if (this->It != this->End)
      {
      this->AIt = this->It->second.cbegin();
      this->AEnd = this->It->second.cend();
      // the current association is empty
      if (this->AIt == this->AEnd)
        this->NextAssociation();
      }
  }

  bool Valid;
  int Mode;
  DataRequirements::AssocArrayMapType::const_iterator It;
  DataRequirements::AssocArrayMapType::const_iterator End;
  std::vector<std::string>::const_iterator AIt;
  std::vector<std::string>::const_iterator AEnd;
};

}

#endif
