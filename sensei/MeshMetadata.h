#ifndef MeshMetadata_h
#define MeshMetadata_h

#include <vector>
#include <string>
#include <vtkDataObject.h> // for enums

namespace sensei
{

/// A container for capturing metadata describing a mesh, and optionally
/// cache a mesh object itself.
///
/// The followiing metadata is captured
///
/// StaticMesh -- a flag indicating if the mesh geometry evolves in time
///               defaults to false
///
/// StructureOnly -- a flag indicating if the current cached object was
///                  created with structure only. (see data adaptor for
///                  explanation of structure only)
///
/// NumberOfGhostCellLayers -- 0 if no ghost cell arrays are present or
///                            the mesh has no ghost cells. if 1 or larger
///                            ghost cells are flagged by cell data array
///                            named vtkGhostType
///
/// NumberOfGhostNodeLayers -- 0 if no ghost node arrays are present or
///                            the mesh has no ghost nodes. if 1 or larger
///                            ghost nodes are flagged by point data array
///                            named vtkGhostType
///
/// PointDataArrayNames -- vector of the available point data arrays
///
/// CellDataArrayNames -- vector of the available cell data arrays
///
struct MeshMetadata
{
  MeshMetadata() : MeshName(), StaticMesh(0), StructureOnly(0),
    NumberOfGhostCellLayers(0), NumberOfGhostNodeLayers(0),
    PointDataArrayNames(), CellDataArrayNames() {}

  MeshMetadata(const std::string &meshName) : MeshName(meshName), StaticMesh(0),
    StructureOnly(0), NumberOfGhostCellLayers(0), NumberOfGhostNodeLayers(0),
    PointDataArrayNames(), CellDataArrayNames() {}

  template <typename StringContainerType>
  MeshMetadata(const std::string &meshName, int staticMesh,
    bool structureOnly, int nGhostCellLayers, int nGhostNodeLayers,
    const StringContainerType &cellDataArrayNames,
    const StringContainerType &pointDataArrayNames);

  virtual ~MeshMetadata() {}

  // TODO -- implement BinaryStream and these methods
  /// serialize/deserialize metadata to a binary stream for communication
  /// and/or I/O
  using BinaryStream = char*;
  virtual int ToStream(BinaryStream &){ return -1; }
  virtual int FromStream(BinaryStream &){ return -1; }

  // helper returns the list of names given a valid association
  // no error checking is done so be sure that you pass a valid
  // association
  std::vector<std::string> &GetArrayNames(int association)
  {
    if (association == vtkDataObject::POINT)
      return this->PointDataArrayNames;
    return this->CellDataArrayNames;
  }

  // helper to set array names given a valid association
  // no error checking is done so be sure that you pass a valid
  // association
  template <typename StringContainerType>
  void SetArrayNames(int association, const StringContainerType &arrayNames);

  std::string MeshName;
  int StaticMesh;
  int StructureOnly;
  int NumberOfGhostCellLayers;
  int NumberOfGhostNodeLayers;
  std::vector<std::string> PointDataArrayNames;
  std::vector<std::string> CellDataArrayNames;
};

// --------------------------------------------------------------------------
template <typename StringContainerType>
MeshMetadata::MeshMetadata(const std::string &meshName, int staticMesh,
  bool structureOnly, int nGhostCellLayers, int nGhostNodeLayers,
  const StringContainerType &cellDataArrayNames,
  const StringContainerType &pointDataArrayNames) : MeshName(meshName),
  StaticMesh(staticMesh), StructureOnly(structureOnly),
  NumberOfGhostCellLayers(nGhostCellLayers),
  NumberOfGhostNodeLayers(nGhostNodeLayers),
  PointDataArrayNames(), CellDataArrayNames()
{
  this->CellDataArrayNames.assign(cellDataArrayNames.begin(),
    cellDataArrayNames.end());

  this->PointDataArrayNames.assign(pointDataArrayNames.begin(),
    pointDataArrayNames.end());
}

// --------------------------------------------------------------------------
template <typename StringContainerType>
void MeshMetadata::SetArrayNames(int association,
  const StringContainerType &arrayNames)
{
  if (association == vtkDataObject::POINT)
    this->PointDataArrayNames.assign(arrayNames.begin(), arrayNames.end());
  this->CellDataArrayNames.assign(arrayNames.begin(), arrayNames.end());
}

};

#endif
