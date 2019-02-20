#ifndef CONDUIT_DATAADAPTOR_H
#define CONDUIT_DATAADAPTOR_H

#include <vector>
#include <vtkDataArray.h>
#include <conduit.hpp>
#include "DataAdaptor.h"

namespace sensei
{

class ConduitDataAdaptor : public sensei::DataAdaptor
{
public:
  static ConduitDataAdaptor* New();
  senseiTypeMacro(ConduitDataAdaptor, sensei::DataAdaptor);

  void SetNode(conduit::Node* node);
  void UpdateFields();

  /// @brief Gets the number of meshes a simulation can provide
  ///
  /// The caller passes a reference to an integer variable in the first
  /// argument upon return this variable contains the number of meshes the
  /// simulation can provide. If successfull the method returns 0, a non-zero
  /// return indicates an error occurred.
  ///
  /// param[out] numMeshes an integer variable where number of meshes is stored
  /// @returns zero if successful, non zero if an error occurred
  int GetNumberOfMeshes(unsigned int &numMeshes) override;

  /// @brief Get the name of the i'th mesh
  ///
  /// The caller passes the integer id of the mesh for which the name is
  /// desired, and a reference to string where the name is stored. See
  /// GetNumberOfMeshes. If successfull the method returns 0, a non-zero
  /// return indicates an error occurred.
  ///
  /// @param[in] id index of the mesh to access
  /// @param[out] meshName reference to a string where the name can be stored
  /// @returns zero if successful, non zero if an error occurred
  int GetMeshName(unsigned int id, std::string &meshName) override;

  /// @brief get a list of all mesh names
  ///
  /// A convenience method that returns a list of all mesh names
  ///
  /// @param[out] meshNames a vector where the list of names will be stored
  /// @returns zero if successful, non-zero if an error occurred
  //int GetMeshNames(std::vector<std::string> &meshNames) override;

  /// @brief Return the data object with appropriate structure.
  ///
  /// This method will return a data object of the appropriate type. The data
  /// object can be a vtkDataSet subclass or a vtkCompositeDataSet subclass.
  /// If \c structureOnly is set to true, then the geometry and topology
  /// information will not be populated. For data adaptors that produce a
  /// vtkCompositeDataSet subclass, passing \c structureOnly will still produce
  /// appropriate composite data hierarchy.
  ///
  /// @param[in] meshName the name of the mesh to access (see GetMeshName)
  /// @param[in] structureOnly When set to true (default; false) the returned mesh
  ///            may not have any geometry or topology information.
  /// @param[out] mesh a reference to a pointer where a new VTK object is stored
  /// @returns zero if successful, non zero if an error occurred
  int GetMesh(const std::string &meshName, bool structureOnly, vtkDataObject *&mesh) override;

  /// @brief convenience method to get full mesh will all arrays added to the
  /// mesh.
  ///
  /// Note that in some cases getting the complete mesh may cause significant
  /// memory pressure.
  ///
  /// @param[in] meshName the name of the mesh to access (see GetMeshName)
  /// @param[in] structureOnly When set to true (default; false) the returned mesh
  ///            may not have any geometry or topology information.
  /// @param[out] mesh a reference to a pointer where a new VTK object is stored
  /// @returns zero if successful, non zero if an error occurred
  //int GetCompleteMesh(const std::string &meshName, bool structureOnly, vtkDataObject *&mesh) override;

  /// @brief Returns whether a mesh has ghost nodes.
  ///
  /// @param[in] meshName the name of the mesh to access (see GetMeshName)
  /// @param[out] nLayers  the number of layers of ghost nodes present or 0.
  /// @returns zero if successful, non zero if an error occurred
  //int GetMeshHasGhostNodes(const std::string &meshName, int &nLayers) override;

  /// @brief Adds ghost nodes on the specified mesh. The array name must be set
  ///        to "vtkGhostType".
  ///
  /// @param[in] mesh the VTK object returned from GetMesh
  /// @param[in] meshName the name of the mesh to access (see GetMeshName)
  /// @returns zero if successful, non zero if an error occurred
  //int AddGhostNodesArray(vtkDataObject* mesh, const std::string &meshName) override;

  /// @brief Returns whether a mesh has ghost cells.
  ///
  /// @param[in] meshName the name of the mesh to access (see GetMeshName)
  /// @param[out] nLayers  the number of layers of ghost cells present or 0.
  /// @returns zero if successful, non zero if an error occurred
  //int GetMeshHasGhostCells(const std::string &meshName, int &nLayers) override;

  /// @brief Adds ghost cells on the specified mesh. The array name must be set
  ///        to "vtkGhostType".
  ///
  /// @param[in] mesh the VTK object returned from GetMesh
  /// @param[in] meshName the name of the mesh to access (see GetMeshName)
  /// @returns zero if successful, non zero if an error occurred
  //int AddGhostCellsArray(vtkDataObject* mesh, const std::string &meshName) override;

  /// @brief Adds the specified field array to the mesh.
  ///
  /// This method will add the requested array to the mesh, if available. If the
  /// array was already added to the mesh, this will not add it again. The mesh
  /// should not be expected to have geometry or topology information.
  ///
  /// @param[in] mesh the VTK object returned from GetMesh
  /// @param[in] meshName the name of the mesh on which the array is stored
  /// @param[in] association field association; one of
  ///            vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @param[in] arrayName name of the array
  /// @returns zero if successful, non zero if an error occurred
  int AddArray(vtkDataObject* mesh, const std::string &meshName, int association, const std::string &arrayName) override;

  /// @brief Return the number of field arrays available.
  ///
  /// This method will return the number of field arrays available. For data
  /// adaptors producing composite datasets, this is a union of arrays available
  /// on all parts of the composite dataset.
  ///
  /// @param association field association; one of
  /// vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @return the number of arrays.
  int GetNumberOfArrays(const std::string &meshName, int association, unsigned int &numberOfArrays) override;

  /// @brief Return the name for a field array.
  ///
  /// This method will return the name for a field array given its index.
  ///
  /// @param association field association; one of
  /// vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @param index index for the array. Must be less than value returned
  /// GetNumberOfArrays().
  ///
  /// @param[in] meshName name of mesh
  /// @param[in] association field association; one of
  ///            vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @param[in] index of the array
  /// @param[out] arrayName reference to a string where the name will be stored
  /// @returns zero if successful, non zero if an error occurred
  int GetArrayName(const std::string &meshName, int association, unsigned int index, std::string &arrayName) override;

  /// @brief Release data allocated for the current timestep.
  ///
  /// Releases the data allocated for the current timestep. This is expected to
  /// be called after each time iteration.
  ///
  /// @returns zero if successful, non zero if an error occurred
  int ReleaseData() override;

protected:
  ConduitDataAdaptor();
  ~ConduitDataAdaptor();

  typedef std::map<std::string, std::vector<std::string>> Fields;
  Fields FieldNames;
  int *GlobalBlockDistribution;

private:
  ConduitDataAdaptor(const ConduitDataAdaptor&)=delete; // not implemented.
  void operator=(const ConduitDataAdaptor&)=delete; // not implemented.

  conduit::Node* Node;
};

} // namespace sensei

#endif
