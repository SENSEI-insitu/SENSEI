#ifndef ADIOSDataAdaptor_h
#define ADIOSDataAdaptor_h

#include "DataAdaptor.h"

#include <mpi.h>
#include <adios.h>
#include <adios_read.h>
#include <map>
#include <string>
#include <vtkSmartPointer.h>

namespace sensei
{

class ADIOSDataAdaptor : public DataAdaptor
{
public:
  static ADIOSDataAdaptor* New();
  senseiTypeMacro(ADIOSDataAdaptor, ADIOSDataAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  int Open(const std::string &method, const std::string& filename);
  int Open(ADIOS_READ_METHOD method, const std::string& filename);

  int Close();

  // Set the mesh type to static or dynamic. A static mesh does not
  // change in time and can be cached. A dynamic mesh must be re-read
  // each time step. By default the mesh is assumed assumed to be
  // dynamic.
  void EnableDynamicMesh(const std::string &meshName, int val);

  // advance the stream to the next available time step
  int Advance();

  /// @breif Gets the number of meshes a simulation can provide
  ///
  /// The caller passes a reference to an integer variable in the first
  /// argument upon return this variable contains the number of meshes the
  /// simulation can provide. If successfull the method returns 0, a non-zero
  /// return indicates an error occurred.
  ///
  /// param[out] numMeshes an integer variable where number of meshes is stored
  /// @returns zero if successful, non zero if an error occurred
  int GetNumberOfMeshes(unsigned int &numMeshes) override;

  /// @brief Get metadata of the i'th mesh
  ///
  /// The caller passes the integer id of the mesh for which the metadata is
  /// desired, and a pointer to a MeshMetadata instance  where the metadata is
  /// stored.
  ///
  /// @param[in] id index of the mesh to access
  /// @param[out] metadata a pointer to instance where metadata is stored
  /// @returns zero if successful, non zero if an error occurred
  int GetMeshMetadata(unsigned int id, MeshMetadataPtr &metadata) override;

  /// @brief Return the data object with appropriate structure.
  ///
  /// This method will return a data object of the appropriate type. The data
  /// object can be a vtkDataSet subclass or a vtkCompositeDataSet subclass.
  /// If \c structure_only is set to true, then the geometry and topology
  /// information will not be populated. For data adaptors that produce a
  /// vtkCompositeDataSet subclass, passing \c structure_only will still produce
  /// appropriate composite data hierarchy.
  ///
  /// @param[in] meshName the name of the mesh to access (see GetMeshName)
  /// @param[in] structure_only When set to true (default; false) the returned mesh
  ///            may not have any geometry or topology information.
  /// @param[out] a reference to a pointer where a new VTK object is stored
  /// @returns zero if successful, non zero if an error occurred
  int GetMesh(const std::string &meshName, bool structure_only,
    vtkDataObject *&mesh) override;

  /// @brief Adds ghost nodes on the specified mesh. The array name must be set
  ///        using the return value from GHOST_NODE_ARRAY_NAME().
  ///
  /// @param[in] mesh the VTK object returned from GetMesh
  /// @param[in] meshName the name of the mesh to access (see GetMeshName)
  /// @returns zero if successful, non zero if an error occurred
  int AddGhostNodesArray(vtkDataObject* mesh, const std::string &meshName) override;

  /// @brief Adds ghost cells on the specified mesh. The array name must be set
  ///        using the return value from GHOST_CELL_ARRAY_NAME().
  ///
  /// @param[in] mesh the VTK object returned from GetMesh
  /// @param[in] meshName the name of the mesh to access (see GetMeshName)
  /// @returns zero if successful, non zero if an error occurred
  int AddGhostCellsArray(vtkDataObject* mesh, const std::string &meshName) override;

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
  int AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName) override;

  /// @brief Release data allocated for the current timestep.
  ///
  /// Releases the data allocated for the current timestep. This is expected to
  /// be called after each time iteration.
  ///
  /// @returns zero if successful, non zero if an error occurred
  int ReleaseData() override;

  void ReleaseAttributeData(vtkDataObject* dobj);

protected:
  ADIOSDataAdaptor();
  ~ADIOSDataAdaptor();

   int GetObject(const std::string &meshName, vtkDataObject *&mesh);

  int GetGhostLayers(const std::string &meshName,
      int &nGhostCellLayers, int &nGhostNodeLayers);

private:
  struct InternalsType;
  InternalsType *Internals;

  // reads the current time step and time values from the stream
  // stores them in the base class information object
  int UpdateTimeStep();

  // return  a CSV list of the available array names.
  //std::string GetAvailableArrays(const std::string &meshName,
  //  int association);

  ADIOSDataAdaptor(const ADIOSDataAdaptor&); // Not implemented.
  void operator=(const ADIOSDataAdaptor&); // Not implemented.
};

}

#endif
