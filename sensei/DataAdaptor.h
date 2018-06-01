#ifndef sensei_DataAdaptor_h
#define sensei_DataAdaptor_h

#include "senseiConfig.h"
#include "vtkObjectBase.h"

#include <vector>
#include <string>
#include <mpi.h>

class vtkAbstractArray;
class vtkDataObject;
class vtkInformation;
class vtkInformationIntegerKey;

namespace sensei
{

/// @class DataAdaptor
/// @brief DataAdaptor is an abstract base class that defines the data interface.
///
/// DataAdaptor defines the data interface. Any simulation code that interfaces with
/// Sensei needs to provide an implementation for this interface. Analysis routines
/// (via AnalysisAdator) use the DataAdaptor implementation to access simulation data.
class DataAdaptor : public vtkObjectBase
{
public:
  senseiBaseTypeMacro(DataAdaptor, vtkObjectBase);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /// @brief Set the communicator used by the adaptor.
  /// The default communicator is a duplicate of MPI_COMMM_WORLD, giving
  /// each adaptor a unique communication space. Users wishing to override
  /// this should set the communicator before doing anything else. Derived
  /// classes should use the communicator returned by GetCommunicator.
  virtual int SetCommunicator(MPI_Comm comm);
  MPI_Comm GetCommunicator() { return this->Comm; }

  /// @brief Gets the number of meshes a simulation can provide
  ///
  /// The caller passes a reference to an integer variable in the first
  /// argument upon return this variable contains the number of meshes the
  /// simulation can provide. If successfull the method returns 0, a non-zero
  /// return indicates an error occurred.
  ///
  /// param[out] numMeshes an integer variable where number of meshes is stored
  /// @returns zero if successful, non zero if an error occurred
  virtual int GetNumberOfMeshes(unsigned int &numMeshes) = 0;

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
  virtual int GetMeshName(unsigned int id, std::string &meshName) = 0;

  /// @brief get a list of all mesh names
  ///
  /// A convenience method that returns a list of all mesh names
  ///
  /// @param[out] meshNames a vector where the list of names will be stored
  /// @returns zero if successful, non-zero if an error occurred
  virtual int GetMeshNames(std::vector<std::string> &meshNames);

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
  virtual int GetMesh(const std::string &meshName, bool structureOnly,
    vtkDataObject *&mesh) = 0;

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
  virtual int GetCompleteMesh(const std::string &meshName,
    bool structureOnly, vtkDataObject *&mesh);

  /// @brief Returns whether a mesh has ghost nodes.
  ///
  /// @param[in] meshName the name of the mesh to access (see GetMeshName)
  /// @param[out] nLayers  the number of layers of ghost nodes present or 0.
  /// @returns zero if successful, non zero if an error occurred
  virtual int GetMeshHasGhostNodes(const std::string &meshName, int &nLayers);

  /// @brief Adds ghost nodes on the specified mesh. The array name must be set
  ///        to "vtkGhostType".
  ///
  /// @param[in] mesh the VTK object returned from GetMesh
  /// @param[in] meshName the name of the mesh to access (see GetMeshName)
  /// @returns zero if successful, non zero if an error occurred
  virtual int AddGhostNodesArray(vtkDataObject* mesh, const std::string &meshName);

  /// @brief Returns whether a mesh has ghost cells.
  ///
  /// @param[in] meshName the name of the mesh to access (see GetMeshName)
  /// @param[out] nLayers  the number of layers of ghost cells present or 0.
  /// @returns zero if successful, non zero if an error occurred
  virtual int GetMeshHasGhostCells(const std::string &meshName, int &nLayers);

  /// @brief Adds ghost cells on the specified mesh. The array name must be set
  ///        to "vtkGhostType".
  ///
  /// @param[in] mesh the VTK object returned from GetMesh
  /// @param[in] meshName the name of the mesh to access (see GetMeshName)
  /// @returns zero if successful, non zero if an error occurred
  virtual int AddGhostCellsArray(vtkDataObject* mesh, const std::string &meshName);

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
  virtual int AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName) = 0;

  /// @brief Adds the vector of field arrays to the mesh.
  ///
  /// This method will add the requested array to the mesh, if available. If the
  /// array was already added to the mesh, this will not add it again. The mesh
  /// should not be expected to have geometry or topology information.
  ///
  /// @param[in] mesh the VTK object returned from GetMesh
  /// @param[in] meshName the name of the mesh on which the array is stored
  /// @param[in] association field association; one of
  ///            vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @param[in] arrayNames a vector of array names to add
  /// @returns zero if successful, non zero if an error occurred
  virtual int AddArrays(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::vector<std::string> &arrayName);

  /// @brief Adds all of specified field's arrays to the mesh.
  ///
  /// This method will add all the field's arrays to the mesh
  ///
  /// @param[in] mesh the VTK object returned from GetMesh
  /// @param[in] meshName the name of the mesh on which the array is stored
  /// @param[in] association field association; one of
  ///            vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @returns zero if successful, non zero if an error occurred
  virtual int AddArrays(vtkDataObject* mesh, const std::string &meshName,
    int association);

  /// @brief Return the number of field arrays available.
  ///
  /// This method will return the number of field arrays available. For data
  /// adaptors producing composite datasets, this is a union of arrays available
  /// on all parts of the composite dataset.
  ///
  /// @param association field association; one of
  /// vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @return the number of arrays.
  virtual int GetNumberOfArrays(const std::string &meshName, int association,
    unsigned int &numberOfArrays) = 0;

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
  virtual int GetArrayName(const std::string &meshName, int association,
    unsigned int index, std::string &arrayName) = 0;

  /// @brief get a list of the arrays on a given mesh
  ///
  /// A convenience method that returns a list of array names
  ///
  /// @param[in] meshName name of the mesh
  /// @param[in] association what type of arrays
  /// @param[out] arrayNames a vector where the list of names will be stored
  /// @returns zero if successful, non-zero if an error occurred
  int GetArrayNames(const std::string &meshName, int association,
    std::vector<std::string> &arrayNames);

  /// @brief Release data allocated for the current timestep.
  ///
  /// Releases the data allocated for the current timestep. This is expected to
  /// be called after each time iteration.
  ///
  /// @returns zero if successful, non zero if an error occurred
  virtual int ReleaseData() = 0;

  /// @brief Provides access to meta-data about the current data.
  ///
  /// This method can provide all meta-data necessary, including global extents,
  /// fields available, etc.
  vtkInformation* GetInformation();

  /// @brief Convenience method to get the time information.
  double GetDataTime();

  /// @brief Convenience method to get the time information.
  double GetDataTime(vtkInformation*);

  /// @brief Convenience methods to set the time information.
  void SetDataTime(double time);

  /// @brief Convenience methods to set the time information.
  void SetDataTime(vtkInformation*, double time);

  /// @brief Convenience method to get the time index information.
  int GetDataTimeStep();

  /// @brief Convenience method to get the time information.
  int GetDataTimeStep(vtkInformation*);

  /// @brief Convenience methods to set the time information.
  void SetDataTimeStep(int index);

  /// @brief Convenience methods to set the time information.
  void SetDataTimeStep(vtkInformation*, int index);

  /// @brief Key to store the timestep index.
  static vtkInformationIntegerKey* DATA_TIME_STEP_INDEX();

protected:
  DataAdaptor();
  ~DataAdaptor();

  DataAdaptor(const DataAdaptor&) = delete;
  void operator=(const DataAdaptor&) = delete;

  struct InternalsType;
  InternalsType *Internals;

  MPI_Comm Comm;
};

}
#endif
