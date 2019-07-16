#ifndef sensei_DataAdaptor_h
#define sensei_DataAdaptor_h

#include "senseiConfig.h"
#include "MeshMetadata.h"

#include <vtkObjectBase.h>

#include <vector>
#include <string>
#include <mpi.h>
#include <memory>

class vtkAbstractArray;
class vtkDataObject;
class vtkCompositeDataSet;

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

  /// @brief Get metadata of the i'th mesh
  ///
  /// The caller passes the integer id of the mesh for which the metadata is
  /// desired, and a pointer to a MeshMetadata instance  where the metadata is
  /// stored.
  ///
  /// @param[in] id index of the mesh to access
  /// @param[out] metadata a pointer to instance where metadata is stored
  /// @returns zero if successful, non zero if an error occurred
  virtual int GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &metadata) = 0;

  /// @brief Return the data object with appropriate structure.
  ///
  /// This method will return a data object of the appropriate type. The data
  /// object can be a vtkDataSet subclass or a vtkCompositeDataSet subclass.
  /// If \c structureOnly is set to true, then the geometry and topology
  /// information will not be populated. For data adaptors that produce a
  /// vtkCompositeDataSet subclass, passing \c structureOnly will still produce
  /// appropriate composite data hierarchy. The caller takes ownership of the
  /// returned mesh object, and will call Delete when it is no longer needed.
  ///
  /// @param[in] meshName the name of the mesh to access (see GetMeshMetadata)
  /// @param[in] structureOnly When set to true (default; false) the returned mesh
  ///            may not have any geometry or topology information.
  /// @param[out] mesh a reference to a pointer where a new VTK object is stored
  /// @returns zero if successful, non zero if an error occurred
  virtual int GetMesh(const std::string &meshName, bool structureOnly,
    vtkDataObject *&mesh) = 0;

  /// @brief Return the data as a composite (multi-block) object
  ///
  /// This method simplifies data access enuring that one can always iterate
  /// through the blocks of data. If the simulation provides a legacy VTK object
  /// per-rank this method efficiently converts it to a composite (multi-block)
  /// object.
  virtual int GetMesh(const std::string &meshName, bool structureOnly,
    vtkCompositeDataSet *&mesh);

  /// @brief Adds ghost nodes on the specified mesh. The array name must be set
  ///        to "vtkGhostType".
  ///
  /// @param[in] mesh the VTK object returned from GetMesh
  /// @param[in] meshName the name of the mesh to access (see GetMeshMetadata)
  /// @returns zero if successful, non zero if an error occurred
  virtual int AddGhostNodesArray(vtkDataObject* mesh, const std::string &meshName);

  /// @brief Adds ghost cells on the specified mesh. The array name must be set
  ///        to "vtkGhostType".
  ///
  /// @param[in] mesh the VTK object returned from GetMesh
  /// @param[in] meshName the name of the mesh to access (see GetMeshMetadata)
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

  /// @brief Release data allocated for the current timestep.
  ///
  /// Releases the data allocated for the current timestep. This is expected to
  /// be called after each time iteration.
  ///
  /// @returns zero if successful, non zero if an error occurred
  virtual int ReleaseData() = 0;

  /// @brief Set/get the current simulated time.
  virtual double GetDataTime();
  virtual void SetDataTime(double time);

  /// @brief Set/get the current time step
  virtual long GetDataTimeStep();
  virtual void SetDataTimeStep(long index);

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
