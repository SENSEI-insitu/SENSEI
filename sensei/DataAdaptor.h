#ifndef sensei_DataAdaptor_h
#define sensei_DataAdaptor_h

#include "senseiConfig.h"
#include "MeshMetadata.h"

#include <svtkObjectBase.h>

#include <vector>
#include <string>
#include <mpi.h>
#include <memory>

class svtkAbstractArray;
class svtkDataObject;
class svtkCompositeDataSet;

namespace sensei
{

/** Base class that defines the interface for fetching data from a simulation.
 * Simulation codes provide an implementation of this interface which is used
 * by data consumers to fetch simulation data for processing, movement, or I/O.
 */
class DataAdaptor : public svtkObjectBase
{
public:
  senseiBaseTypeMacro(DataAdaptor, svtkObjectBase);

  /// Prints the current state of the adaptor.
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /** Set the communicator used by the adaptor. The default communicator is a
   * duplicate of MPI_COMMM_WORLD, giving each adaptor a unique communication
   * space. Users wishing to override this should set the communicator before
   * doing anything else. Derived classes should use the communicator returned
   * by GetCommunicator.
   */
  virtual int SetCommunicator(MPI_Comm comm);

  /// Get the communicator used by the adaptor.
  MPI_Comm GetCommunicator() { return this->Comm; }

  /** Gets the number of meshes a simulation can provide.  The caller passes a
   * reference to an integer variable in the first argument upon return this
   * variable contains the number of meshes the simulation can provide. If
   * successfull the method returns 0, a non-zero return indicates an error
   * occurred.
   *
   *  @param[out] numMeshes an integer variable where number of meshes is stored
   *  @returns zero if successful, non zero if an error occurred
   */
  virtual int GetNumberOfMeshes(unsigned int &numMeshes) = 0;

  /** Get metadata of the i'th mesh.  The caller passes the integer id of the
   * mesh for which the metadata is desired, and a pointer to a MeshMetadata
   * instance  where the metadata is stored.
   *
   * @param[in] id index of the mesh to access
   * @param[out] metadata a pointer to instance where metadata is stored
   * @returns zero if successful, non zero if an error occurred
   */
  virtual int GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &metadata) = 0;

  /** Fetches the requested data object from the simulation.  This method will
   * return a svtkDataObject containing the simulation state. Implementors
   * should prefer svtkMultiBlockData over svtkDataSet types. However, both
   * approaches are supported.  Callers should typically pass \c structureOnly
   * to false. Caller may set \c stuctureOnly to true when data arrays without
   * mesh geometry and connectivity are sufficient for processing.  If \c
   * structureOnly is set to true, then implementors should not populate the
   * mesh geometry and connectivity information. This can result in large
   * savings in data transfer.
   *
   * @param[in] meshName the name of the mesh to access (see GetMeshMetadata)
   * @param[in] structureOnly When set to true the returned mesh
   *            may not have any geometry or topology information.
   * @param[out] mesh a reference to a pointer where a new VTK object is stored
   * @returns zero if successful, non zero if an error occurred
   */
  virtual int GetMesh(const std::string &meshName, bool structureOnly,
    svtkDataObject *&mesh) = 0;

  virtual int GetMesh(const std::string &meshName, bool structureOnly,
    svtkCompositeDataSet *&mesh);
  /** Adds ghost nodes on the specified mesh. Implementors shouls set the name
   * of the array to "vtkGhostType".
   *
   *  @param[in] mesh the VTK object returned from GetMesh
   *  @param[in] meshName the name of the mesh to access (see GetMeshMetadata)
   *  @returns zero if successful, non zero if an error occurred
   */
  virtual int AddGhostNodesArray(svtkDataObject* mesh, const std::string &meshName);

  /** Adds ghost cells on the specified mesh. Implementors should set the array name to
   * "vtkGhostType".
   *
   *  @param[in] mesh the svtkDataObject returned from GetMesh
   *  @param[in] meshName the name of the mesh to access (see ::GetMeshMetadata)
   *  @returns zero if successful, non zero if an error occurred
   */
  virtual int AddGhostCellsArray(svtkDataObject* mesh, const std::string &meshName);

  /** Fetches the named array from the simulation and adds it to the passed
   * mesh. Implementors should pass the data by zero copy when possible. See
   * svtkAOSDataArrayTemplate and svtkSOADataArrayTemplate for details of passing
   * data zero copy.
   *
   * @param[in] mesh the VTK object returned from GetMesh
   * @param[in] meshName the name of the mesh on which the array is stored
   * @param[in] association field association; one of
   *            svtkDataObject::FieldAssociations or svtkDataObject::AttributeTypes.
   * @param[in] arrayName name of the array
   * @returns zero if successful, non zero if an error occurred
   */
  virtual int AddArray(svtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName) = 0;

  /** Fetches multiple arrays from the simulation and adds them to the mesh.
   * Implementors typically do not have to override this method.
   *
   * @param[in] mesh the VTK object returned from GetMesh
   * @param[in] meshName the name of the mesh on which the array is stored
   * @param[in] association field association; one of
   *            svtkDataObject::FieldAssociations or svtkDataObject::AttributeTypes.
   * @param[in] arrayNames a vector of array names to add
   * @returns zero if successful, non zero if an error occurred
   */
  virtual int AddArrays(svtkDataObject* mesh, const std::string &meshName,
    int association, const std::vector<std::string> &arrayName);

  /** Release data allocated for the current timestep. Implementors typically
   * should not have to override this method. This method allows implementors to
   * free resources that were used in the conversion of the simulation data.
   * However, note that callers of ::GetMesh must delete the returned
   * svtkDataObject which typically obviates the need for the adaptor to hold
   * references to the returned data. Data consumers must call this method
   * when done processing the data to ensure that all resources are released.
   *
   * @returns zero if successful, non zero if an error occurred
   */
  virtual int ReleaseData() { return 0; }

  /// Get the current simulated time.
  virtual double GetDataTime();

  /** Set the current simulated time. The default implementation stores the
   * passed time. Implemetors typically call this method rather than
   * overriding the method.
   */
  virtual void SetDataTime(double time);

  /// Get the current time step.
  virtual long GetDataTimeStep();

  /** Set the current simulated time step. The default implementation stores the
   * passed time step. Implemetors typically call this method rather than
   * overriding the method.
   */
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
