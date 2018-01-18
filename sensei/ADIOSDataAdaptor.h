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

  int Open(MPI_Comm comm, const std::string &method, const std::string& filename);
  int Open(MPI_Comm comm, ADIOS_READ_METHOD method, const std::string& filename);

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

  /// @ breif Get the name of the i'th mesh
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

  /// @brief Return the number of field arrays available.
  ///
  /// This method will return the number of field arrays available. For data
  /// adaptors producing composite datasets, this is a union of arrays available
  /// on all parts of the composite dataset.
  ///
  /// @param association field association; one of
  /// vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @return the number of arrays.
  int GetNumberOfArrays(const std::string &meshName, int association,
    unsigned int &numberOfArrays) override;

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
  int GetArrayName(const std::string &meshName, int association,
    unsigned int index, std::string &arrayName) override;

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
