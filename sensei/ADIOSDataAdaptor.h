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
  void EnableDynamicMesh(int val);

  // advance the stream to the next available time step
  int Advance();

  /// @brief Return the data object with appropriate structure.
  ///
  /// This method will return a data object of the appropriate type. The data
  /// object can be a vtkDataSet subclass or a vtkCompositeDataSet subclass.
  /// If \c structure_only is set to true, then the geometry and topology
  /// information will not be populated. For data adaptors that produce a
  /// vtkCompositeDataSet subclass, passing \c structure_only will still produce
  /// appropriate composite data hierarchy.
  ///
  /// @param structure_only When set to true (default; false) the returned mesh
  /// may not have any geometry or topology information.
  vtkDataObject* GetMesh(bool structure_only=false) override;

  /// @brief Adds the specified field array to the mesh.
  ///
  /// This method will add the requested array to the mesh, if available. If the
  /// array was already added to the mesh, this will not add it again. The mesh
  /// should not be expected to have geometry or topology information.
  ///
  /// @param association field association; one of
  /// vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @return true if array was added (or already added), false is request array
  /// is not available.
  bool AddArray(vtkDataObject* mesh, int association,
    const std::string& arrayname) override;

  /// @brief Return the number of field arrays available.
  ///
  /// This method will return the number of field arrays available. For data
  /// adaptors producing composite datasets, this is a union of arrays available
  /// on all parts of the composite dataset.
  ///
  /// @param association field association; one of
  /// vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @return the number of arrays.
  unsigned int GetNumberOfArrays(int association) override;

  /// @brief Return the name for a field array.
  ///
  /// This method will return the name for a field array given its index.
  ///
  /// @param association field association; one of
  /// vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @param index index for the array. Must be less than value returned
  /// GetNumberOfArrays().
  /// @return name of the array.
  std::string GetArrayName(int association, unsigned int index) override;

  /// @brief Release data allocated for the current timestep.
  ///
  /// Releases the data allocated for the current timestep. This is expected to
  /// be called after each time iteration.
  void ReleaseData() override;
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
  std::string GetAvailableArrays(int association);

  ADIOSDataAdaptor(const ADIOSDataAdaptor&); // Not implemented.
  void operator=(const ADIOSDataAdaptor&); // Not implemented.
};

}

#endif
