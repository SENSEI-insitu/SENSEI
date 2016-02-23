#ifndef sensei_DataAdaptor_h
#define sensei_DataAdaptor_h

#include "vtkObjectBase.h"
#include "vtkSetGet.h" // needed for vtkTypeMacro.

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
  vtkTypeMacro(DataAdaptor, vtkObjectBase);
  void PrintSelf(ostream& os, vtkIndent indent);

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
  virtual vtkDataObject* GetMesh(bool structure_only=false) = 0;

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
  virtual bool AddArray(vtkDataObject* mesh, int association, const char* arrayname) = 0;

  /// @brief Return the number of field arrays available.
  ///
  /// This method will return the number of field arrays available. For data
  /// adaptors producing composite datasets, this is a union of arrays available
  /// on all parts of the composite dataset.
  ///
  /// @param association field association; one of
  /// vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @return the number of arrays.
  virtual unsigned int GetNumberOfArrays(int association) = 0;

  /// @brief Return the name for a field array.
  ///
  /// This method will return the name for a field array given its index.
  ///
  /// @param association field association; one of
  /// vtkDataObject::FieldAssociations or vtkDataObject::AttributeTypes.
  /// @param index index for the array. Must be less than value returned
  /// GetNumberOfArrays().
  /// @return name of the array.
  virtual const char* GetArrayName(int association, unsigned int index) = 0;

  /// @brief Release data allocated for the current timestep.
  ///
  /// Releases the data allocated for the current timestep. This is expected to
  /// be called after each time iteration.
  virtual void ReleaseData() = 0;

  /// @brief Provides access to meta-data about the current data.
  ///
  /// This method can provide all meta-data necessary, including global extents,
  /// fields available, etc.
  vtkInformation* GetInformation() { return this->Information; }

  /// @brief Convenience method to get the time information.
  double GetDataTime() { return this->GetDataTime(this->Information); }

  /// @brief Convenience method to get the time information.
  static double GetDataTime(vtkInformation*);

  /// @brief Convenience methods to set the time information.
  void SetDataTime(double time) { this->SetDataTime(this->Information, time); }

  /// @brief Convenience methods to set the time information.
  static void SetDataTime(vtkInformation*, double time);

  /// @brief Convenience method to get the time index information.
  int GetDataTimeStep() { return this->GetDataTimeStep(this->Information); }

  /// @brief Convenience method to get the time information.
  static int GetDataTimeStep(vtkInformation*);

  /// @brief Convenience methods to set the time information.
  void SetDataTimeStep(int index) { this->SetDataTimeStep(this->Information, index); }

  /// @brief Convenience methods to set the time information.
  static void SetDataTimeStep(vtkInformation*, int index);

  /// @brief convenience method to get full mesh will all arrays added to the
  /// mesh.
  vtkDataObject* GetCompleteMesh();

  /// @brief Key to store the timestep index.
  static vtkInformationIntegerKey* DATA_TIME_STEP_INDEX();
protected:
  DataAdaptor();
  ~DataAdaptor();

  vtkInformation* Information;
private:
  DataAdaptor(const DataAdaptor&); // Not implemented.
  void operator=(const DataAdaptor&); // Not implemented.
};

}
#endif
