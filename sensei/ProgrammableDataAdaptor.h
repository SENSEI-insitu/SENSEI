#ifndef sensei_ProgrammableDataAdaptor_h
#define sensei_ProgrammableDataAdaptor_h

#include "senseiConfig.h"
#include "DataAdaptor.h"

#include <string>
#include <functional>

namespace sensei
{

/// @class ProgrammableDataAdaptor
/// @brief ProgrammableDataAdaptor is class that implements the data interface with user provided callables
///
/// ProgrammableDataAdaptor allows one to provide callbacks implementing
/// the data interface. This is an alternative to using polymorphism.
/// Callbacks can be any callable including functors and C function pointers
///
/// One can provide implementations for following sensei::DataAdaptor
/// virtual methods by passing a callable(functor/function) to the corresponding
/// set method (Set_X_Callback, where _X_ is the method name).
///
///     vtkDataObject* GetMesh(bool structure_only);
///
///     bool AddArray(vtkDataObject* mesh, int association,
///       const std::string& arrayname);
///
///     unsigned int GetNumberOfArrays(int association);
///
///     std::string GetArrayName(int association, unsigned int index);
///
///     void ReleaseData();
///
class ProgrammableDataAdaptor : public DataAdaptor
{
public:
  static ProgrammableDataAdaptor *New();
  senseiTypeMacro(ProgrammableDataAdaptor, DataAdaptor);

  /// Set the callable that will be invoked when GetMesh is called
  /// See GetMesh for details of what the callback must do.
  using GetMeshCallbackType = std::function<vtkDataObject*(bool)>;
  void SetGetMeshCallback(const GetMeshCallbackType &callback);

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
  ///
  /// Calls to GetMesh will be forwarded to the user provided callback
  /// see SetGetMeshCallback
  vtkDataObject* GetMesh(bool structure_only) override;

  /// Set the callable that will be invoked when AddArray is called
  /// See AddArray for details of what the callback must do.
  using AddArrayCallbackType =
    std::function<bool(vtkDataObject*, int, const std::string&)>;

  void SetAddArrayCallback(const AddArrayCallbackType &callback);

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
  ///
  /// Calls to AddArray will be forwarded to the user provided callback
  /// see SetAddArrayCallback
  bool AddArray(vtkDataObject* mesh, int association,
    const std::string& arrayname) override;

  /// Set the callable that will be invokde when GetNumberOfArrays is called.
  /// See GetNumberOfArrays for details of what the callback must do
  using GetNumberOfArraysCallbackType = std::function<unsigned int(int)>;

  void SetGetNumberOfArraysCallback(const GetNumberOfArraysCallbackType &
    callback);

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

  /// Set the callable that will be invoked when GetArrayName is called.
  /// See GetArrayName for details of what the callback must do.
  using GetArrayNameCallbackType =
    std::function<std::string(int, unsigned int)>;

  void SetGetArrayNameCallback(const GetArrayNameCallbackType &callback);

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

  ///Set the callable that will be invoked when ReleaseData is called.
  /// See ReleaseData for details about wwhat the callback should do.
  using ReleaseDataCallbackType = std::function<void()>;
  void SetReleaseDataCallback(const ReleaseDataCallbackType &callback);

  /// @brief Release data allocated for the current timestep.
  ///
  /// Releases the data allocated for the current timestep. This is expected to
  /// be called after each time iteration.
  void ReleaseData() override;

protected:
  ProgrammableDataAdaptor();
  ~ProgrammableDataAdaptor();

  // callbacks
  GetMeshCallbackType GetMeshCallback;
  AddArrayCallbackType AddArrayCallback;
  GetNumberOfArraysCallbackType GetNumberOfArraysCallback;
  GetArrayNameCallbackType GetArrayNameCallback;
  ReleaseDataCallbackType ReleaseDataCallback;

private:
  ProgrammableDataAdaptor(const ProgrammableDataAdaptor&) = delete;
  void operator=(const ProgrammableDataAdaptor&) = delete;
};

}
#endif
