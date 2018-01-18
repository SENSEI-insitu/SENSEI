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

  /// Set the callable that will be invoked when GetNumberOfMeshes is called
  /// See GetNumberOfMeshes for details of what the callback must do.
  using GetNumberOfMeshesFunction = std::function<int(unsigned int&)>;

  void SetGetNumberOfMeshesCallback(const GetNumberOfMeshesFunction &callback);

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

  /// Set the callable that will be invoked when GetMeshName is called
  /// See GetMeshName for details of what the callback must do.
  using GetMeshNameFunction =
    std::function<int(unsigned int, std::string &)>;

  void SetGetMeshNameCallback(const GetMeshNameFunction &callback);

  /// @breif Get the name of the i'th mesh
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

  /// Set the callable that will be invoked when GetMesh is called
  /// See GetMesh for details of what the callback must do.
  using GetMeshFunction =
   std::function<int(const std::string &, bool, vtkDataObject *&)>;

  void SetGetMeshCallback(const GetMeshFunction &callback);

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
  int GetMesh(const std::string &meshName, bool structureOnly,
    vtkDataObject *&mesh) override;

  /// Set the callable that will be invoked when AddArray is called
  /// See AddArray for details of what the callback must do.
  using AddArrayFunction = std::function<int(vtkDataObject*,
    const std::string &, int, const std::string &)>;

  void SetAddArrayCallback(const AddArrayFunction &callback);

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

  /// Set the callable that will be invokde when GetNumberOfArrays is called.
  /// See GetNumberOfArrays for details of what the callback must do
  using GetNumberOfArraysFunction = std::function<int(const std::string &,
    int, unsigned int &)>;

  void SetGetNumberOfArraysCallback(const GetNumberOfArraysFunction &callback);

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

  /// Set the callable that will be invoked when GetArrayName is called.
  /// See GetArrayName for details of what the callback must do.
  using GetArrayNameFunction =
    std::function<int(const std::string &, int, unsigned int, std::string &)>;

  void SetGetArrayNameCallback(const GetArrayNameFunction &callback);

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

  ///Set the callable that will be invoked when ReleaseData is called.
  /// See ReleaseData for details about wwhat the callback should do.
  using ReleaseDataFunction = std::function<int()>;
  void SetReleaseDataCallback(const ReleaseDataFunction &callback);

  /// @brief Release data allocated for the current timestep.
  ///
  /// Releases the data allocated for the current timestep. This is expected to
  /// be called after each time iteration. The default implementation calls
  /// the RelaseData(meshName) override  onece for each mesh name.
  ///
  /// @returns zero if successful, non zero if an error occurred
  int ReleaseData() override;

protected:
  ProgrammableDataAdaptor();
  ~ProgrammableDataAdaptor();

  // callbacks
  GetNumberOfMeshesFunction GetNumberOfMeshesCallback;
  GetMeshNameFunction GetMeshNameCallback;
  GetMeshFunction GetMeshCallback;
  AddArrayFunction AddArrayCallback;
  GetNumberOfArraysFunction GetNumberOfArraysCallback;
  GetArrayNameFunction GetArrayNameCallback;
  ReleaseDataFunction ReleaseDataCallback;

private:
  ProgrammableDataAdaptor(const ProgrammableDataAdaptor&) = delete;
  void operator=(const ProgrammableDataAdaptor&) = delete;
};

}
#endif
