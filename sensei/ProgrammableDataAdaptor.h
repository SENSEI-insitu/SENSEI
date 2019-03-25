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

  /// Set the callable that will be invoked when GetMeshMetadata is called
  /// See GetMeshMetadata for details of what the callback must do.
  using GetMeshMetadataFunction =
    std::function<int(unsigned int, MeshMetadataPtr &)>;

  void SetGetMeshMetadataCallback(const GetMeshMetadataFunction &callback);

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
  GetMeshMetadataFunction GetMeshMetadataCallback;
  GetMeshFunction GetMeshCallback;
  AddArrayFunction AddArrayCallback;
  ReleaseDataFunction ReleaseDataCallback;

private:
  ProgrammableDataAdaptor(const ProgrammableDataAdaptor&) = delete;
  void operator=(const ProgrammableDataAdaptor&) = delete;
};

}
#endif
