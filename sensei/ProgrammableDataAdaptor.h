#ifndef sensei_ProgrammableDataAdaptor_h
#define sensei_ProgrammableDataAdaptor_h

#include "senseiConfig.h"
#include "DataAdaptor.h"

#include <string>
#include <functional>

namespace sensei
{

/** Implements the sensei::DataAdaptor interface with user provided callables
 * ProgrammableDataAdaptor allows one to provide callbacks implementing
 * the data interface. This is an alternative to using polymorphism.
 * Callbacks can be any callable including functors and C function pointers
 */
class SENSEI_EXPORT ProgrammableDataAdaptor : public DataAdaptor
{
public:
  static ProgrammableDataAdaptor *New();
  senseiTypeMacro(ProgrammableDataAdaptor, DataAdaptor);

  using GetNumberOfMeshesFunction = std::function<int(unsigned int&)>;

  using GetMeshMetadataFunction =
    std::function<int(unsigned int, MeshMetadataPtr &)>;

  using GetMeshFunction =
   std::function<int(const std::string &, bool, svtkDataObject *&)>;

  using AddArrayFunction = std::function<int(svtkDataObject*,
    const std::string &, int, const std::string &)>;

  using ReleaseDataFunction = std::function<int()>;

  /** Set the callable that will be invoked when GetNumberOfMeshes is called
   *  See GetNumberOfMeshes for details of what the callback must do.
   */
  void SetGetNumberOfMeshesCallback(const GetNumberOfMeshesFunction &callback);

  /**  Set the callable that will be invoked when GetMeshMetadata is called
   *  See GetMeshMetadata for details of what the callback must do.
   */
  void SetGetMeshMetadataCallback(const GetMeshMetadataFunction &callback);

  /**  Set the callable that will be invoked when GetMesh is called
   *  See GetMesh for details of what the callback must do.
   */
  void SetGetMeshCallback(const GetMeshFunction &callback);

  /**  Set the callable that will be invoked when AddArray is called
   *  See AddArray for details of what the callback must do.
   */
  void SetAddArrayCallback(const AddArrayFunction &callback);

  /** Set the callable that will be invoked when ReleaseData is called.
   *  See ReleaseData for details about wwhat the callback should do.
   */
  void SetReleaseDataCallback(const ReleaseDataFunction &callback);

  int GetNumberOfMeshes(unsigned int &numMeshes) override;

  int GetMeshMetadata(unsigned int id, MeshMetadataPtr &metadata) override;

  int GetMesh(const std::string &meshName, bool structureOnly,
    svtkDataObject *&mesh) override;

  int AddArray(svtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName) override;
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
