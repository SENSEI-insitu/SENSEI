#ifndef sensei_SVTKDataAdaptor_h
#define sensei_SVTKDataAdaptor_h

#include "DataAdaptor.h"
#include <svtkSmartPointer.h>

class svtkDataObject;

namespace sensei
{
/**  A sensei::DataAdaptor for a svtkDataObject.  To use this data adaptor for
 * codes that produce a svtkDataObject natively. Once simply passes the
 * svtkDataObject instance to this class and one then connect it to the
 * sensei::AnalysisAdators.
 *
 *  If the DataObject is a composite-dataset, the first non-null block on the
 *  current rank is assumed to the representative block for answering all
 *  queries.
 */
class SENSEI_EXPORT SVTKDataAdaptor : public DataAdaptor
{
public:
  static SVTKDataAdaptor *New();
  senseiTypeMacro(SVTKDataAdaptor, DataAdaptor);

  /**  Set the data object for current timestep.  This method must be called to
   * the set the data object for the current timestep.
   *
   *  @param[in] meshName name of the mesh represented in the object
   *  @param[in] mesh the mesh or nullptr if this rank has no data
   */
  void SetDataObject(const std::string &meshName, svtkDataObject *mesh);

  /** Get the data object for the current time step.  This method returns the
   * mesh associated with the passed in name. It returns zero if the mesh is
   * present and non-zero if it is not.
   *
   *  @param[in] meshName the name of the mesh to get
   *  @param[out] mesh a reference to store a pointer to the named mesh
   *  @returns zero if the named mesh is present, non zero if it was not
   */
  int GetDataObject(const std::string &meshName, svtkDataObject *&dobj);


  int GetNumberOfMeshes(unsigned int &numMeshes) override;
  int GetMeshMetadata(unsigned int id, MeshMetadataPtr &metadata) override;

  int GetMesh(const std::string &meshName, bool structure_only,
    svtkDataObject *&mesh) override;

  using sensei::DataAdaptor::GetMesh;

  int AddArray(svtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName) override;

  int ReleaseData() override;

protected:
  SVTKDataAdaptor();
  ~SVTKDataAdaptor();

private:
  SVTKDataAdaptor(const SVTKDataAdaptor&); // Not implemented.
  void operator=(const SVTKDataAdaptor&); // Not implemented.

  struct InternalsType;
  InternalsType *Internals;

};

}

#endif
