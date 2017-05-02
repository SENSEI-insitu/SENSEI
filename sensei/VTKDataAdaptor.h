#ifndef sensei_VTKDataAdaptor_h
#define sensei_VTKDataAdaptor_h

#include "DataAdaptor.h"
#include <vtkSmartPointer.h>

class vtkDataObject;

namespace sensei
{
/// @brief DataAdaptor for a vtkDataObject.
///
/// sensei::VTKDataAdaptor is a simple implementation of sensei::DataAdaptor
/// that provides the sensei::DataAdaptor on a vtkDataObject. To use this data
/// adaptor for codes that produce a vtkDataObject natively. Once simply passes
/// the vtkDataObject instance to this class and one then connect it to the
/// sensei::AnalysisAdators.
///
/// If the DataObject is a composite-dataset, the first non-null block on the
/// current rank is assumed to the representative block for answering all
/// queries.
class VTKDataAdaptor : public DataAdaptor
{
public:
  static VTKDataAdaptor* New();
  senseiTypeMacro(VTKDataAdaptor, DataAdaptor);

  /// @brief Set the data object for current timestep.
  ///
  /// This method must be called to the set the data object for
  /// the current timestep.
  void SetDataObject(vtkDataObject* dobj);

  vtkDataObject* GetMesh(bool structure_only=false) override;
  bool AddArray(vtkDataObject* mesh, int association, const std::string& arrayname) override;
  unsigned int GetNumberOfArrays(int association) override;
  std::string GetArrayName(int association, unsigned int index) override;
  void ReleaseData() override;
  vtkDataObject* GetCompleteMesh() override;

protected:
  VTKDataAdaptor();
  ~VTKDataAdaptor();

private:
  VTKDataAdaptor(const VTKDataAdaptor&); // Not implemented.
  void operator=(const VTKDataAdaptor&); // Not implemented.
  vtkSmartPointer<vtkDataObject> DataObject;
};

}

#endif
