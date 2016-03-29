#ifndef sensei_vtk_DataAdaptor_h
#define sensei_vtk_DataAdaptor_h

#include <sensei/DataAdaptor.h>
#include <vtkSmartPointer.h>
namespace sensei
{
namespace vtk
{

/// @brief DataAdaptor for a vtkDataObject.
///
/// sensei::vtk::DataAdaptor is a simple implementation of sensei::DataAdaptor
/// that provides the sensei::DataAdaptor on a vtkDataObject. To use this data
/// adaptor for codes that produce a vtkDataObject natively. Once simply passes
/// the vtkDataObject instance to this class and one then connect it to the
/// sensei::AnalysisAdators.
///
/// If the DataObject is a composite-dataset, the first non-null block on the
/// current rank is assumed to the representative block for answering all
/// queries.
class DataAdaptor : public sensei::DataAdaptor
{
public:
  static DataAdaptor* New();
  vtkTypeMacro(DataAdaptor, sensei::DataAdaptor);

  /// @brief Set the data object for current timestep.
  ///
  /// This method must be called to the set the data object for
  /// the current timestep.
  void SetDataObject(vtkDataObject* dobj);

  virtual vtkDataObject* GetMesh(bool structure_only=false);
  virtual bool AddArray(vtkDataObject* mesh, int association, const std::string& arrayname);
  virtual unsigned int GetNumberOfArrays(int association);
  virtual std::string GetArrayName(int association, unsigned int index);
  virtual void ReleaseData();
  virtual vtkDataObject* GetCompleteMesh();

protected:
  DataAdaptor();
  ~DataAdaptor();

private:
  DataAdaptor(const DataAdaptor&); // Not implemented.
  void operator=(const DataAdaptor&); // Not implemented.
  vtkSmartPointer<vtkDataObject> DataObject;
};

} // vtk
} // sensei
#endif
