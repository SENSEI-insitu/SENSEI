#ifndef PARALLEL3D_DATAADAPTOR_H
#define PARALLEL3D_DATAADAPTOR_H

#include <sensei/DataAdaptor.h>
#include "vtkSmartPointer.h"
#include <map>
#include <string>
#include <cstdint>

class vtkDoubleArray;
class vtkImageData;

namespace parallel3d
{
/// DataAdaptor is an adaptor for the parallel_3d simulation (miniapp).
/// Its purpose is to map the simulation datastructures to VTK
/// data model.
class DataAdaptor : public sensei::DataAdaptor
{
public:
  static DataAdaptor* New();
  vtkTypeMacro(DataAdaptor, sensei::DataAdaptor);

  /// Initialize the data adaptor.
  void Initialize(
    int g_x, int g_y, int g_z,
    int l_x, int l_y, int l_z,
    uint64_t start_extents_x, uint64_t start_extents_y, uint64_t start_extents_z,
    int tot_blocks_x, int tot_blocks_y, int tot_blocks_z,
    int block_id_x, int block_id_y, int block_id_z);

  /// Set the pointers to simulation memory.
  void AddArray(const std::string& name, double* data);

  /// Clear all arrays.
  void ClearArrays();

  virtual vtkDataObject* GetMesh(bool structure_only=false);
  virtual bool AddArray(vtkDataObject* mesh, int association, const std::string& arrayname);
  virtual unsigned int GetNumberOfArrays(int association);
  virtual std::string GetArrayName(int association, unsigned int index);
  virtual void ReleaseData();

protected:
  DataAdaptor();
  virtual ~DataAdaptor();

  typedef std::map<std::string, double*> VariablesType;
  VariablesType Variables;

  typedef std::map<std::string, vtkSmartPointer<vtkDoubleArray> > ArraysType;
  ArraysType Arrays;

  vtkSmartPointer<vtkImageData> Mesh;
  int CellExtent[6];
  int WholeExtent[6];
private:
  DataAdaptor(const DataAdaptor&); // not implemented.
  void operator=(const DataAdaptor&); // not implemented.
};

}

#endif
