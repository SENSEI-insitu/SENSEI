#ifndef PARALLEL3D_DATAADAPTOR_H
#define PARALLEL3D_DATAADAPTOR_H

#include "vtkInsituDataAdaptor.h"
#include "vtkSmartPointer.h"
#include <map>
#include <string>

class vtkDoubleArray;

namespace parallel3d
{
/// DataAdaptor is an adaptor for the parallel_3d simulation (miniapp).
/// Its purpose is to map the simulation datastructures to VTK
/// data model.
class DataAdaptor : public vtkInsituDataAdaptor
{
public:
  static DataAdaptor* New();
  vtkTypeMacro(DataAdaptor, vtkInsituDataAdaptor);

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

  /// Return the topology/geometry for the simulation grid.
  virtual vtkDataObject* GetMesh();

  /// Return an array.
  virtual vtkAbstractArray* GetArray(int association, const char* name);

  /// Return number of arrays.
  virtual unsigned int GetNumberOfArrays(int association);

  /// Returns an arrays name given the index.
  virtual const char* GetArrayName(int association, unsigned int index);

  /// Release all data.
  virtual void ReleaseData();

protected:
  DataAdaptor();
  virtual ~DataAdaptor();

  typedef std::map<std::string, double*> VariablesType;
  VariablesType Variables;

  typedef std::map<std::string, vtkSmartPointer<vtkDoubleArray> > ArraysType;
  ArraysType Arrays;

  int Extent[6];
private:
  DataAdaptor(const DataAdaptor&); // not implemented.
  void operator=(const DataAdaptor&); // not implemented.
};

}

#endif
