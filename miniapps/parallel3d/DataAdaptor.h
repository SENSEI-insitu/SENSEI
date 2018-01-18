#ifndef PARALLEL3D_DATAADAPTOR_H
#define PARALLEL3D_DATAADAPTOR_H

#include "sensei/DataAdaptor.h"

#include <vtkSmartPointer.h>
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
  senseiTypeMacro(DataAdaptor, sensei::DataAdaptor);

  /// Initialize the data adaptor.
  void Initialize(int g_x, int g_y, int g_z, int l_x, int l_y, int l_z,
    uint64_t start_extents_x, uint64_t start_extents_y, uint64_t start_extents_z,
    int tot_blocks_x, int tot_blocks_y, int tot_blocks_z,
    int block_id_x, int block_id_y, int block_id_z);

  /// Set the pointers to simulation memory.
  void AddArray(const std::string& name, double* data);

  /// Clear all arrays.
  void ClearArrays();

  // SENSEI API
  int GetNumberOfMeshes(unsigned int &numMeshes) override;

  int GetMeshName(unsigned int id, std::string &meshName) override;

  int GetMesh(const std::string &meshName, bool structureOnly,
    vtkDataObject *&mesh) override;

  int AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName) override;

  int GetNumberOfArrays(const std::string &meshName, int association,
    unsigned int &numberOfArrays) override;

  int GetArrayName(const std::string &meshName, int association,
    unsigned int index, std::string &arrayName) override;

  int ReleaseData() override;

protected:
  DataAdaptor();
  ~DataAdaptor();

  using VariablesType = std::map<std::string, double*>;
  using ArraysType = std::map<std::string, vtkSmartPointer<vtkDoubleArray>>;
  using vtkImageDataPtr = vtkSmartPointer<vtkImageData>;
  using vtkDoubleArrayPtr = vtkSmartPointer<vtkDoubleArray>;

  VariablesType Variables;
  ArraysType Arrays;
  vtkImageDataPtr Mesh;
  int CellExtent[6];
  int WholeExtent[6];

private:
  DataAdaptor(const DataAdaptor&) = delete;
  void operator=(const DataAdaptor&) = delete;
};

}

#endif
