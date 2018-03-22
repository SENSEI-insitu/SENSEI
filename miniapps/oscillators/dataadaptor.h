#ifndef OSCILLATORS_DATAADAPTOR_H
#define OSCILLATORS_DATAADAPTOR_H

#include <sensei/DataAdaptor.h>

class vtkDataArray;

namespace oscillators
{

class DataAdaptor : public sensei::DataAdaptor
{
public:
  static DataAdaptor* New();
  senseiTypeMacro(DataAdaptor, sensei::DataAdaptor);

  /// @brief Initialize the data adaptor.
  ///
  /// This initializes the data adaptor. This must be called once per simulation run.
  /// @param nblocks is the total number of blocks in the simulation run.
  void Initialize(size_t nblocks, const int *shape, int ghostLevels);

  /// @brief Set the extents for local blocks.
  void SetBlockExtent(int gid,
    int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);

  /// @brief Set the extent of the dataset.
  void SetDataExtent(int ext[6]);

  /// Set data for a specific block.
  void SetBlockData(int gid, float* data);

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

  int GetMeshHasGhostCells(const std::string &meshName, int &nLayers) override;

  int AddGhostCellsArray(vtkDataObject* mesh, const std::string &meshName) override;

  int ReleaseData() override;

protected:
  DataAdaptor();
  ~DataAdaptor();

  vtkDataObject* GetBlockMesh(int gid);
  vtkDataObject* GetUnstructuredMesh(int gid, bool structureOnly);
  vtkDataArray*  CreateGhostCellsArray(int cc) const;

private:
  DataAdaptor(const DataAdaptor&); // not implemented.
  void operator=(const DataAdaptor&); // not implemented.

  struct DInternals;
  DInternals* Internals;
};

}
#endif
