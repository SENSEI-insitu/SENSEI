#ifndef OSCILLATORS_DATAADAPTOR_H
#define OSCILLATORS_DATAADAPTOR_H

#include <sensei/DataAdaptor.h>

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
  void Initialize(size_t nblocks);

  /// @brief Set the extents for local blocks.
  void SetBlockExtent(int gid,
    int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);

  /// @brief Set the extent of the dataset.
  void SetDataExtent(int ext[6]);

  /// Set data for a specific block.
  void SetBlockData(int gid, float* data);

  vtkDataObject* GetMesh(bool structure_only=false) override;
  bool AddArray(vtkDataObject* mesh, int association, const std::string& arrayname) override;
  unsigned int GetNumberOfArrays(int) override { return 1; }
  std::string GetArrayName(int, unsigned int index) override
  { return index==0? "data" : std::string(); }
  void ReleaseData() override;

protected:
  DataAdaptor();
  ~DataAdaptor();

  vtkDataObject* GetBlockMesh(int gid);

private:
  DataAdaptor(const DataAdaptor&); // not implemented.
  void operator=(const DataAdaptor&); // not implemented.

  struct DInternals;
  DInternals* Internals;
};

}
#endif
