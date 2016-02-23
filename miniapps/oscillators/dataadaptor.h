#ifndef OSCILLATORS_DATAADAPTOR_H
#define OSCILLATORS_DATAADAPTOR_H

#include <sensei/DataAdaptor.h>

namespace oscillators
{
class DataAdaptor : public sensei::DataAdaptor
{
public:
  static DataAdaptor* New();
  vtkTypeMacro(DataAdaptor, sensei::DataAdaptor);

  /// @brief Initialize the data adaptor.
  ///
  /// This initializes the data adaptor. This must be called once per simulation run.
  /// @param nblocks is the total number of blocks in the simulation run.
  void Initialize(size_t nblocks);

  /// @brief Set the extents for local blocks.
  void SetBlockExtent(int gid,
    int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);

  /// Set data for a specific block.
  void SetBlockData(int gid, float* data);

  virtual vtkDataObject* GetMesh(bool structure_only=false);
  virtual bool AddArray(vtkDataObject* mesh, int association, const char* arrayname);
  virtual unsigned int GetNumberOfArrays(int association) { return 1; }
  virtual const char* GetArrayName(int association, unsigned int index)
    { return index==0? "data" : NULL; }
  virtual void ReleaseData();

protected:
  DataAdaptor();
  virtual ~DataAdaptor();

  vtkDataObject* GetBlockMesh(int gid);

private:
  DataAdaptor(const DataAdaptor&); // not implemented.
  void operator=(const DataAdaptor&); // not implemented.

  class DInternals;
  DInternals* Internals;
};

}
#endif
