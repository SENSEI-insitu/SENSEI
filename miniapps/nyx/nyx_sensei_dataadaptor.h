#ifndef _NYX_SENSEI_DATAADAPTOR_H
#define _NYX_SENSEI_DATAADAPTOR_H

#include <sensei/DataAdaptor.h>
#include <REAL.H>

class vtkImageData;

namespace nyx_sensei_bridge
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

#ifdef NYX_SENSEI_NO_COPY
  /// @brief Set the valid (non-ghost) extent for local blocks (index space).
  void SetValidBlockExtent(int gid,
    int xmin, int xmax, int ymin, int ymax, int zmin, int zmax);
#endif

  /// @brief Set the extent of the dataset (index space).
  void SetDataExtent(int ext[6]);

  /// @brief Set the extent of the dataset (physical space).
  void SetPhysicalExtents(double pext[6]);

  // @brief Compute origin and spacing from physical and index sapce data extents
  void ComputeSpacingAndOrigin();

  /// Set data for a specific block.
#ifdef NYX_SENSEI_NO_COPY
  void SetBlockData(int gid, const Real* data);
#else
  void SetBlockData(int gid, float* data);
#endif

  virtual vtkDataObject* GetMesh(bool structure_only=false);
  virtual bool AddArray(vtkDataObject* mesh, int association, const std::string& arrayname);
  virtual unsigned int GetNumberOfArrays(int) { return 1; }
  virtual std::string GetArrayName(int association, unsigned int index)
  { return index==0? "data" : std::string(); }
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
