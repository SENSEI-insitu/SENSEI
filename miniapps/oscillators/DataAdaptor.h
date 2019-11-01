#ifndef OSCILLATORS_DATAADAPTOR_H
#define OSCILLATORS_DATAADAPTOR_H

#include <sensei/DataAdaptor.h>

#include "Particles.h"

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
  void Initialize(size_t nblocks, size_t n_local_blocks,
    float *origin, float *spacing, int domain_shape_x, int domain_shape_y,
    int domain_shape_z, int *gid, int *from_x, int *from_y, int *from_z,
    int *to_x, int *to_y, int *to_z, int *shape, int ghostLevels);

  /// Set the extents for local blocks.
  void SetBlockExtent(int gid, int xmin, int xmax, int ymin,
    int ymax, int zmin, int zmax);

  /// Set the extent of the simulation domain
  void SetDomainExtent(int xmin, int xmax, int ymin, int ymax,
    int zmin, int zmax);

  /// Set data for a specific block.
  void SetBlockData(int gid, float* data);

  /// Set particles for a specific block
  void SetParticleData(int gid, const std::vector<Particle> &particles);

  // SENSEI API
  int GetNumberOfMeshes(unsigned int &numMeshes) override;

  int GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &md) override;

  int GetMesh(const std::string &meshName, bool structureOnly,
    vtkDataObject *&mesh) override;

  int AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName) override;

  int AddGhostCellsArray(vtkDataObject* mesh, const std::string &meshName) override;

  int ReleaseData() override;

protected:
  DataAdaptor();
  ~DataAdaptor();

  vtkDataArray*  CreateGhostCellsArray(int cc) const;

  vtkDataObject* GetParticlesBlock(int gid, bool structureOnly);

private:
  DataAdaptor(const DataAdaptor&); // not implemented.
  void operator=(const DataAdaptor&); // not implemented.

  struct InternalsType;
  InternalsType *Internals;
};

}
#endif
