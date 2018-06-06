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
  static DataAdaptor *New();
  senseiTypeMacro(DataAdaptor, sensei::DataAdaptor);

  /// Update the mesh geometry.
  ///
  /// The mini-app uses a regular Cartesian mesh and each MPI rank
  /// will have one piece of it.
  ///
  /// x_0, y_0, z_0          -- global simulation domain lower left corner
  /// dx, dy, dz             -- Cartesian mesh spacing
  /// g_nx, g_ny, g_nz       -- number of cells in the global simulation domain
  /// offs_x, offs_y, offs_z -- starting indices of the local domain
  /// l_nx, l_ny, l_nz       -- number of cells in the local domain
  void UpdateGeometry(double x_0, double y_0, double z_0,
    double dx, double dy, double dz, long g_nx, long g_ny, long g_nz,
    long offs_x, long offs_y, long offs_z, long l_nx, long l_ny, long l_nz);

  /// Update the simulation arrays.
  //
  /// The mini-app has 3 state arrays, pressure, temperature, and density
  /// the adaptor will zero-copy these.
  void UpdateArrays(double *pressure, double *temperature, double *density);

  // SENSEI API
  int GetNumberOfMeshes(unsigned int &numMeshes) override;

  int GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &md) override;

  int GetMesh(const std::string &meshName, bool structureOnly,
    vtkDataObject *&mesh) override;

  int AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName) override;

  int ReleaseData() override;

protected:
  DataAdaptor();
  ~DataAdaptor();

  int LocalExtent[6];  // local block's index space bounds
  int GlobalExtent[6]; // simulation's global index space bounds
  double Origin[3];    // lower left corner of the simulation domain
  double Spacing[3];   // mesh spacing
  double *Arrays[3];   // pointers to the simulation data

private:
  DataAdaptor(const DataAdaptor&) = delete;
  void operator=(const DataAdaptor&) = delete;
};

}

#endif
