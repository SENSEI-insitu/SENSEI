#ifndef ParticleDensity_h
#define ParticleDensity_h

#include "AnalysisAdaptor.h"
#include <mpi.h>
#include <vector>

class svtkDataObject;
class svtkDataArray;

namespace sensei
{
/** Computes number and mass density on a mesh from a set of particle positions
 * and masses. At each time step the results are stored on disk in PGM format.
 *
 * The input data is currently required to be on a svtkTable. Global mesh
 * bounds are obtained from the mesh metadata. One must additionally provide
 * during initialization the following: the mesh name; names of the particle
 * positions and masses; the resolution of the output image; and an output file
 * name where the substring %t% is replaced with the current simulation time.
 */
class SENSEI_EXPORT ParticleDensity : public AnalysisAdaptor
{
public:
  /// allocates a new instance
  static ParticleDensity* New();

  senseiTypeMacro(ParticleDensity, AnalysisAdaptor);

  // names the projection computed
  enum {PROJECT_XY=1, PROJECT_XZ=2, PROJECT_YZ=3 };

  /** set the name of the mesh and arrays to request from the simulation.
   * Sets the output image resolution and the name of a file to write them in.
   * the substring %t% is replaced with the current simulation time step. if
   * one of resx or resy is -1 it is calculated from the data's aspect ratio
   * so that image pixels are square.
   */
  void Initialize(const std::string &meshName,
    const std::string &xPosArray, const std::string &yPosArray,
    const std::string &zPosArray, const std::string &massArray,
    const std::string proj, long xres, long yres, const std::string &outDir,
    int returnData);

  /// compute the histogram for this time step
  bool Execute(DataAdaptor* data, DataAdaptor**) override;

  /// finalize the run
  int Finalize() override;

protected:
  ParticleDensity();
  ~ParticleDensity();

  ParticleDensity(const ParticleDensity&) = delete;
  void operator=(const ParticleDensity&) = delete;

  int Projection;
  long XRes;
  long YRes;
  std::string OutDir;
  unsigned long Iteration;

  std::string MeshName;
  std::string XPosArray;
  std::string YPosArray;
  std::string ZPosArray;
  std::string MassArray;
  int ReturnData;
};

}

#endif
