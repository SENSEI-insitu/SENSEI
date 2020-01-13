#ifndef sensei_VTKPosthocIO_h
#define sensei_VTKPosthocIO_h

#include "AnalysisAdaptor.h"
#include "DataRequirements.h"
#include "MeshMetadata.h"

#include <vtkSmartPointer.h>

#include <mpi.h>
#include <vector>
#include <string>


namespace sensei
{
class VTKPosthocIO;
using VTKPosthocIOPtr = vtkSmartPointer<VTKPosthocIO>;

/**  Writes simulation data to disk in a VTK based format. This can be useful
 * for generating preview datasets that allow configuration of Catalyst and/or
 * Libsim scripts, or for staging data on resources such as burst buffers. This
 * adaptor supports writing to VisIt(.visit) or ParaView(.pvd) compatible
 * format. One must provide a set of data requirments, consisting of a list of
 * meshes and the arrays to write from each mesh. File names are derived using
 * the output directory, the mesh name, and the mode.
 */
class VTKPosthocIO : public AnalysisAdaptor
{
public:
  /// Constructs a VTKPosthocIO instance.
  static VTKPosthocIO* New();

  senseiTypeMacro(VTKPosthocIO, AnalysisAdaptor);

  /// @name Run time configuration
  /// @{

  /// Sets the directory files will be written to.
  int SetOutputDir(const std::string &outputDir);

  /// File creation modes.
  enum {MODE_PARAVIEW=0, MODE_VISIT=1};

  /// Sets the file creation mode. MODE_PARAVIEW=0, MODE_VISIT=1
  int SetMode(int mode);

  /// Set the file creation mode by string Use either "paraview" or "visit".
  int SetMode(std::string mode);

  enum {WRITER_VTK_LEGACY=0, WRITER_VTK_XML=1};

  /** Sets the writer type to a VTK legacy writer(WRITER_VTK_LEGACY=0) or the
   * VTK XML writer (WRITER_VTK_XML=1).
   */
  int SetWriter(int writer);

  /** Sets the writer type to a VTK legacy writer("legacy") or the VTK XML
   * writer ("xml").
   */
  int SetWriter(std::string writer);

  /**  if set this overrrides the default of vtkGhostType for ParaView and
   * avtGhostZones for VisIt
   */
  void SetGhostArrayName(const std::string &name);

  /// Get the name of the ghost cells array.
  std::string GetGhostArrayName();

  /** Adds a set of ::DataRequirements, typically this will come from an XML
   * configuratiopn file. Data requirements tell the adaptor what to fetch from
   * the simulation and write to disk. If none are given then all available
   * data is fetched and written.
   */
  int SetDataRequirements(const DataRequirements &reqs);

  /** Add an indivudal data requirement. Data requirements tell the adaptor
   * what to fetch from the simulation and write to disk. If none are given
   * then all available data is fetched and written.

   * @param[in] meshName    the name of the mesh to fetch and write
   * @param[in] association the type of data array to fetch and write
   *                        vtkDataObject::POINT or vtkDataObject::CELL
   * @param[in] arrays      a list of arrays to fetch and write
   * @returns zero if successful.
   */
  int AddDataRequirement(const std::string &meshName,
    int association, const std::vector<std::string> &arrays);

  /// Controls how many calls to ::Execute do nothing between actual I/O
  int SetFrequency(unsigned int frequency);

  /// @}

  /// Invokes I/O
  bool Execute(DataAdaptor* data, DataAdaptor*&) override;

  /// Closes and flushes all files and creates metadata files for VisIt or ParaView.
  int Finalize() override;

protected:
  VTKPosthocIO();
  ~VTKPosthocIO();

  VTKPosthocIO(const VTKPosthocIO&) = delete;
  void operator=(const VTKPosthocIO&) = delete;

private:
#if !defined(SWIG)
  std::string OutputDir;
  DataRequirements Requirements;
  int Mode;
  int Writer;
  std::string GhostArrayName;

  template<typename T>
  using NameMap = std::map<std::string, T>;

  unsigned int Frequency;
  NameMap<std::vector<double>> Time;
  NameMap<std::vector<long>> TimeStep;
  NameMap<std::vector<MeshMetadataPtr>> Metadata;
  NameMap<std::string> BlockExt;
  NameMap<long> FileId;
  NameMap<int> HaveBlockInfo;
#endif
};

}
#endif
