#ifndef sensei_VTKAmrWriter_h
#define sensei_VTKAmrWriter_h

#include "AnalysisAdaptor.h"
#include "DataRequirements.h"

#include <mpi.h>
#include <vector>
#include <string>


namespace sensei
{
/** Writes AMR data to disk in VTK format. This can be useful for generating
 * preview datasets that allow configuration of Catalyst and/or Libsim scripts,
 * or for staging data on resources such as burst buffers. This adaptor
 * supports writing to a VTK(PXML), VisIt(.visit) or ParaView(.pvd) compatible
 * format. One must provide a set of data requirments, consisting of a list of
 * meshes and the arrays to write from each mesh. File names are derived using
 * the output directory, the mesh name, and the mode.
 */
class VTKAmrWriter : public AnalysisAdaptor
{
public:
  /// constructs a new VTKAmrWriter
  static VTKAmrWriter* New();

  senseiTypeMacro(VTKAmrWriter, AnalysisAdaptor);

  /// @name Run time configuration
  /// @{

  /// Sets the directory files will be written to.
  int SetOutputDir(const std::string &outputDir);

  enum {MODE_PARAVIEW=0, MODE_VISIT=1};

  /// Sets the file creation mode. MODE_PARAVIEW=0, MODE_VISIT=1
  int SetMode(int mode);

  /// Set the file creation mode by string Use either "paraview" or "visit".
  int SetMode(std::string mode);

  /** Adds a set of sensei::DataRequirements, typically this will come from an XML
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

  /// Must be called before ::Execute to configure for the run.
  int Initialize();
  /// @}

  /// Invokes the write.
  bool Execute(DataAdaptor* data, DataAdaptor*&) override;

  /** Finalizes output, flushes buffers, creates metadata files, and closes all
   * streams.
   */
  int Finalize() override;

protected:
  VTKAmrWriter();
  ~VTKAmrWriter();

  VTKAmrWriter(const VTKAmrWriter&) = delete;
  void operator=(const VTKAmrWriter&) = delete;

private:
#if !defined(SWIG)
  std::string OutputDir;
  DataRequirements Requirements;
  int Mode;

  template<typename T>
  using NameMap = std::map<std::string, T>;

  NameMap<std::vector<double>> Time;
  NameMap<std::vector<long>> TimeStep;
  NameMap<long> FileId;
  NameMap<int> HaveBlockInfo;
#endif
};

}
#endif
