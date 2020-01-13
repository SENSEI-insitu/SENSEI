#ifndef sensei_SliceExtract_h
#define sensei_SliceExtract_h

#include "AnalysisAdaptor.h"

#include <vector>
#include <array>
#include <string>


class vtkCompositeDataSet;
namespace pugi { class xml_node; }
namespace sensei { class DataRequirements; }

namespace sensei
{

/// Extract a slice defined by a point and a normal and writes it to disk
class SliceExtract : public AnalysisAdaptor
{
public:
  /// Create an instance of SliceExtract
  static SliceExtract *New();

  senseiTypeMacro(SliceExtract, AnalysisAdaptor);

  /// @name Run time configuration
  /// @{

  /// Enable the use of an optimized partitioner
  void EnablePartitioner(int val);

  enum {OP_ISO_SURFACE=0, OP_PLANAR_SLICE=1};

  /** Set which operation will be used. Valid values are OP_ISO_SURFACE=0,
   * OP_PLANAR_SLICE=1
   */
  int SetOperation(int op);

  /** set which operation will be used. Valid values are "planar_slice",
   * "iso_surface"
   */
  int SetOperation(std::string op);

  /** set the values to compute iso-surfaces for. if these aren't set the
   * NumberOfIsoSurfaces field is used to generate equally spaced values on the
   * fly. This applies in OP_ISO_SURFACE
   */
  void SetIsoValues(const std::string &mesh, const std::string &arrayName,
    int arrayCentering, const std::vector<double> &vals);

  /** set the number of iso-surfaces to extract. Equally spaced iso-values are
   * determined at each update. This applies in OP_ISO_SURFACE when explicit
   * values have not been provided
   */
  void SetNumberOfIsoValues(const std::string &mesh,
    const std::string &array, int centering, int numIsos);

  /** set the point and nornal defining the slice plane this applies when
   * running in OP_PLANAR_SLICE
   */
  int SetPoint(const std::array<double,3> &point);

  /** set the point and nornal defining the slice plane this applies when
   * running in OP_PLANAR_SLICE
   */
  int SetNormal(const std::array<double,3> &normal);

  /// Sets the directory files will be written to.
  int SetWriterOutputDir(const std::string &outputDir);

  /// Set the file creation mode by string Use either "paraview" or "visit".
  int SetWriterMode(const std::string &mode);

  /** Sets the writer type to a VTK legacy writer("legacy") or the VTK XML
   * writer ("xml").
   */
  int SetWriterWriter(const std::string &writer);

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

  /// Sets the verbosity level of the internally used adaptors.
  void SetVerbose(int val) override;

  /// @}

  /// Compute a slice and write it to disk.
  bool Execute(DataAdaptor* data, DataAdaptor*&) override;

  /// Flush and close all open files.
  int Finalize() override;

private:

    bool ExecuteSlice(DataAdaptor* dataAdaptor);
    bool ExecuteIsoSurface(DataAdaptor* dataAdaptor);

    int Slice(vtkCompositeDataSet *input, const std::array<double,3> &point,
      const std::array<double,3> &normal, vtkCompositeDataSet *&output);

    int IsoSurface(vtkCompositeDataSet *input,
      const std::string &arrayName, int arrayCen,
      const std::vector<double> &vals, vtkCompositeDataSet *&output);

    int WriteExtract(long timeStep, double time, const std::string &mesh,
      vtkCompositeDataSet *input);

protected:
  SliceExtract();
  ~SliceExtract();

  SliceExtract(const SliceExtract&) = delete;
  void operator=(const SliceExtract&) = delete;

  struct InternalsType;
  InternalsType *Internals;
};

}
#endif
