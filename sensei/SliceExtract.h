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

/// @class SliceExtract
/// Extract a slice defined by a point and a normal and writes it to disk
class SliceExtract : public AnalysisAdaptor
{
public:
  static SliceExtract *New();
  senseiTypeMacro(SliceExtract, AnalysisAdaptor);
  //void PrintSelf(ostream& os, vtkIndent indent) override;

  // set which operation will be used. Valid values are OP_ISO_SURFACE=0,
  // OP_PLANAR_SLICE=1
  enum {OP_ISO_SURFACE=0, OP_PLANAR_SLICE=1};
  int SetOperation(int op);

  // set which operation will be used. Valid values are "planar_slice",
  // "iso_surface"
  int SetOperation(std::string op);

  // set the values to compute iso-surfaces for. if these aren't set the
  // NumberOfIsoSurfaces field is used to generate equally spaced values on the
  // fly. This applies in OP_ISO_SURFACE
  void SetIsoValues(const std::string &mesh, const std::string &arrayName,
    int arrayCentering, const std::vector<double> &vals);

  // set the number of iso-surfaces to extract. Equally spaced iso-values are
  // determined at each update. This applies in OP_ISO_SURFACE when explicit
  // values have not been provided
  void SetNumberOfIsoValues(const std::string &mesh,
    const std::string &array, int centering, int numIsos);

  // set the point and nornal defining the slice plane
  // this applies when running in OP_PLANAR_SLICE
  int SetPoint(const std::array<double,3> &point);
  int SetNormal(const std::array<double,3> &normal);

  // set writer parameters
  int SetWriterOutputDir(const std::string &outputDir);
  int SetWriterMode(const std::string &mode);
  int SetWriterWriter(const std::string &writer);

  // data requirements tell the adaptor what mesh/arrays
  // to process
  int SetDataRequirements(const DataRequirements &reqs);

  int AddDataRequirement(const std::string &meshName,
    int association, const std::vector<std::string> &arrays);

  // SENSEI API
  void SetVerbose(int val) override;
  bool Execute(DataAdaptor* data) override;
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
