#ifndef CatalystAnalysisAdaptor_h
#define CatalystAnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include "MeshMetadata.h"

#include <memory>
#include <vector>

class vtkCPDataDescription;
class vtkCPInputDataDescription;
class vtkCPPipeline;
class vtkDataObject;

namespace sensei
{

/// @brief Analysis adaptor for Catalyst-based analysis pipelines.
///
/// AnalysisAdaptor is a subclass of AnalysisAdaptor that is
/// that can be used as the superclass for all analysis that uses Catalyst.
class CatalystAnalysisAdaptor : public AnalysisAdaptor
{
public:
  static CatalystAnalysisAdaptor* New();
  senseiTypeMacro(CatalystAnalysisAdaptor, AnalysisAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /// @brief Add a vtkCPPipeline subclass to the global Catalyst Co-Processor.
  ///
  /// Adds a vtkCPPipeline subclass to the global Catalyst Co-Processor.
  virtual void AddPipeline(vtkCPPipeline* pipeline);

  /// Adds a pipeline initialized from a Catalyst python script
  virtual void AddPythonScriptPipeline(const std::string &fileName);

  bool Execute(DataAdaptor* data) override;

  int Finalize() override;

protected:
  CatalystAnalysisAdaptor();
  ~CatalystAnalysisAdaptor();

  void Initialize();

  int DescribeData(int timeStep, double time,
    const std::vector<MeshMetadataPtr> &metadata, vtkCPDataDescription *dataDesc);

  int SelectData(DataAdaptor *dataAdaptor,
    const std::vector<MeshMetadataPtr> &reqs, vtkCPDataDescription *dataDesc);

  int SetWholeExtent(vtkDataObject *dobj, vtkCPInputDataDescription *desc);

private:
  CatalystAnalysisAdaptor(const CatalystAnalysisAdaptor&); // Not implemented.
  void operator=(const CatalystAnalysisAdaptor&); // Not implemented.
};

}

#endif
