#ifndef sensei_CatalystAnalysisAdaptor_h
#define sensei_CatalystAnalysisAdaptor_h

#include <AnalysisAdaptor.h>

class vtkCPInputDataDescription;
class vtkCPPipeline;

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

  /// @brief Fill \c desc with meta data from \c DataAdaptor.
  ///
  /// Called before the RequestDataDescription step to fill \c desc with
  /// information about fields (and any other meta-data) available in the
  /// \c DataAdaptor.
  /// @return true on success, false to abort execution.
  virtual bool FillDataDescriptionWithMetaData(
    DataAdaptor* dataAdaptor, vtkCPInputDataDescription* desc);

  /// @brief Fill \c desc with data from \c DataAdaptor.
  ///
  /// Called before the CoProcess() step to fill \c desc with
  /// simulation data.
  /// @return true on success, false to abort execution.
  virtual bool FillDataDescriptionWithData(
    DataAdaptor* dataAdaptor, vtkCPInputDataDescription* desc);
private:
  CatalystAnalysisAdaptor(const CatalystAnalysisAdaptor&); // Not implemented.
  void operator=(const CatalystAnalysisAdaptor&); // Not implemented.
};

}

#endif
