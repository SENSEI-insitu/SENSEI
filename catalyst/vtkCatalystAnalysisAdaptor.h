#ifndef vtkCatalystAnalysisAdaptor_h
#define vtkCatalystAnalysisAdaptor_h

#include "vtkInsituAnalysisAdaptor.h"
class vtkCPInputDataDescription;
class vtkCPPipeline;

/// @brief Analysis adaptor for Catalyst-based analysis pipelines.
///
/// vtkCatalystAnalysisAdaptor is a subclass of vtkInsituAnalysisAdaptor that is
/// that can be used as the superclass for all analysis that uses Catalyst.
class vtkCatalystAnalysisAdaptor : public vtkInsituAnalysisAdaptor
{
public:
  static vtkCatalystAnalysisAdaptor* New();
  vtkTypeMacro(vtkCatalystAnalysisAdaptor, vtkInsituAnalysisAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent);

  /// @brief Add a vtkCPPipeline subclass to the global Catalyst Co-Processor.
  ///
  /// Adds a vtkCPPipeline subclass to the global Catalyst Co-Processor.
  virtual void AddPipeline(vtkCPPipeline* pipeline);

  virtual bool Execute(vtkInsituDataAdaptor* data);
//BTX
protected:
  vtkCatalystAnalysisAdaptor();
  ~vtkCatalystAnalysisAdaptor();

  /// @brief Fill \c desc with meta data from \c dataAdaptor.
  ///
  /// Called before the RequestDataDescription step to fill \c desc with
  /// information about fields (and any other meta-data) available in the
  /// \c dataAdaptor.
  /// @return true on success, false to abort execution.
  virtual bool FillDataDescriptionWithMetaData(
    vtkInsituDataAdaptor* dataAdaptor, vtkCPInputDataDescription* desc);

  /// @brief Fill \c desc with data from \c dataAdaptor.
  ///
  /// Called before the CoProcess() step to fill \c desc with
  /// simulation data.
  /// @return true on success, false to abort execution.
  virtual bool FillDataDescriptionWithData(
    vtkInsituDataAdaptor* dataAdaptor, vtkCPInputDataDescription* desc);
private:
  vtkCatalystAnalysisAdaptor(const vtkCatalystAnalysisAdaptor&); // Not implemented.
  void operator=(const vtkCatalystAnalysisAdaptor&); // Not implemented.
//ETX
};

#endif
