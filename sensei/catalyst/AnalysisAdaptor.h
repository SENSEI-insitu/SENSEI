#ifndef sensei_catalyst_AnalysisAdaptor_h
#define sensei_catalyst_AnalysisAdaptor_h

#include <sensei/AnalysisAdaptor.h>

class vtkCPInputDataDescription;
class vtkCPPipeline;

namespace sensei
{
namespace catalyst
{
using sensei::DataAdaptor;

/// @brief Analysis adaptor for Catalyst-based analysis pipelines.
///
/// AnalysisAdaptor is a subclass of AnalysisAdaptor that is
/// that can be used as the superclass for all analysis that uses Catalyst.
class AnalysisAdaptor : public sensei::AnalysisAdaptor
{
public:
  static AnalysisAdaptor* New();
  vtkTypeMacro(AnalysisAdaptor, sensei::AnalysisAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent);

  /// @brief Add a vtkCPPipeline subclass to the global Catalyst Co-Processor.
  ///
  /// Adds a vtkCPPipeline subclass to the global Catalyst Co-Processor.
  virtual void AddPipeline(vtkCPPipeline* pipeline);

  virtual bool Execute(DataAdaptor* data);
//BTX
protected:
  AnalysisAdaptor();
  ~AnalysisAdaptor();

  /// @brief Fill \c desc with meta data from \c dataAdaptor.
  ///
  /// Called before the RequestDataDescription step to fill \c desc with
  /// information about fields (and any other meta-data) available in the
  /// \c dataAdaptor.
  /// @return true on success, false to abort execution.
  virtual bool FillDataDescriptionWithMetaData(
    DataAdaptor* dataAdaptor, vtkCPInputDataDescription* desc);

  /// @brief Fill \c desc with data from \c dataAdaptor.
  ///
  /// Called before the CoProcess() step to fill \c desc with
  /// simulation data.
  /// @return true on success, false to abort execution.
  virtual bool FillDataDescriptionWithData(
    DataAdaptor* dataAdaptor, vtkCPInputDataDescription* desc);
private:
  AnalysisAdaptor(const AnalysisAdaptor&); // Not implemented.
  void operator=(const AnalysisAdaptor&); // Not implemented.
//ETX
};

} // catalyst
} // sensei
#endif
