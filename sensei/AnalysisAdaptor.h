#ifndef sensei_AnalysisAdaptor_h
#define sensei_AnalysisAdaptor_h

#include "vtkObjectBase.h"
#include "vtkSetGet.h"

namespace sensei
{
class DataAdaptor;

/// @class AnalysisAdaptor
/// @brief AnalysisAdaptor is an abstract base class that defines the analysis interface.
///
/// AnalysisAdaptor is an adaptor for any insitu analysis framework or
/// algorithm. Concrete subclasses use DataAdaptor instance passed to
/// the Execute() method to access simulation data for further processing.
class AnalysisAdaptor : public vtkObjectBase
{
public:
  vtkTypeMacro(AnalysisAdaptor, vtkObjectBase);
  void PrintSelf(ostream& os, vtkIndent indent);

  /// @brief Execute the analysis routine.
  ///
  /// This method is called to execute the analysis routine per simulation
  /// iteration.
  virtual bool Execute(DataAdaptor* data) = 0;

protected:
  AnalysisAdaptor();
  ~AnalysisAdaptor();

private:
  AnalysisAdaptor(const AnalysisAdaptor&); // Not implemented.
  void operator=(const AnalysisAdaptor&); // Not implemented.
};

}
#endif
