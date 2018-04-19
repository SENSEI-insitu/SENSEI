#ifndef sensei_AnalysisAdaptor_h
#define sensei_AnalysisAdaptor_h

#include "senseiConfig.h"
#include <vtkVersionMacros.h>
#if VTK_MAJOR_VERSION >= 8
#include <vtkObject.h>
#else
#include <vtkObjectBase.h>
#endif

#include <vtkSetGet.h>

namespace sensei
{
#if VTK_MAJOR_VERSION >= 8
typedef vtkObject AnalysisAdaptorBase;
#else
typedef vtkObjectBase AnalysisAdaptorBase;
#endif

class DataAdaptor;

/// @class AnalysisAdaptor
/// @brief AnalysisAdaptor is an abstract base class that defines the analysis interface.
///
/// AnalysisAdaptor is an adaptor for any insitu analysis framework or
/// algorithm. Concrete subclasses use DataAdaptor instance passed to
/// the Execute() method to access simulation data for further processing.
class AnalysisAdaptor : public AnalysisAdaptorBase
{
public:
  senseiBaseTypeMacro(AnalysisAdaptor, AnalysisAdaptorBase);
  virtual void PrintSelf(ostream& os, vtkIndent indent) override;

  /// @brief Execute the analysis routine.
  ///
  /// This method is called to execute the analysis routine per simulation
  /// iteration.
  virtual bool Execute(DataAdaptor* data) = 0;

  /// @breif Finalize the analyis routine
  ///
  /// This method is called when the run is finsihed clean up
  /// and shut down should occur here rather than in the destructor
  /// as MPI may not be available at desctruction time for instance
  /// when smart pointers are used MPI is finalized before the
  /// pointer goes out of scope and is destroyed.
  ///
  /// @returns zero if successful
  virtual int Finalize() = 0;

protected:
  AnalysisAdaptor();
  ~AnalysisAdaptor();

private:
  AnalysisAdaptor(const AnalysisAdaptor&); // Not implemented.
  void operator=(const AnalysisAdaptor&); // Not implemented.
};

}
#endif
