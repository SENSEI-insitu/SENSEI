#ifndef vtkConfigurableAnalysisAdaptor_h
#define vtkConfigurableAnalysisAdaptor_h

#include <vtkInsituAnalysisAdaptor.h>

#include <string>
#include <mpi.h>

/// @brief vtkConfigurableAnalysisAdaptor is all-in-one analysis adaptor that
/// can execute all available analysis adaptors.
class vtkConfigurableAnalysisAdaptor : public vtkInsituAnalysisAdaptor
{
public:
  static vtkConfigurableAnalysisAdaptor* New();
  vtkTypeMacro(vtkConfigurableAnalysisAdaptor, vtkInsituAnalysisAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent);

  /// @brief Initialize the adaptor using the configuration specified.
  bool Initialize(MPI_Comm world, const std::string& filename);

  virtual bool Execute(vtkInsituDataAdaptor* data);
//BTX
protected:
  vtkConfigurableAnalysisAdaptor();
  ~vtkConfigurableAnalysisAdaptor();

private:
  vtkConfigurableAnalysisAdaptor(const vtkConfigurableAnalysisAdaptor&); // Not implemented.
  void operator=(const vtkConfigurableAnalysisAdaptor&); // Not implemented.

  class vtkInternals;
  vtkInternals* Internals;
//ETX
};

#endif
