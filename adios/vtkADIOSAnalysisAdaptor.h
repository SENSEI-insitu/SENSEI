#ifndef vtkADIOSAnalysisAdaptor_h
#define vtkADIOSAnalysisAdaptor_h

#include <vtkInsituAnalysisAdaptor.h>
#include <vtkNew.h>

class vtkADIOSAnalysisAdaptor : public vtkInsituAnalysisAdaptor
{
public:
  static vtkADIOSAnalysisAdaptor* New();
  vtkTypeMacro(vtkADIOSAnalysisAdaptor, vtkInsituAnalysisAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent);

  /// @brief Set the ADIOS method e.g. MPI, FLEXPATH etc.
  vtkSetStringMacro(Method);
  vtkGetStringMacro(Method);

  /// @brief Set the filename.
  vtkSetStringMacro(FileName);
  vtkGetStringMacro(FileName);

  virtual bool Execute(vtkInsituDataAdaptor* data);

//BTX
protected:
  vtkADIOSAnalysisAdaptor();
  ~vtkADIOSAnalysisAdaptor();

  void InitializeADIOS(vtkInsituDataAdaptor* data);
  void WriteTimestep(vtkInsituDataAdaptor* data);

  char* Method;
  char* FileName;
private:
  vtkADIOSAnalysisAdaptor(const vtkADIOSAnalysisAdaptor&); // Not implemented.
  void operator=(const vtkADIOSAnalysisAdaptor&); // Not implemented.

  bool Initialized;

//ETX
};

#endif
