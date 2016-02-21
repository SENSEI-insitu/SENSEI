#ifndef vtkADIOSAnalysisAdaptor_h
#define vtkADIOSAnalysisAdaptor_h

#include <vtkInsituAnalysisAdaptor.h>
#include <string>

/// @brief Analysis adaptor for ADIOS.
///
/// vtkADIOSAnalysisAdaptor is an subclass of vtkInsituAnalysisAdaptor. Despite
/// being called an analysis adaptor, this adaptor doesn't do any analysis. It's
/// main purpose is to serialize data provided via vtkInsituDataAdaptor using
/// ADIOS. In theory, this class should be able to handle all types of
/// vtkDataObject subclasses. Current implementation only supports vtkImageData
/// and vtkMultiBlockDataSet of vtkImageData.
///
/// \sa vtkADIOSDataAdaptor, ADIOSAnalysisEndPoint
class vtkADIOSAnalysisAdaptor : public vtkInsituAnalysisAdaptor
{
public:
  static vtkADIOSAnalysisAdaptor* New();
  vtkTypeMacro(vtkADIOSAnalysisAdaptor, vtkInsituAnalysisAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent);

  /// @brief Set the ADIOS method e.g. MPI, FLEXPATH etc.
  ///
  /// Default value is "MPI".
  void SetMethod(const std::string &method)
    { this->Method = method; }
  std::string GetMethod() const
    { return this->Method; }

  /// @brief Set the filename.
  ///
  /// Default value is "sensei.bp"
  void SetFileName(const std::string &filename)
    { this->FileName = filename; }
  std::string GetFileName() const
    { return this->FileName; }

  virtual bool Execute(vtkInsituDataAdaptor* data);

//BTX
protected:
  vtkADIOSAnalysisAdaptor();
  ~vtkADIOSAnalysisAdaptor();

  void InitializeADIOS(vtkInsituDataAdaptor* data);
  void WriteTimestep(vtkInsituDataAdaptor* data);

  std::string Method;
  std::string FileName;
private:
  vtkADIOSAnalysisAdaptor(const vtkADIOSAnalysisAdaptor&); // Not implemented.
  void operator=(const vtkADIOSAnalysisAdaptor&); // Not implemented.

  bool Initialized;
  int64_t FixedLengthVarSize;

//ETX
};

#endif
