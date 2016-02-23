#ifndef sensei_adios_AnalysisAdaptor_h
#define sensei_adios_AnalysisAdaptor_h

#include <sensei/AnalysisAdaptor.h>
#include <string>

namespace sensei
{
namespace adios
{
using sensei::DataAdaptor;

/// @brief Analysis adaptor for ADIOS.
///
/// adios::AnalysisAdaptor is an subclass of sensei::AnalysisAdaptor. Despite
/// being called an analysis adaptor, this adaptor doesn't do any analysis. It's
/// main purpose is to serialize data provided via DataAdaptor using
/// ADIOS. In theory, this class should be able to handle all types of
/// vtkDataObject subclasses. Current implementation only supports vtkImageData
/// and vtkMultiBlockDataSet of vtkImageData.
///
/// \sa vtkADIOSDataAdaptor, ADIOSAnalysisEndPoint
class AnalysisAdaptor : public sensei::AnalysisAdaptor
{
public:
  static AnalysisAdaptor* New();
  vtkTypeMacro(AnalysisAdaptor, sensei::AnalysisAdaptor);
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

  virtual bool Execute(DataAdaptor* data);

//BTX
protected:
  AnalysisAdaptor();
  ~AnalysisAdaptor();

  void InitializeADIOS(DataAdaptor* data);
  void WriteTimestep(DataAdaptor* data);

  std::string Method;
  std::string FileName;
private:
  AnalysisAdaptor(const AnalysisAdaptor&); // Not implemented.
  void operator=(const AnalysisAdaptor&); // Not implemented.

  bool Initialized;
  int64_t FixedLengthVarSize;

//ETX
};

} // adios
} // sensei
#endif
