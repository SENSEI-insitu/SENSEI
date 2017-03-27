#ifndef sensei_ADIOSAnalysisAdaptor_h
#define sensei_ADIOSAnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include <string>

namespace sensei
{

/// @brief Analysis adaptor for ADIOS.
///
/// ADIOSAnalysisAdaptor is an subclass of sensei::AnalysisAdaptor. Despite
/// being called an analysis adaptor, this adaptor doesn't do any analysis. It's
/// main purpose is to serialize data provided via DataAdaptor using
/// ADIOS. In theory, this class should be able to handle all types of
/// vtkDataObject subclasses. Current implementation only supports vtkImageData
/// and vtkMultiBlockDataSet of vtkImageData.
///
/// \sa vtkADIOSDataAdaptor, ADIOSAnalysisEndPoint
class ADIOSAnalysisAdaptor : public AnalysisAdaptor
{
public:
  static ADIOSAnalysisAdaptor* New();
  senseiTypeMacro(ADIOSAnalysisAdaptor, AnalysisAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent) override;

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

  bool Execute(DataAdaptor* data) override;

protected:
  ADIOSAnalysisAdaptor();
  ~ADIOSAnalysisAdaptor();

  void InitializeADIOS(DataAdaptor* data);
  void WriteTimestep(DataAdaptor* data);

  std::string Method;
  std::string FileName;
private:
  ADIOSAnalysisAdaptor(const ADIOSAnalysisAdaptor&); // Not implemented.
  void operator=(const ADIOSAnalysisAdaptor&); // Not implemented.

  bool Initialized;
  int64_t FixedLengthVarSize;
};

}

#endif
