#ifndef sensei_ADIOSAnalysisAdaptor_h
#define sensei_ADIOSAnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include <string>
#include <mpi.h>

namespace senseiADIOS { class DataObjectSchema; }
class vtkDataObject;

namespace sensei
{

/// @brief Analysis adaptor for ADIOS.
///
/// ADIOSAnalysisAdaptor is an subclass of sensei::AnalysisAdaptor. Despite
/// being called an analysis adaptor, this adaptor doesn't do any analysis. It's
/// main purpose is to serialize data provided via DataAdaptor using
/// ADIOS.
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

  /// Set the communicator to use for MPI calls
  void SetCommunicator(MPI_Comm comm)
  { this->Comm = comm; }

  bool Execute(DataAdaptor* data) override;

  int Finalize() override;

protected:
  ADIOSAnalysisAdaptor();
  ~ADIOSAnalysisAdaptor();

  void InitializeADIOS(vtkDataObject *dobj);
  void WriteTimestep(unsigned long timeStep, double time, vtkDataObject *dobj);
  void FinalizeADIOS();

  MPI_Comm Comm;
  senseiADIOS::DataObjectSchema *Schema;
  std::string Method;
  std::string FileName;

private:
  ADIOSAnalysisAdaptor(const ADIOSAnalysisAdaptor&); // Not implemented.
  void operator=(const ADIOSAnalysisAdaptor&); // Not implemented.
};

}

#endif
