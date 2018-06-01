#ifndef sensei_ADIOSAnalysisAdaptor_h
#define sensei_ADIOSAnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include "DataRequirements.h"

#include <vector>
#include <string>
#include <mpi.h>

namespace senseiADIOS { class DataObjectCollectionSchema; }
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

  /// Sets the maximum buffer allocated by ADIOS in MB
  /// takes affect on first Execute
  void SetMaxBufferSize(unsigned int size)
  { this->MaxBufferSize = size; }

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

  /// data requirements tell the adaptor what to push
  /// if none are given then all data is pushed.
  int SetDataRequirements(const DataRequirements &reqs);

  int AddDataRequirement(const std::string &meshName,
    int association, const std::vector<std::string> &arrays);

  bool Execute(DataAdaptor* data) override;

  int Finalize() override;

protected:
  ADIOSAnalysisAdaptor();
  ~ADIOSAnalysisAdaptor();

  // intializes ADIOS in no-xml mode, allocate buffers, and declares a group
  int InitializeADIOS(const std::vector<std::string> &objectNames,
    const std::vector<vtkDataObject*> &objects);

  // writes the data collection
  int WriteTimestep(unsigned long timeStep, double time,
    const std::vector<std::string> &objectNames,
    const std::vector<vtkDataObject*> &dobjects);

  // shuts down ADIOS
  int FinalizeADIOS();

  unsigned int MaxBufferSize;
  senseiADIOS::DataObjectCollectionSchema *Schema;
  sensei::DataRequirements Requirements;
  std::string Method;
  std::string FileName;


private:
  ADIOSAnalysisAdaptor(const ADIOSAnalysisAdaptor&); // Not implemented.
  void operator=(const ADIOSAnalysisAdaptor&); // Not implemented.
};

}

#endif
