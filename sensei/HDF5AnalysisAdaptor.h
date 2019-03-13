#ifndef HDF5AnalysisAdaptor_h
#define HDF5AnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include "DataRequirements.h"
#include "MeshMetadata.h"

#include "hdf5.h"
#include <mpi.h>
#include <string>
#include <vector>

#include "HDF5Schema.h"

// namespace senseiHDF5 { class DataObjectCollectionSchema; }
class vtkDataObject;
class vtkCompositeDataSet;

namespace sensei {

/// The write side of the ADIOS 1 transport
class HDF5AnalysisAdaptor : public AnalysisAdaptor {
public:
  static HDF5AnalysisAdaptor *New();
  senseiTypeMacro(HDF5AnalysisAdaptor, AnalysisAdaptor);
  void PrintSelf(ostream &os, vtkIndent indent) override;

  /// Sets the maximum buffer allocated by HDF5 in MB
  /// takes affect on first Execute
  void SetMaxBufferSize(unsigned int size) { this->MaxBufferSize = size; }

  /// @brief Set the filename.
  ///
  /// Default value is "sensei.bp"
  void SetFileName(const std::string &filename) { this->m_FileName = filename; }

  void SetStreaming(bool streamOption) { this->m_DoStreaming = streamOption; }

  void SetCollective(bool s) { m_Collective = s; }

  std::string GetFileName() const { return this->m_FileName; }

  /// data requirements tell the adaptor what to push
  /// if none are given then all data is pushed.
  int SetDataRequirements(const DataRequirements &reqs);

  int AddDataRequirement(const std::string &meshName, int association,
                         const std::vector<std::string> &arrays);

  // SENSEI AnalysisAdaptor API
  bool Execute(DataAdaptor *data) override;
  int Finalize() override;

protected:
  HDF5AnalysisAdaptor();
  ~HDF5AnalysisAdaptor();

  // intializes HDF5 in no-xml mode, allocate buffers, and declares a group
  // bool InitializeHDF5(const std::vector<MeshMetadataPtr> &metadata);
  bool InitializeHDF5();

  // writes the data collection
  /*
  bool WriteTimestep(unsigned long timeStep, double time,
                     const std::vector<MeshMetadataPtr> &metadata,
                     const std::vector<vtkCompositeDataSet*> &dobjects);
  */
  unsigned int MaxBufferSize;
  sensei::DataRequirements Requirements;
  std::string m_FileName;
  bool m_DoStreaming = false;
  bool m_Collective = false;

private:
  senseiHDF5::WriteStream *m_HDF5Writer;

  HDF5AnalysisAdaptor(const HDF5AnalysisAdaptor &) = delete;
  void operator=(const HDF5AnalysisAdaptor &) = delete;
};

} // namespace sensei

#endif
