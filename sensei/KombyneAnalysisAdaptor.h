#ifndef KombyneAnalysisAdaptor_h
#define KombyneAnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include "MeshMetadata.h"

#include <kombyne_execution.h>

#include <svtkDataObject.h>
#include <svtkSmartPointer.h>

#include <string>

using svtkDataObjectPtr = svtkSmartPointer<svtkDataObject>;

namespace sensei
{

/** An adaptor that invokes Kombyne. The adaptor is configured via a
 * Kombyne pipeline config file.
 */
class SENSEI_EXPORT KombyneAnalysisAdaptor : public AnalysisAdaptor
{
public:
  senseiTypeMacro(KombyneAnalysisAdaptor, AnalysisAdaptor);

  /// creates a new instance
  static KombyneAnalysisAdaptor* New();

  /// Initializes Kombyne
  void Initialize();

  /// Invoke in situ processing using Kombyne
  bool Execute(DataAdaptor* data, DataAdaptor**) override;

  /// Shuts Kombyne down
  int Finalize() override;

  ///@name Run time configuration
  ///@{

  int SetPipelineFile(std::string filename);
  int SetSessionName(std::string sessionname);
  int SetMode(std::string mode);
  void SetVerbose(int verbose);

  // KB_RESULTS_CACHE_ENABLED:  true
  // KB_RESULTS_CACHE_STORAGE:  8589934592 
  // KB_SIGNALS:                true
  // KB_START_PORT:             5900
  // KB_TIMERS_ENABLED:         false
  // KB_TIMERS_FILENAME:        ""
  // KB_TIMERS_PROGRESSIVE:     true
  // KB_TRANSPORT:              adios2 or zeromq
  // KB_VERBOSE:                true

  ///@}

protected:
  KombyneAnalysisAdaptor();
  ~KombyneAnalysisAdaptor();

  int GetMetaData(void);
  int GetMesh(const std::string &meshName, svtkDataObjectPtr &dobjp);

  svtkDataObject *GetMeshBlock(
      MPI_Comm comm, const int domain, MeshMetadataPtr mdptr);

  void ClearCache();

  DataAdaptor *Adaptor;
  std::map<std::string, svtkDataObjectPtr> Meshes;
  std::map<std::string, sensei::MeshMetadataPtr> Metadata;

  std::string pipelineFile;
  std::string sessionName;
  kb_role role;
  bool verbose;
  bool initialized;

  kb_pipeline_collection_handle hp;

private:
  KombyneAnalysisAdaptor(const KombyneAnalysisAdaptor&); // Not implemented.
  void operator=(const KombyneAnalysisAdaptor&); // Not implemented.
};

}

#endif
