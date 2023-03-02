#ifndef KombyneAnalysisAdaptor_h
#define KombyneAnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include "MeshMetadata.h"

#include <string>

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

  int SetConfigFile(std::string cfgfile);
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

private:
  KombyneAnalysisAdaptor(const KombyneAnalysisAdaptor&); // Not implemented.
  void operator=(const KombyneAnalysisAdaptor&); // Not implemented.

  std::string configFile;
  std::string logFile;
  std::string mode;
  bool verbose;
};

}

#endif
