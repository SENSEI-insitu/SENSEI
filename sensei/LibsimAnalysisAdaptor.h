#ifndef sensei_LibsimAnalysisAdaptor_h
#define sensei_LibsimAnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include <string>
#include <mpi.h>

namespace sensei
{

class LibsimImageProperties;

/// @brief Analysis adaptor for libsim-based analysis pipelines.
///
/// LibsimAnalysisAdaptor is a subclass of sensei::AnalysisAdaptor that
/// can be used as the superclass for all analysis that uses libsim.
class LibsimAnalysisAdaptor : public AnalysisAdaptor
{
public:
  static LibsimAnalysisAdaptor* New();
  senseiTypeMacro(LibsimAnalysisAdaptor, AnalysisAdaptor);

  // Set some Libsim startup options.
  void SetTraceFile(const std::string &traceFile);
  void SetOptions(const std::string &options);
  void SetVisItDirectory(const std::string &dir);
  void SetMode(const std::string &mode);

  // if set then the adaptor will generate the nesting information
  // for VisIt. If one wants to do subsetting in VisIt, setting this
  // flag will give VisIt the information it needs to re-generate ghost
  // zones based on the selected blocks. It is off by default because
  // if not doing subsetting this information is not needed, and
  // this information takes up a good deal of space and the algorithm
  // used here is O(N^2) in the number of blocks.
  void SetComputeNesting(int val);

  // Let the caller explicitly initialize.
  void Initialize();

  bool Execute(DataAdaptor* data) override;

  int Finalize() override;

  // Simple method to add some VisIt plots and render. The limit is how complex
  // we want to make this.
  bool AddRender(int frequency,
            const std::string &session,
            const std::string &plots,
            const std::string &plotVars,
            bool slice, bool project2d,
            const double origin[3], const double normal[3],
            const LibsimImageProperties &imgProps);

  bool AddExport(int frequency,
                 const std::string &session,
                 const std::string &plot, const std::string &plotVars,
                 bool slice, bool project2d,
                 const double origin[3], const double normal[3],
                 const std::string &filename);
protected:
  LibsimAnalysisAdaptor();
  ~LibsimAnalysisAdaptor();

private:
  LibsimAnalysisAdaptor(const LibsimAnalysisAdaptor&); // Not implemented.
  void operator=(const LibsimAnalysisAdaptor&); // Not implemented.

  class PrivateData;
  PrivateData *internals;
};

}

#endif
