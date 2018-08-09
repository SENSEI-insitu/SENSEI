#ifndef sensei_AscentAnalysisAdaptor_h
#define sensei_AscentAnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include <string>
#include <mpi.h>

namespace sensei
{

/// @brief Analysis adaptor for ascent-based analysis pipelines.
///
/// AscentAnalysisAdaptor is a subclass of sensei::AnalysisAdaptor that
/// can be used as the superclass for all analysis that uses libsim.
class AscentAnalysisAdaptor : public AnalysisAdaptor
{
public:
  static AscentAnalysisAdaptor* New();
  senseiTypeMacro(AscentAnalysisAdaptor, AnalysisAdaptor);

  // Let the caller explicitly initialize.
  void Initialize();

  bool Execute(DataAdaptor* data) override;

  int Finalize() override;

protected:
  AscentAnalysisAdaptor();
  ~AscentAnalysisAdaptor();

private:
  AscentAnalysisAdaptor(const AscentAnalysisAdaptor&)=delete; // Not implemented.
  void operator=(const AscentAnalysisAdaptor&)=delete; // Not implemented.

};

}

#endif
