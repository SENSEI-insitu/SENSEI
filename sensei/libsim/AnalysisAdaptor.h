#ifndef sensei_libsim_AnalysisAdaptor_h
#define sensei_libsim_AnalysisAdaptor_h
#include <string>

#include <sensei/AnalysisAdaptor.h>
#include <mpi.h>

namespace sensei
{
namespace libsim
{
using sensei::DataAdaptor;

class ImageProperties;

/// @brief Analysis adaptor for libsim-based analysis pipelines.
///
/// AnalysisAdaptor is a subclass of AnalysisAdaptor that is
/// that can be used as the superclass for all analysis that uses libsim.
class AnalysisAdaptor : public sensei::AnalysisAdaptor
{
public:
  static AnalysisAdaptor* New();
  vtkTypeMacro(AnalysisAdaptor, sensei::AnalysisAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Set some Libsim startup options.
  void SetTraceFile(const std::string &traceFile);
  void SetOptions(const std::string &options);
  void SetVisItDirectory(const std::string &dir);
  void SetComm(MPI_Comm comm);

  // Let the caller explicitly initialize.
  void Initialize();

  virtual bool Execute(DataAdaptor* data);
  
  // Simple method to add some VisIt plots. The limit is how complex
  // we want to make this.
  bool AddPlots(const std::string &plots,
                const std::string &plotVars,
	        bool slice, bool project2d,
	        const double origin[3], const double normal[3],
	        const ImageProperties &imgProps);
//BTX
protected:
  AnalysisAdaptor();
  ~AnalysisAdaptor();

private:
  AnalysisAdaptor(const AnalysisAdaptor&); // Not implemented.
  void operator=(const AnalysisAdaptor&); // Not implemented.
  
  class PrivateData;
  PrivateData *d;
//ETX
};

} // libsim
} // sensei
#endif
