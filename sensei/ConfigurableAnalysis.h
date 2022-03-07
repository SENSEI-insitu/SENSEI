#ifndef sensei_ConfigurableAnalysis_h
#define sensei_ConfigurableAnalysis_h

#include "AnalysisAdaptor.h"

#include <string>
#include <mpi.h>

/// @cond
namespace pugi { class xml_node; }
/// @endcond

namespace sensei
{
/** An adaptor that creates and configures one or more adaptors from XML.  When
 * the Execute method is invoked the calls are forwarded to the active
 * instances. The supported adaptors include:
 *
 * | Class | Description |
 * | ----- | ----------- |
 * | sensei::Histogram | Computes histograms |
 * | sensei::ADIOS2AnalysisAdaptor | The write side of the ADIOS2 transport |
 * | sensei::HDF5AnalysisAdaptor | The write side of the HDF5 transport |
 * | sensei::AscentAnalysisAdaptor | Processes simulation data using Ascent |
 * | sensei::CatalystAnalysisAdaptor | Processes simulation data using ParaView Catalyst |
 * | sensei::LibsimAnalysisAdaptor | Processes simulation data using VisIt Libsim |
 * | sensei::Autocorrelation | Compute autocorrelation of simulation data over time |
 * | sensei::VTKPosthocIO | Writes simulation data to disk in a SVTK format |
 * | sensei::VTKAmrWriter | Writes simulation data to disk in a SVTK format |
 * | sensei::PythonAnalysis | Invokes user provided Pythons scripts that process simulation data |
 * | sensei::SliceExtract | Computes planar slices and iso-surfaces on simulation data |
 *
 */
class SENSEI_EXPORT ConfigurableAnalysis : public AnalysisAdaptor
{
public:
  /// creates a new instance.
  static ConfigurableAnalysis *New();

  senseiTypeMacro(ConfigurableAnalysis, AnalysisAdaptor);

  /// Prints the current adaptor state
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /** Set the communicator used by the adaptor.
   * The default communicator is a duplicate of MPI_COMMM_WORLD, giving
   * each adaptor a unique communication space. Users wishing to override
   * this should set the communicator before doing anything else. Derived
   * classes should use the communicator returned by GetCommunicator.
   */
  int SetCommunicator(MPI_Comm comm) override;

  /// Initialize the adaptor using the configuration specified.
  int Initialize(const std::string &filename);

  /// Initialize the adaptor using the configuration specified.
  int Initialize(const pugi::xml_node &root);

  /// Invokes the Execute method on the currently configured adaptors.
  bool Execute(DataAdaptor *data, DataAdaptor **result) override;

  /// Invokes the Finalize method on the currently configured adaptors.
  int Finalize() override;

protected:
  ConfigurableAnalysis();
  ~ConfigurableAnalysis();

  ConfigurableAnalysis(const ConfigurableAnalysis&) = delete;
  void operator=(const ConfigurableAnalysis&) = delete;

private:
  struct InternalsType;
  InternalsType *Internals;
};

}

#endif
