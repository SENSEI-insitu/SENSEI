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
 * | sensei::DataBinning | Bins point/particle data onto a cartesian grid with arbitrary axes |
 * | sensei::ParticleDensity | Computes particle spatial density on a cartesian grid |
 *
 */
class SENSEI_EXPORT ConfigurableAnalysis : public AnalysisAdaptor
{
public:
  /// creates a new instance.
  static ConfigurableAnalysis *New();

  senseiTypeMacro(ConfigurableAnalysis, AnalysisAdaptor);

  void PrintSelf(ostream& os, svtkIndent indent) override;

  int SetCommunicator(MPI_Comm comm) override;

  void SetVerbose(int val) override;

  void SetAsynchronous(int val) override;

  void SetDeviceId(int val) override;
  void SetDevicesToUse(int val) override;
  void SetDeviceStart(int val) override;
  void SetDeviceStride(int val) override;

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
