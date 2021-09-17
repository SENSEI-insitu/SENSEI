#ifndef sensei_AnalysisAdaptor_h
#define sensei_AnalysisAdaptor_h

#include "senseiConfig.h"
#include <svtkObjectBase.h>
#include <mpi.h>

namespace sensei
{
class DataAdaptor;

/** The base class for data consumers. Implementors will override ::Eexcute,
 * and ::Finalize and can choose to implement an Initialze method. In ::Execute
 * the sensei::DataAdaptor API is used to fetch simulation data. The fetched
 * data is transformed as needed by the underlying processing, movement or I/O
 * library.  Simulations invoke in situ processing by calling the ::Execute
 * method.  Typically simulations will make use of the
 * sensei::ConfigurableAnalysis.
 *
 * An analysis adaptor can optionally return a sensei::DataAdaptor instance
 * from ::Execute. Such an output may be used for further analysis or provide
 * feedback and other control information  back to the simulation itself.
 */
class SENSEI_EXPORT AnalysisAdaptor : public svtkObjectBase
{
public:
  senseiBaseTypeMacro(AnalysisAdaptor, svtkObjectBase);

  /// Prints the current state of the adaptor.
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /** Set the MPI communicator to be used by the adaptor.
   * The default communicator is a duplicate of MPI_COMMM_WORLD, giving
   * each adaptor a unique communication space. Users wishing to override
   * this should set the communicator before doing anything else. Derived
   * classes should use the communicator returned by GetCommunicator.
   */
  virtual int SetCommunicator(MPI_Comm comm);

  /// returns the MPI communicator to be used for all communication
  MPI_Comm GetCommunicator() { return this->Comm; }

  /** Set the level of verbosity of console output. The environment variable
   * `SENSEI_VERBOSE` provides the initial value if set. Otherwise the default
   * is 0.
   */
  virtual void SetVerbose(int val){ this->Verbose = val; }

  /// Get the level of verbosity of console output.
  virtual int GetVerbose(){ return this->Verbose; }

  /** When set the analysis should buffer the simulation data and run in the
   * background returning to the simulation immediately. This mode requires
   * MPI_THREAD_MULTIPLE and each thread must use a different communictor
   * or serialize the calls to MPI collectives. The environment variable
   * `SENSEI_ASYNCHRONOUS` provides the initial value. Oterwise the default
   * is 0.
   */
  virtual void SetAsynchronous(int val){ this->Asynchronous = val; }

  /// Get asynchronous mode.
  virtual int GetAsynchronous(){ return this->Asynchronous; }

  /// values controling device selection
  enum {DEVICE_HOST=-1, DEVICE_AUTO=-2};

  /** Set the device that the analysis should run on. A value of DEVICE_HOST
   * (-1) indicates that the analysis should run on the host while a value
   * greater or equal to 0 specifies the device explicitly. The special value
   * of DEVICE_AUTO (-2) is reserved for automatic device selection. The
   * environment variable `SENSEI_DEVICE_ID` provides the initial value.
   * Otherwise the default is DEVICE_HOST. See ::GetDeviceId for an explanation
   * of the automatic device selection algorithm.
   */
  virtual void SetDeviceId(int val){ this->DeviceId = val; }

  /** Get the device that the analysis should run on. When ::SetDeviceId has
   * the special value of DEVICE_AUTO (-2) the * following algorithm is used:
   * ```
   * DeviceId = ( MPI rank % DevicesInUse * DeviceStride + DeviceStart ) % DevicesPerNode
   * ```
   * See ::SetDeviceId.
   */
  virtual int GetDeviceId();

  /** Set the number of devices to use per node. This value can be smaller than
   * the number of actual devices but should not exceed it. The environment
   * variable `SENSEI_DEVICES_TO_USE` provides the initial value. Otherwise the
   * default is the number of actual devices available. See ::SetDeviceId for
   * an explanation of automatic device selection.
   */
  virtual void SetDevicesToUse(int val){ this->DevicesToUse = val; }

  /// Get the number of devices to use per node.
  virtual int GetDevicesToUse(){ return this->DevicesToUse; }

  /** Set the first on node device to use in automatic device selection.  The
   * environment variable `SENSEI_DEVICE_START` provides the initial value.
   * Otherwise the default is 0.  See ::SetDeviceId for an explanation of
   * automatic device selection.
   */
  virtual void SetDeviceStart(int val){ this->DeviceStart = val; }

  /// Get the first device to use
  virtual int GetDeviceStart(){ return this->DeviceStart; }

  /** Set the number of devices to skip in automatic device selection.  The
   * environment variable `SENSEI_DEVICE_STRIDE` provides the initial value.
   * Otherwise the default is 0.  See ::SetDeviceId for an explanation of
   * automatic device selection.
   */
  virtual void SetDeviceStride(int val){ this->DeviceStride = val; }

  /// Get the number of devices to skip
  virtual int GetDeviceStride(){ return this->DeviceStride; }


  /** Invokes in situ processing, data movement or I/O. The simulation will
   * call this method when data is ready to be processed. Callers will pass a
   * simulation specific sensei::DataAdaptor that can be used to fetch the
   * needed simulation data for processing. Callers pass a non-null address to
   * a pointer to a sensei::DataAdaptor to signal that output is desired if it
   * is available. In that case a newly allocated data adaptor instance is
   * returned in the pointer. This data adaptor can be used to fetch the
   * output. The caller trakes ownership of the returned data adaptor instance
   * and must call Delete on it when finished. Callers that do not want the
   * output data should pass nullptr to signal that it is not needed.
   *
   * @param [in] dataIn the simulation provided data adaptor used to fetch data
   *                    for processing
   * @param [out] dataOut the address of a pointer to a data adaptor that could
   *                      be used to fetch data from the analysis. This should
   *                      be null if the caller does not want to access the
   *                      output data. If it is not null and if the
   *                      implementation can provide output a data, a data
   *                      adaptor is allocated and returned via this pointer.
   *                      In that case the caller can use the adaptor to fetch
   *                      the data. The caller takes ownership of the returned
   *                      data adaptor and must call Delete on it when
   *                      finished.
   * @returns false if an error has occurred. Typically this means that in
   *          situ processing is not possible due to misconfiguration or communication
   *          error. In that case callers should abort so as not to waste compute
   *          resources.
   */
  virtual bool Execute(DataAdaptor* dataIn, DataAdaptor** dataOut) = 0;

  /** Clean up and shut down the data consuming library if needed.  This method
   * is called when the run is finsihed clean up and shut down should occur
   * here rather than in the destructor as MPI may not be available at
   * desctruction time.
   *
   * @returns zero if successful.
   */
  virtual int Finalize() { return 0; }

protected:
  AnalysisAdaptor();
  virtual ~AnalysisAdaptor();

  AnalysisAdaptor(const AnalysisAdaptor&) = delete;
  void operator=(const AnalysisAdaptor&) = delete;

  MPI_Comm Comm;
  int Verbose;
  int DeviceId;
  int DevicesPerNode;
  int DevicesToUse;
  int DeviceStart;
  int DeviceStride;
  int Asynchronous;
};

}
#endif
