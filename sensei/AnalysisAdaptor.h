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
 * the ::DataAdaptor API is used to fetch simulation data. The fetched data is
 * transformed as needed by the underlying processing, movement or I/O library.
 * Simulations invoke in situ processing by calling the ::Execute method.
 * Typically simulations will make use of the ::ConfigurableAnalysis.
 *
 * An analysis adaptor can optionally generate an "output" in Execute, and
 * return it via a DataAdaptor.  Such an output may be used for further
 * analysis or provide feedback and other control information  back to the
 * simulation itself.
 */
class AnalysisAdaptor : public svtkObjectBase
{
public:
  senseiBaseTypeMacro(AnalysisAdaptor, svtkObjectBase);

  /// Prints the current state of the adaptor.
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /// Set the level of verbosity of console output.
  virtual void SetVerbose(int val){ this->Verbose = val; }

  /// Get the level of verbosity of console output.
  virtual int GetVerbose(){ return this->Verbose; }

  /** Set the MPI communicator to be used by the adaptor.
   * The default communicator is a duplicate of MPI_COMMM_WORLD, giving
   * each adaptor a unique communication space. Users wishing to override
   * this should set the communicator before doing anything else. Derived
   * classes should use the communicator returned by GetCommunicator.
   */
  virtual int SetCommunicator(MPI_Comm comm);

  /// returns the MPI communicator to be used for all communication
  MPI_Comm GetCommunicator() { return this->Comm; }

  /** Invokes in situ processing, data movement or I/O. The simulation will
   * call this method when data is ready to be processed. Callers will pass a
   * simulation specific ::DataAdaptor that can be used to fetch the needed
   * simulation data for processing.
   *
   * @param [in] data the simulation provided data adaptor used to fetch data
   *                  for processing
   * @returns false if an error has occurred. Typically this means that in
   *          situ processing is not possible due to misconfiguration or communication
   *          error. In that case callers should abort so as not to waste compute
   *          resources.
   */
  virtual bool Execute(DataAdaptor* data, DataAdaptor*& result) = 0;

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
  ~AnalysisAdaptor();

  AnalysisAdaptor(const AnalysisAdaptor&) = delete;
  void operator=(const AnalysisAdaptor&) = delete;

  MPI_Comm Comm;
  int Verbose;
};

}
#endif
