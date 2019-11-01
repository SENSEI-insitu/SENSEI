#ifndef sensei_AnalysisAdaptor_h
#define sensei_AnalysisAdaptor_h

#include "senseiConfig.h"
#include <vtkObjectBase.h>
#include <mpi.h>

namespace sensei
{
class DataAdaptor;

/// @class AnalysisAdaptor
/// @brief AnalysisAdaptor is an abstract base class that defines the analysis interface.
///
/// AnalysisAdaptor is an adaptor for any insitu analysis framework or
/// algorithm. Concrete subclasses use DataAdaptor instance passed to
/// the Execute() method to access simulation data for further processing.
class AnalysisAdaptor : public vtkObjectBase
{
public:
  senseiBaseTypeMacro(AnalysisAdaptor, vtkObjectBase);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /// @breif Set the level of verbosity of console output.
  virtual void SetVerbose(int val){ this->Verbose = val; }
  virtual int GetVerbose(){ return this->Verbose; }

  /// @brief Set the communicator used by the adaptor.
  /// The default communicator is a duplicate of MPI_COMMM_WORLD, giving
  /// each adaptor a unique communication space. Users wishing to override
  /// this should set the communicator before doing anything else. Derived
  /// classes should use the communicator returned by GetCommunicator.
  virtual int SetCommunicator(MPI_Comm comm);
  MPI_Comm GetCommunicator() { return this->Comm; }

  /// @brief Execute the analysis routine.
  ///
  /// This method is called to execute the analysis routine per simulation
  /// iteration.
  virtual bool Execute(DataAdaptor* data) = 0;

  /// @breif Finalize the analyis routine
  ///
  /// This method is called when the run is finsihed clean up
  /// and shut down should occur here rather than in the destructor
  /// as MPI may not be available at desctruction time for instance
  /// when smart pointers are used MPI is finalized before the
  /// pointer goes out of scope and is destroyed.
  ///
  /// @returns zero if successful
  virtual int Finalize() = 0;

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
