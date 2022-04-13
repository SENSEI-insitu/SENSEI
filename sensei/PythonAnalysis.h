#ifndef sensei_PythonAnalysis_h
#define sensei_PythonAnalysis_h

#include "senseiConfig.h"
#include "AnalysisAdaptor.h"
#include <mpi.h>

namespace sensei
{
class DataAdaptor;

/** Loads and executes a Python script impementing the sensei::AnalysisAdaptor
 * API. The script should define the following functions:
 *
 * ```python
 * def Initialize():
 *   """ Initialization code here """
 *   return
 *
 * def Execute(dataAdaptor):
 *   """ Use sensei::DataAdaptor API to process data here """
 *   return
 *
 * def Finalize():
 *   """ Finalization code here """
 *   return
 * ```
 *
 * "Initialize" and "Finalize" are optional, while "Execute" is required. The script
 * is specified at run time either as a module or a file. If a module is
 * specified (see SetScriptModule) the provided module is imported through
 * Python's built in import mechanism. This means that it must be in a
 * directory in the `PYTHONPATH`. If a file is specified (see SetScriptFile)
 * the file is read on rank 0 and broadcast to other ranks. Use either the
 * module or the file approach, but not both.
 *
 * The active MPI communicator is made available to the script through the
 * global variable `comm`.
 *
 * To fine tune run time behavior we provide "initialization source". The
 * initialization source (see SetInitializeSource) is provided in a string and
 * will be executed prior to your script functions. This lets you set global
 * variables that can modify the scripts run time behavior.
 *
 * The compiled artifacts of this class and the sensei Python module  must be
 * findable in both the `PYTHONPATH` and the `LD_LIBRARY_PATH`
 * (`DYLD_LIBRARY_PATH` on Mac OS)
 *
 * The user provided Execute function returns one of four possible things:
 *
 * 1. None. Success is assumed in this case, and processing continues.
 * 2. An integer status code where 0 indicates to stop in situ processing and
 *    non-zero indicates to continue in situ processing.
 * 3. A data adaptor instance. The data adaptor instance can be used to access
 *    the results of the operation. Success is assumed in this case and processing
 *    continues.
 * 4. A tuple containing an integer status code (see 2 above for description of
 *    status codes) and a data adaptor instance through which results of the
 *    operation may be accessed.
 *
 * At any point the user provided code may raise an exception if an error
 * occurred. This will be caught and preoprted. An error code will be returned
 * to the simulation indicating to halt in situ processing.
 *
 * The user provided Execute function should call DataAdaptor::ReleaseData when
 * processing is completed to ensure all resources are released.
 */
class SENSEI_EXPORT PythonAnalysis : public AnalysisAdaptor
{
public:
  /// Creates a new instance of the PythonAnalysis
  static PythonAnalysis* New();

  senseiTypeMacro(PythonAnalysis, AnalysisAdaptor);

  /** Set the file to load the Python source code from rank 0 reads and
   * broadcasts to all.
   */
  void SetScriptFile(const std::string &fileName);

  /** Set a module to import Python source code from.  Makes use of Python's
   * import mechanism to load your script. Your script must be in the
   * `PYTHONPATH`.
   */
  void SetScriptModule(const std::string &moduleName);

  /** Set a string containing Python source code that will be executed during
   * initialization. This can be used for instance to set global variables
   * controlling execution.  This source will be executed after loading or
   * importing the script (see SetScriptFile and SetScriptModule) and
   * before the script's functions.
   */
  void SetInitializeSource(const std::string &source);

  /**  Initialize the interpreter. One must set file name or module name before
   * initialization.
   */
  int Initialize();

  /// Invoke in situ processing by calling the user provided Python function.
  bool Execute(DataAdaptor* data, DataAdaptor**) override;

  /// Shut down and clean up the embedded interpreter.
  int Finalize() override;

protected:
  PythonAnalysis();
  ~PythonAnalysis();

  PythonAnalysis(const PythonAnalysis&) = delete;
  void operator=(const PythonAnalysis&) = delete;

private:
  struct InternalsType;
  InternalsType *Internals;
};

}
#endif
