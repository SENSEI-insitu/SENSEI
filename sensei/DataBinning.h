#ifndef DataBinning_h
#define DataBinning_h

#include "AnalysisAdaptor.h"
#include <mpi.h>
#include <vector>
#include <thread>
#include <future>
#include <list>

/// @cond
namespace pugi { class xml_node; }
/// @endcond

namespace sensei
{
/** Bins a set of arrays onto a user defined uniform Cartesian mesh.  The
 * binning operations supported are: min, max, and sum.  A count is always
 * computed. From count and sum bin averages can be obtained.  At each time
 * step the results are stored on disk in VTK format.  Binning can be used to
 * compute phase space plots or visualize particle data using mesh based
 * visualization techniques.
 */
class SENSEI_EXPORT DataBinning : public AnalysisAdaptor
{
public:
  /// allocates a new instance
  static DataBinning* New();
  senseiTypeMacro(DataBinning, AnalysisAdaptor);

  /// When set binning runs in a thread and ::Execute returns immediately
  void SetAsynchronous(int val) override;

  /** set the name of the mesh and arrays to request from the simulation.
   * Sets the output image resolution and the name of a file to write them in.
   * the substring %t% is replaced with the current simulation time step. if
   * one of resx or resy is -1 it is calculated from the data's aspect ratio
   * so that image pixels are square.
   */
  int Initialize(const std::string &meshName,
    const std::string &xAxisArray,
    const std::string &yAxisArray,
    const std::vector<std::string> &binnedArray,
    const std::vector<std::string> &operation,
    long xres, long yres, const std::string &outDir,
    int returnData, int maxThreads);

  /** Initialize from XML
   * Input mesh, coordinate axis arrays, grid resolution, and output path are
   * specified as attributes.
   *
   * The supported attributes are:
   * | attribute | description |
   * | --------- | ----------- |
   * | mesh | names the mesh to process |
   * | x_axis | names the array to use as x-axis coordinates |
   * | y_axis  | names the array to use as y-axis coordinates |
   * | out_dir | the path to write results to |
   * | x_res | the grid resolution to bin to in the x-direction |
   * | y_res | the grid resolution to bin to in the y-direction |
   * | return_data | set to 1 to return a data adaptor w/ the results |
   * | max_threads | the max number of threads to use |
   *
   * Nested elements:
   * | element | description |
   * | arrays | a list of comma or space delimited array names to bin |
   * | operations |  a list of comma or space delimited binning operations |
   *
   * The arrays to bin and binning operations are specified in a nested
   * elements with a space or comma delimited lists. There must be an operation
   * specified for each array.
   */
  int Initialize(pugi::xml_node node);

  /// compute the histogram for this time step
  bool Execute(DataAdaptor* data, DataAdaptor**) override;

  /// finalize the run
  int Finalize() override;

  /// The supported binning operations
  enum {INVALID_OP, BIN_SUM, BIN_AVG, BIN_MIN, BIN_MAX};

  /** converts the string to the corresponding enumeration.
   * @param[in] opName string naming the operation(one of: "sum", "avg", "min", "max")
   * @param[out] opCode the enumerated value
   * @returns zero if successful
   */
  static int GetOperation(const std::string &opName, int &opCode);

  /** converts the enumeration to a string
   * @param[in] opCode the enumerated value(one of: BIN_SUM, BIN_AVG, BIN_MIN, BIN_MAX)
   * @param[out] opName string naming the operation
   * @returns zero if successful
   */
  static int GetOperation(int opCode, std::string &opName);

protected:
  DataBinning();
  ~DataBinning();

  DataBinning(const DataBinning&) = delete;
  void operator=(const DataBinning&) = delete;

  void InitializeThreads();
  int WaitThreads();

  long XRes;
  long YRes;
  std::string OutDir;
  unsigned long Iteration;
  double AxisFactor;
  std::string MeshName;
  std::string XAxisArray;
  std::string YAxisArray;
  std::vector<std::string> BinnedArray;
  std::vector<int> Operation;
  int ReturnData;
  int MaxThreads;
  std::vector<std::future<int>> Threads;
  std::vector<MPI_Comm> ThreadComm;
};

}

#endif
