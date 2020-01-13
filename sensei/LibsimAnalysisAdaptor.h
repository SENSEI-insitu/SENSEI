#ifndef sensei_LibsimAnalysisAdaptor_h
#define sensei_LibsimAnalysisAdaptor_h

#include "AnalysisAdaptor.h"
#include <string>
#include <mpi.h>

namespace sensei
{
/// @cond
class LibsimImageProperties;
/// @endcond

/** An implementation that invokes VisIt libsim. Run time configuration is
 * specified using a VisIt session file (see ::AddRenderer or ::AddExtract).
 */
class LibsimAnalysisAdaptor : public AnalysisAdaptor
{
public:
  /// Allocates an instance of LibsimAnalysisAdaptor.
  static LibsimAnalysisAdaptor* New();

  senseiTypeMacro(LibsimAnalysisAdaptor, AnalysisAdaptor);

  /// @name Run time configuration
  /// @{

  /// Set the file to write libsim debugging info to
  void SetTraceFile(const std::string &traceFile);

  /// Set libsim specific options
  void SetOptions(const std::string &options);

  /** Set the path to the VisIt install. Note that it is recommened to set
   * VisIt spoecific enviornment variables instead of ::SetVisItDirectory.
   */
  void SetVisItDirectory(const std::string &dir);

  /// Set the libsim mode
  void SetMode(const std::string &mode);

  /** If set then the adaptor will generate the nesting information for VisIt.
   * If one wants to do subsetting in VisIt, setting this flag will give VisIt
   * the information it needs to re-generate ghost zones based on the selected
   * blocks. It is off by default because if not doing subsetting this
   * information is not needed, and this information takes up a good deal of
   * space and the algorithm used here is O(N^2) in the number of blocks.
   */
  void SetComputeNesting(int val);

  /** Initialize VisIt and Libsim after setting all desired run time
   * configuration paramters.
   */
  void Initialize();

  /** Explicitly create VisIt plots and rendered output. We recommend
   * using a VisIt session file instead of explictly creating plots.
   *
   * @param[in] frequency sets how many steps in between libsim processing
   * @param[in] session the path to a VisIt generated session file
   * @param[in] plots the name of a VisIt plot (do not use with session)
   * @param[in] plotVars the name of the variable to plot (do not use with session)
   * @param[in] slice if true a slice is taking (do not use with session)
   * @param[in] project2d if true the slice is projected in 2D (do not use with session)
   * @param[in] origin the point defining the slice plane (do not use with session)
   * @param[in] normal the normal of the slice plane (no not use with session)
   */
  bool AddRender(int frequency,
            const std::string &session,
            const std::string &plots,
            const std::string &plotVars,
            bool slice, bool project2d,
            const double origin[3], const double normal[3],
            const LibsimImageProperties &imgProps);

  /** Explicitly create VisIt extract based output. We recommend
   * using a VisIt session file instead of explictly creating plots.
   *
   * @param[in] frequency sets how many steps in between libsim processing
   * @param[in] session the path to a VisIt generated session file
   * @param[in] plots the name of a VisIt plot (do not use with session)
   * @param[in] plotVars the name of the variable to plot (do not use with session)
   * @param[in] slice if true a slice is taking (do not use with session)
   * @param[in] project2d if true the slice is projected in 2D (do not use with session)
   * @param[in] origin the point defining the slice plane (do not use with session)
   * @param[in] normal the normal of the slice plane (no not use with session)
   */
  bool AddExport(int frequency,
                 const std::string &session,
                 const std::string &plot, const std::string &plotVars,
                 bool slice, bool project2d,
                 const double origin[3], const double normal[3],
                 const std::string &filename);
  /// @}

  /// invoke in situ processing with VisIt Libsim
  bool Execute(DataAdaptor* data, DataAdaptor*&) override;

  /// shut down and cleanup VisIt Libsim.
  int Finalize() override;

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
