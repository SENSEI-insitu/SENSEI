#ifndef sensei_VTKmSmartContour_h
#define sensei_VTKmSmartContour_h

#include "senseiConfig.h"
#include "AnalysisAdaptor.h"

#include <string>
#include <vector>
#include <mpi.h>

namespace sensei
{
class DataAdaptor;

/// @class VTKmSmartContour
/// @brief VTKmSmartContour computes contours using a contour tree
///
/// VTKmSmartContour computes contours using a contour tree
///
/// Input Parameters:
/// ScalarField :  Name  of the scalar field
///
/// ScalarFieldAssociation : Scalar field entering, POINT or CELL
///
/// UseMarchingCubes : Use marching cubes connectivity for contour tree
///                    calculations (3D only)
///
/// NumberOfLevels : Number of iso levels to be computed
/// NumberOfComps : Number of components the tree should be simplified to.
///                 For SelectMethod=0 this must be NumberOfLevels+1
///
/// ContourType : Approach to be used to select contours based on the tree.
///               0=saddle+-eps; 1=mid point between saddle and extremum,
///               2=extremum+-eps. (default=0)
///
/// Eps : Error away from the critical point. See ContourType.
///
/// SelectMethod : Method to be used to compute the relevant iso values.
///                Leave at 0 for now, method 1 is not quite right yet.
///
/// CatalystScript : Python script for visualization and/or I/O
///
/// Outputs:
///
/// ContourValues : vector containing the computed contour levels
///
class VTKmSmartContour : public AnalysisAdaptor
{
public:
  static VTKmSmartContour *New();
  senseiTypeMacro(VTKmSmartContour, AnalysisAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  /// @brief Execute the analysis routine.
  ///
  /// This method is called to execute the analysis routine per simulation
  /// iteration.
  bool Execute(DataAdaptor* data) override;

  void SetMeshName(const std::string &name);
  void SetArrayName(const std::string &name);
  void SetArrayCentering(int association);
  void SetUseMarchingCubes(int useMarchinCubes);
  void SetUsePersistenceSorter(int usePersistenceSorter);
  void SetNumberOfLevels(int numberOfLevels);
  void SetContourType(int contourType);
  void SetEps(double eps);
  void SetSelectMethod(int selectMethod);
  void SetNumberOfComps(int numberOfComps);
  void SetCatalystScript(const std::string &catalystScript);
  void SetOutputDir(const std::string &dir);

  /// initialize the analysis after setting desired
  /// control parameters and before the first execute.
  int Initialize();

  /// finalize the analysis
  int Finalize() override;

  /// Get the best values to compute iso contours over
  int GetContourValues(std::vector<double> &vals);

protected:
  VTKmSmartContour();
  ~VTKmSmartContour();

  class InternalsType;
  InternalsType *Internals;

private:
  VTKmSmartContour(const VTKmSmartContour&); // Not implemented.
  void operator=(const VTKmSmartContour&); // Not implemented.
};

}
#endif
