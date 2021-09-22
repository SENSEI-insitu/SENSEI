#ifndef Histogram_h
#define Histogram_h

#include "AnalysisAdaptor.h"
#include <mpi.h>
#include <vector>

class vtkDataObject;
class vtkDataArray;

namespace sensei
{

class HistogramInternals;

/// @class Histogram
/// @brief Computes a parallel histogram
class Histogram : public AnalysisAdaptor
{
public:
  static Histogram* New();
  senseiTypeMacro(Histogram, AnalysisAdaptor);

  /// initialize for the run
  void Initialize(int bins, const std::string &meshName,
    int association, const std::string& arrayName,
    const std::string &fileName);

  /// compute the histogram for this time step
  bool Execute(DataAdaptor* data) override;

  /// finalize the run
  int Finalize() override;

  /// the computed hostgram may be accessed through the following data structure.
  struct Data
  {
      Data() : NumberOfBins(1), BinMin(1.0), BinMax(0.0), BinWidth(1.0), Histogram() {}

      int NumberOfBins;
      double BinMin;
      double BinMax;
      double BinWidth;
      std::vector<unsigned int> Histogram;
  };

  /// return the histogram computed by the most recent call to ::Execute
  int GetHistogram(Histogram::Data &data);

protected:
  Histogram();
  ~Histogram();

  Histogram(const Histogram&) = delete;
  void operator=(const Histogram&) = delete;

  static const char *GetGhostArrayName();
  vtkDataArray* GetArray(vtkDataObject* dobj, const std::string& arrayname);

  int NumberOfBins;
  std::string MeshName;
  std::string ArrayName;
  int Association;
  std::string FileName;
  Histogram::Data LastResult;
};

}

#endif
