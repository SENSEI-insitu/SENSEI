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

/// Computes a histogram in parallel.
class Histogram : public AnalysisAdaptor
{
public:
  /// allocates a new instance
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

      int NumberOfBins; /// The number of bins in the histogram
      double BinMin;    /// The left most bin edge
      double BinMax;    /// The right most bin edge
      double BinWidth;  /// The width of the equally spaced bins
      std::vector<unsigned int> Histogram; /// The counts of each bin
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
