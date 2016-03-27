#include <vtkDataArrayDispatcher.h>
#include <vtkDataArrayTemplate.h>
#include <vtkSmartPointer.h>

#include "DataAdaptor.h"

#include <timer/Timer.h>

#include <algorithm>
#include <vector>
namespace
{

// Private worker for histogram method. Computes the local histogram on
// array (passed to operator()). To be used with vtkArrayDispatch.
//
// Inputs:
// range: Global range of data
// bins: Number of histogram bins
// array: Local data.
//
// Outputs:
// Histogram: The histogram of the local data.
struct HistogramWorker
{
  vtkUnsignedCharArray* GhostArray;
  const double *Range;
  int Bins;
  std::vector<unsigned int> Histogram;
  HistogramWorker(const double *range, int bins) : GhostArray(NULL), Range(range), Bins(bins) {}

  template <typename T>
  void operator()(const vtkDataArrayDispatcherPointer<T>& array)
  {
    assert(array.NumberOfComponents == 1);

    typedef T ValueType;
    ValueType width = static_cast<ValueType>((this->Range[1] - this->Range[0]) / this->Bins);
    ValueType min = static_cast<ValueType>(this->Range[0]);
    vtkIdType numTuples = array.NumberOfTuples;

    // + 1 to store val == max. These will be moved to this last bin before
    // returning (Avoids having to branch in the loop below);
    this->Histogram.resize(this->Bins + 1, 0);
    if (this->GhostArray)
      {
      for (vtkIdType tIdx = 0; tIdx < numTuples; ++tIdx)
        {
        if (this->GhostArray->GetValue(tIdx) == 0)
          {
          int bin = static_cast<int>((array.RawPointer[tIdx] - min) / width);
          ++this->Histogram[bin];
          }
        }
      }
    else
      {
      for (vtkIdType tIdx = 0; tIdx < numTuples; ++tIdx)
        {
        int bin = static_cast<int>((array.RawPointer[tIdx] - min) / width);
        ++this->Histogram[bin];
        }
      }

    // Merge the last two bins (the last is only when val == max)
    this->Histogram[this->Bins-1] += this->Histogram[this->Bins];
    this->Histogram.resize(this->Bins);
  }
};

// Compute array range by skipping ghost elements.
class ComponentRangeWorker
{
  double Range[2];
  vtkUnsignedCharArray* GhostArray;
public:
  ComponentRangeWorker(vtkUnsignedCharArray* ghostArray) : GhostArray(ghostArray)
    {
    this->Range[0] = vtkTypeTraits<double>::Max();
    this->Range[1] = vtkTypeTraits<double>::Min();
    }

  template <typename T>
  void operator()(const vtkDataArrayDispatcherPointer<T>& array)
    {
    assert(array.NumberOfComponents == 1);
    typedef T ValueType;
    vtkIdType numTuples = array.NumberOfTuples;
    for (vtkIdType cc=0; cc < numTuples; cc++)
      {
      if (this->GhostArray->GetValue(cc) == 0)
        {
        this->Range[0] = std::min(this->Range[0], static_cast<double>(array.RawPointer[cc]));
        this->Range[1] = std::max(this->Range[1], static_cast<double>(array.RawPointer[cc]));
        }
      }
    }

  void GetRange(double r[2])
    {
    std::copy(this->Range, this->Range+2, r);
    }
};

class vtkHistogram
{
  double Range[2];
  HistogramWorker *Worker;
public:
    vtkHistogram()
      {
      this->Range[0] = vtkTypeTraits<double>::Max();
      this->Range[1] = vtkTypeTraits<double>::Min();
      this->Worker = NULL;
      }
    ~vtkHistogram()
      {
      delete this->Worker;
      }

    void AddRange(vtkDataArray* da, vtkUnsignedCharArray* ghostArray)
      {
      double crange[2] = { vtkTypeTraits<double>::Max(), vtkTypeTraits<double>::Min() };
      if (da && ghostArray)
        {
        ComponentRangeWorker worker(ghostArray);
        vtkDataArrayDispatcher<ComponentRangeWorker> dispatcher(worker);
        dispatcher.Go(da);
        worker.GetRange(crange);
        }
      else if (da)
        {
        da->GetRange(crange);
        }
      this->Range[0] = std::min(this->Range[0], crange[0]);
      this->Range[1] = std::max(this->Range[1], crange[1]);
      }
    void PreCompute(MPI_Comm comm, int bins)
      {
      double g_range[2];
      // Find the global max/min
      MPI_Allreduce(&this->Range[0], &g_range[0], 1, MPI_DOUBLE, MPI_MIN, comm);
      MPI_Allreduce(&this->Range[1], &g_range[1], 1, MPI_DOUBLE, MPI_MAX, comm);
      this->Range[0] = g_range[0];
      this->Range[1] = g_range[1];
      this->Worker = new HistogramWorker(this->Range, bins);
      }

    void Compute(vtkDataArray* da, vtkUnsignedCharArray* ghostArray)
      {
      if (da)
        {
        this->Worker->GhostArray = ghostArray;
        vtkDataArrayDispatcher<HistogramWorker> dispatcher(*this->Worker);
        dispatcher.Go(da);
        this->Worker->GhostArray = NULL;
        }
      }

    void PostCompute(MPI_Comm comm, int bins, const std::string& name)
      {
      std::vector<unsigned int> gHist(bins, 0);
      MPI_Reduce(&this->Worker->Histogram[0], &gHist[0], bins, MPI_UNSIGNED, MPI_SUM, 0, comm);

      int rank;
      MPI_Comm_rank(comm, &rank);
      if (rank == 0)
        {
        std::cout << "Histogram '" << name << "' (VTK):\n";
        double width = (this->Range[1] - this->Range[0]) / bins;
        for (int i = 0; i < bins; ++i)
          {
          printf("  %f-%f: %d\n", this->Range[0] + i*width, this->Range[0] + (i+1)*width, gHist[i]);
          }
        }
      }
};

}
