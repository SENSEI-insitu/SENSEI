#include "vtk_histogram.h"

#include <vtkAOSDataArrayTemplate.h>
#include <vtkArrayDispatch.h>
#include <vtkSOADataArrayTemplate.h>

namespace {
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
  const double *Range;
  int Bins;
  std::vector<unsigned int> Histogram;
  HistogramWorker(const double *range, int bins) : Range(range), Bins(bins) {}

  template <typename ArrayT>
  void operator()(ArrayT *array)
  {
    assert(array);
    assert(array->GetNumberOfComponents() == 1);

    typedef typename ArrayT::ValueType ValueType;
    ValueType width = static_cast<ValueType>((this->Range[1] - this->Range[0]) / this->Bins);
    ValueType min = static_cast<ValueType>(this->Range[0]);
    vtkIdType numTuples = array->GetNumberOfTuples();

    this->Histogram.clear();
    // + 1 to store val == max. These will be moved to this last bin before
    // returning (Avoids having to branch in the loop below);
    this->Histogram.resize(this->Bins + 1, 0);
    for (vtkIdType tIdx = 0; tIdx < numTuples; ++tIdx)
      {
      int bin = static_cast<int>((array->GetComponent(tIdx, 0) - min) / width);
      ++this->Histogram[bin];
      }

    // Merge the last two bins (the last is only when val == max)
    this->Histogram[this->Bins-1] += this->Histogram[this->Bins];
    this->Histogram.resize(this->Bins);
  }
};

} // end anon namespace

void vtk_histogram(MPI_Comm comm, vtkDataArray* array, int bins)
{
  int rank;
  MPI_Comm_rank(comm, &rank);

  // find max and min.
  double range[2];
  array->GetRange(range);

  double g_range[2];
  // Find the global max/min
  MPI_Allreduce(&range[0], &g_range[0], 1, MPI_DOUBLE, MPI_MIN, comm);
  MPI_Allreduce(&range[1], &g_range[1], 1, MPI_DOUBLE, MPI_MAX, comm);

  // Compute local histogram
  HistogramWorker worker(g_range, bins);
  // vtkArrayDispatch downcasts the array to a concrete typed implementation
  // that provides faster access methods.
  if (!vtkArrayDispatch::Dispatch::Execute(array, worker))
    {
    // This happens if vtkArrayDispatch doesn't know about the array subclass
    // in use.
    std::cerr << "HistogramWorker dispatch failed on rank " << rank << "!\n";
    worker.Histogram.resize(bins, 0);
    }

  // Global histogram:
  std::vector<unsigned int> gHist(bins, 0);
  MPI_Reduce(&worker.Histogram[0], &gHist[0], bins, MPI_UNSIGNED, MPI_SUM, 0, comm);
  if (rank == 0)
    {
    std::cout << "Histogram '" << array->GetName() << "' (VTK):\n";
    double width = (g_range[1] - g_range[0]) / bins;
    for (int i = 0; i < bins; ++i)
      {
      printf("  %f-%f: %d\n", g_range[0] + i*width, g_range[0] + (i+1)*width, gHist[i]);
      }
    }
}


