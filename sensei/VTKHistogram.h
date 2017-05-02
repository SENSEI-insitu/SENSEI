#ifndef sensei_VTKHistogram_h
#define sensei_VTKHistogram_h

class vtkUnsignedCharArray;
class vtkDataArray;

#include <mpi.h>
#include <string>

namespace sensei
{

class VTKHistogram
{
public:
    VTKHistogram();
    ~VTKHistogram();
    void AddRange(vtkDataArray* da, vtkUnsignedCharArray* ghostArray);
    void PreCompute(MPI_Comm comm, int bins);
    void Compute(vtkDataArray* da, vtkUnsignedCharArray* ghostArray);
    void PostCompute(MPI_Comm comm, int bins, const std::string& name);
private:
  double Range[2];
  struct Internals;
  Internals *Worker;
};

}

#endif
