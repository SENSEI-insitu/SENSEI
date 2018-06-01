#ifndef sensei_Histogram_h
#define sensei_Histogram_h

#include "AnalysisAdaptor.h"
#include <mpi.h>
#include <vector>

class vtkDataObject;
class vtkDataArray;

namespace sensei
{

class VTKHistogram;

/// @class Histogram
/// @brief Computes a parallel histogram
class Histogram : public AnalysisAdaptor
{
public:
  static Histogram* New();
  senseiTypeMacro(Histogram, AnalysisAdaptor);

  void Initialize(int bins, const std::string &meshName,
    int association, const std::string& arrayname);

  bool Execute(DataAdaptor* data) override;

  int Finalize() override;

  // return the last computed histogram
  int GetHistogram(double &min, double &max,
    std::vector<unsigned int> &bins);

protected:
  Histogram();
  ~Histogram();

  Histogram(const Histogram&) = delete;
  void operator=(const Histogram&) = delete;

  static const char *GetGhostArrayName();
  vtkDataArray* GetArray(vtkDataObject* dobj, const std::string& arrayname);

  int Bins;
  std::string MeshName;
  std::string ArrayName;
  int Association;

  VTKHistogram *Internals;

};

}

#endif
