#ifndef MANDELBROT_DATAADAPTOR_H
#define MANDELBROT_DATAADAPTOR_H

#include <sensei/DataAdaptor.h>
#include "simulation_data.h"

class vtkDataArray;

class MandelbrotDataAdaptor : public sensei::DataAdaptor
{
public:
  static MandelbrotDataAdaptor* New();
  senseiTypeMacro(MandelbrotDataAdaptor, sensei::DataAdaptor);

  /// @brief Initialize the data adaptor.
  ///
  /// This initializes the data adaptor. This must be called once per simulation run.
  /// @param sim is a pointer to the simulation's main data structure.
  void Initialize(simulation_data *sim);

  // SENSEI API
  int GetNumberOfMeshes(unsigned int &numMeshes) override;

  int GetMeshName(unsigned int id, std::string &meshName) override;

  int GetMesh(const std::string &meshName, bool structureOnly,
    vtkDataObject *&mesh) override;

  int AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName) override;

  int GetNumberOfArrays(const std::string &meshName, int association,
    unsigned int &numberOfArrays) override;

  int GetArrayName(const std::string &meshName, int association,
    unsigned int index, std::string &arrayName) override;

  int ReleaseData() override;

protected:
  MandelbrotDataAdaptor();
  ~MandelbrotDataAdaptor();

private:
  MandelbrotDataAdaptor(const MandelbrotDataAdaptor&); // not implemented.
  void operator=(const MandelbrotDataAdaptor&); // not implemented.

  struct DInternals;
  DInternals* Internals;
};

#endif
