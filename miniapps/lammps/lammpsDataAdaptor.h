#pragma once

#include <DataAdaptor.h>
#include <vtkDoubleArray.h>
#include <vtkIntArray.h>

namespace senseiLammps
{

class lammpsDataAdaptor : public sensei::DataAdaptor
{
public:
  static lammpsDataAdaptor* New();
  senseiTypeMacro(lammpsDataAdaptor, sensei::DataAdaptor);

  /// @brief Initialize the data adaptor.
  ///
  /// This initializes the data adaptor. This must be called once per simulation run.
  void Initialize();

  void AddLAMMPSData( long ntimestep, int nlocal, int *id, 
                      int nghost, int *type, double **x, 
                      double xsublo, double xsubhi, 
                      double ysublo, double ysubhi, 
                      double zsublo, double zsubhi);

  void GetBounds ( double &xsublo, double &xsubhi, 
                   double &ysublo, double &ysubhi, 
                   double &zsublo, double &zsubhi);

  void GetN ( int &nlocal, int &nghost );

  void GetPointers ( double **&x, int *&type);

  void GetAtoms ( vtkDoubleArray *&atoms );

  void GetTypes ( vtkIntArray *&types );

  void GetIDs ( vtkIntArray *&ids );

// SENSEI API
  int GetNumberOfMeshes(unsigned int &numMeshes) override;

  int GetMeshMetadata(unsigned int id, sensei::MeshMetadataPtr &md) override;

  //int GetMeshName(unsigned int id, std::string &meshName) override;

  int GetMesh(const std::string &meshName, bool structureOnly,
    vtkDataObject *&mesh) override;

  int AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName) override;

  //int GetNumberOfArrays(const std::string &meshName, int association,
  //  unsigned int &numberOfArrays) override;

  //int GetArrayName(const std::string &meshName, int association,
  //  unsigned int index, std::string &arrayName) override;

  //int GetMeshHasGhostCells(const std::string &meshName, int &nLayers) override;

  int AddGhostCellsArray(vtkDataObject* mesh, const std::string &meshName) override;

  int ReleaseData() override;

protected:
  lammpsDataAdaptor();
  ~lammpsDataAdaptor();

private:
  lammpsDataAdaptor(const lammpsDataAdaptor&); // not implemented.
  void operator=(const lammpsDataAdaptor&); // not implemented.

  struct DInternals;
  DInternals* Internals;
};

}
