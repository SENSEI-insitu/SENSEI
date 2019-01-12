#ifndef ADIOS1DataAdaptor_h
#define ADIOS1DataAdaptor_h

#include "DataAdaptor.h"

#include <mpi.h>
#include <adios.h>
#include <adios_read.h>
#include <map>
#include <string>
#include <vtkSmartPointer.h>

namespace sensei
{


/// The read side of the ADIOS 1 transport layer
class ADIOS1DataAdaptor : public DataAdaptor
{
public:
  static ADIOS1DataAdaptor* New();
  senseiTypeMacro(ADIOS1DataAdaptor, ADIOS1DataAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  int Open(const std::string &method, const std::string& filename);
  int Open(ADIOS_READ_METHOD method, const std::string& filename);

  int Advance();

  int Close();

  /// SENSEI DataAdaptor API
  int GetNumberOfMeshes(unsigned int &numMeshes) override;

  int GetMeshMetadata(unsigned int id, MeshMetadataPtr &metadata) override;

  int GetMesh(const std::string &meshName, bool structure_only,
    vtkDataObject *&mesh) override;

  int AddGhostNodesArray(vtkDataObject* mesh, const std::string &meshName) override;
  int AddGhostCellsArray(vtkDataObject* mesh, const std::string &meshName) override;

  int AddArray(vtkDataObject* mesh, const std::string &meshName,
    int association, const std::string &arrayName) override;

  int ReleaseData() override;

protected:
  ADIOS1DataAdaptor();
  ~ADIOS1DataAdaptor();

  // reads the current time step and time values from the stream
  // stores them in the base class information object
  int UpdateTimeStep();

private:
  struct InternalsType;
  InternalsType *Internals;

  ADIOS1DataAdaptor(const ADIOS1DataAdaptor&) = delete;
  void operator=(const ADIOS1DataAdaptor&) = delete;
};

}

#endif
