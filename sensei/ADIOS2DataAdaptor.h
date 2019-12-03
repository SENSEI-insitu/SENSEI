#ifndef ADIOS2DataAdaptor_h
#define ADIOS2DataAdaptor_h

#include "InTransitDataAdaptor.h"

#include <adios2_c.h>
#include <mpi.h>
#include <string>

namespace pugi { class xml_node; }

namespace sensei
{

/// The read side of the ADIOS 2 transport layer
class ADIOS2DataAdaptor : public sensei::InTransitDataAdaptor
{
public:
  static ADIOS2DataAdaptor* New();
  senseiTypeMacro(ADIOS2DataAdaptor, sensei::InTransitDataAdaptor);

  int SetFileName(const std::string &fileName);

  int SetReadEngine(const std::string &readEngine);

  /// SENSEI InTransitDataAdaptor control API
  int Initialize(pugi::xml_node &parent) override;
  int Finalize() override;

  int OpenStream() override;
  int CloseStream() override;
  int AdvanceStream() override;
  int StreamGood() override;

  /// SENSEI InTransitDataAdaptor explicit paritioning API
  int GetSenderMeshMetadata(unsigned int id, MeshMetadataPtr &metadata) override;

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
  ADIOS2DataAdaptor();
  ~ADIOS2DataAdaptor();

  // reads the current time step and time values from the stream
  // stores them in the base class information object
  int UpdateTimeStep();

private:
  struct InternalsType;
  InternalsType *Internals;

  ADIOS2DataAdaptor(const ADIOS2DataAdaptor&) = delete;
  void operator=(const ADIOS2DataAdaptor&) = delete;
};

}

#endif
