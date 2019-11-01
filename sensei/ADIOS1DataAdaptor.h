#ifndef ADIOS1DataAdaptor_h
#define ADIOS1DataAdaptor_h

#include "InTransitDataAdaptor.h"

#include <adios.h>
#include <adios_read.h>
#include <mpi.h>
#include <map>
#include <string>

namespace pugi { class xml_node; }

namespace sensei
{

/// The read side of the ADIOS 1 transport layer
class ADIOS1DataAdaptor : public sensei::InTransitDataAdaptor
{
public:
  static ADIOS1DataAdaptor* New();
  senseiTypeMacro(ADIOS1DataAdaptor, sensei::InTransitDataAdaptor);

  int SetFileName(const std::string &fileName);

  int SetReadMethod(const std::string &readMethod);
  int SetReadMethod(ADIOS_READ_METHOD readMethod);

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
