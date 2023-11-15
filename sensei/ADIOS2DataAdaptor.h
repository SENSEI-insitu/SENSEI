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
class SENSEI_EXPORT ADIOS2DataAdaptor : public sensei::InTransitDataAdaptor
{
public:
  static ADIOS2DataAdaptor* New();
  senseiTypeMacro(ADIOS2DataAdaptor, sensei::InTransitDataAdaptor);

  /// @brief Set the filename.
  /// Default value is "sensei.bp" which is suitable for use with streams or
  /// transport engines such as SST. When reading file series from disk using
  /// engines such as BP4 one should include a integer printf format specifier,
  /// for example "sensei_%04d.bp".
  void SetFileName(const std::string &fileName);

  // set te adios engine to use. this will be the same engine
  // given to the write side analysis adaptor
  void SetReadEngine(const std::string &readEngine);

  // add name value pairs to pass into ADIOS after the
  // engine has been created
  void AddParameter(const std::string &name, const std::string &value);

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
    svtkDataObject *&mesh) override;

  int AddGhostNodesArray(svtkDataObject* mesh, const std::string &meshName) override;
  int AddGhostCellsArray(svtkDataObject* mesh, const std::string &meshName) override;

  int AddArray(svtkDataObject* mesh, const std::string &meshName,
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
