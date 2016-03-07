#ifndef sensei_adios_DataAdaptor_h
#define sensei_adios_DataAdaptor_h

#include <sensei/DataAdaptor.h>

#include <mpi.h>
#include <adios.h>
#include <adios_read.h>
#include <map>
#include <string>
#include <vtkSmartPointer.h>

class vtkDataArray;
class vtkImageData;

namespace sensei
{
namespace adios
{

class DataAdaptor : public sensei::DataAdaptor
{
public:
  static DataAdaptor* New();
  vtkTypeMacro(DataAdaptor, sensei::DataAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent);

  bool Open(MPI_Comm comm, ADIOS_READ_METHOD method, const std::string& filename);
  bool Advance();
  bool ReadStep();

  virtual vtkDataObject* GetMesh(bool structure_only=false);
  virtual bool AddArray(vtkDataObject* mesh, int association, const std::string& arrayname);
  virtual unsigned int GetNumberOfArrays(int association);
  virtual std::string GetArrayName(int association, unsigned int index);
  virtual void ReleaseData();

//BTX
protected:
  DataAdaptor();
  ~DataAdaptor();
  ADIOS_FILE* File;
  MPI_Comm Comm;

  typedef std::map<std::string, bool > ArraysType;
  typedef std::map<int, ArraysType> AssociatedArraysType;
  AssociatedArraysType AssociatedArrays;

  vtkSmartPointer<vtkDataObject>  Mesh;

private:
  DataAdaptor(const DataAdaptor&); // Not implemented.
  void operator=(const DataAdaptor&); // Not implemented.
//ETX
};

} // adios
} // sensei
#endif
