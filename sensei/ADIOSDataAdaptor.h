#ifndef ADIOSDataAdaptor_h
#define ADIOSDataAdaptor_h

#include "DataAdaptor.h"

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

class ADIOSDataAdaptor : public DataAdaptor
{
public:
  static ADIOSDataAdaptor* New();
  senseiTypeMacro(ADIOSDataAdaptor, ADIOSDataAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent) override;

  bool Open(MPI_Comm comm, ADIOS_READ_METHOD method, const std::string& filename);
  bool Advance();
  bool ReadStep();

  vtkDataObject* GetMesh(bool structure_only=false) override;
  bool AddArray(vtkDataObject* mesh, int association, const std::string& arrayname) override;
  unsigned int GetNumberOfArrays(int association) override;
  std::string GetArrayName(int association, unsigned int index) override;
  void ReleaseData() override;

protected:
  ADIOSDataAdaptor();
  ~ADIOSDataAdaptor();
  ADIOS_FILE* File;
  MPI_Comm Comm;

  typedef std::map<std::string, bool > ArraysType;
  typedef std::map<int, ArraysType> AssociatedArraysType;
  AssociatedArraysType AssociatedArrays;

  vtkSmartPointer<vtkDataObject>  Mesh;

private:
  ADIOSDataAdaptor(const ADIOSDataAdaptor&); // Not implemented.
  void operator=(const ADIOSDataAdaptor&); // Not implemented.
};

}

#endif
