#ifndef vtkADIOSDataAdaptor_h
#define vtkADIOSDataAdaptor_h

#include "vtkInsituDataAdaptor.h"
#include <mpi.h>
#include <adios.h>
#include <adios_read.h>
#include <map>
#include <string>
#include <vtkSmartPointer.h>

class vtkDataArray;
class vtkImageData;
class vtkADIOSDataAdaptor : public vtkInsituDataAdaptor
{
public:
  static vtkADIOSDataAdaptor* New();
  vtkTypeMacro(vtkADIOSDataAdaptor, vtkInsituDataAdaptor);
  void PrintSelf(ostream& os, vtkIndent indent);

  bool Open(MPI_Comm comm, ADIOS_READ_METHOD method, const std::string& filename);
  bool Advance();
  bool ReadStep();

  virtual vtkDataObject* GetMesh(bool structure_only=false);
  virtual bool AddArray(vtkDataObject* mesh, int association, const char* arrayname);
  virtual unsigned int GetNumberOfArrays(int association);
  virtual const char* GetArrayName(int association, unsigned int index);
  virtual void ReleaseData();

//BTX
protected:
  vtkADIOSDataAdaptor();
  ~vtkADIOSDataAdaptor();
  ADIOS_FILE* File;
  MPI_Comm Comm;

  typedef std::map<std::string, bool > ArraysType;
  typedef std::map<int, ArraysType> AssociatedArraysType;
  AssociatedArraysType AssociatedArrays;

  vtkSmartPointer<vtkDataObject>  Mesh;

private:
  vtkADIOSDataAdaptor(const vtkADIOSDataAdaptor&); // Not implemented.
  void operator=(const vtkADIOSDataAdaptor&); // Not implemented.
//ETX
};

#endif
