/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkInsituDataAdaptor.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkInsituDataAdaptor
// .SECTION Description
// vtkInsituDataAdaptor is an adaptor that is used to get vtkDataObject and
// vtkAbstractArray. Each simulation code provides an implementation of
// vtkInsituDataAdaptor which maps simulation data-structures to VTK data model
// on request.

#ifndef vtkInsituDataAdaptor_h
#define vtkInsituDataAdaptor_h

#include "vtkObjectBase.h"
#include "vtkSetGet.h"

class vtkInformation;
class vtkDataObject;
class vtkAbstractArray;
class vtkInsituDataAdaptor : public vtkObjectBase
{
public:
  vtkTypeMacro(vtkInsituDataAdaptor, vtkObjectBase);
  void PrintSelf(ostream& os, vtkIndent indent);

  // Description:
  // Provides access to meta-data about the current data.
  // This can provide all metadata necessary, including global extents, fields
  // available, etc.
  vtkGetObjectMacro(Information, vtkInformation);

  // Description:
  // Convenience method to get the time information.
  double GetDataTime() { return this->GetDataTime(this->Information); }
  static double GetDataTime(vtkInformation*);

  // Description:
  // Convenience methods to set the time information.
  void SetDataTime(double time) { this->SetDataTime(this->Information, time); }
  static void SetDataTime(vtkInformation*, double time);

  // Description:
  // Subclasses should override this method to generate the mesh.
  virtual vtkDataObject* GetMesh()=0;

  // Description:
  // Subclasses should override this method to provide field arrays.
  virtual vtkAbstractArray* GetArray(int association, const char* name) = 0;

  // Description:
  // API to inquire about arrays.
  virtual unsigned int GetNumberOfArrays(int association) = 0;
  virtual const char* GetArrayName(int association, unsigned int index) = 0;

  // Description:
  // Method called to release data and end of each execution iteration.
  virtual void ReleaseData()=0;
protected:
  vtkInsituDataAdaptor();
  ~vtkInsituDataAdaptor();

  vtkInformation* Information;
private:
  vtkInsituDataAdaptor(const vtkInsituDataAdaptor&); // Not implemented.
  void operator=(const vtkInsituDataAdaptor&); // Not implemented.
};

#endif
