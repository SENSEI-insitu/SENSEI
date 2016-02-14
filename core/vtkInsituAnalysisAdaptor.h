/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkInsituAnalysisAdaptor.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// .NAME vtkInsituAnalysisAdaptor
// .SECTION Description
// vtkInsituAnalysisAdaptor is an adaptor for any insitu analysis framework or
// algorithm. Concrete subclasses use vtkInsituDataAdaptor instance passed to
// the Execute() method to access simulation data for further processing.
#ifndef vtkInsituAnalysisAdaptor_h
#define vtkInsituAnalysisAdaptor_h

#include "vtkObjectBase.h"
#include "vtkSetGet.h"

class vtkInsituDataAdaptor;
class vtkInsituAnalysisAdaptor : public vtkObjectBase
{
public:
  vtkTypeMacro(vtkInsituAnalysisAdaptor, vtkObjectBase);
  void PrintSelf(ostream& os, vtkIndent indent);

  virtual bool Execute(vtkInsituDataAdaptor* data) = 0;

protected:
  vtkInsituAnalysisAdaptor();
  ~vtkInsituAnalysisAdaptor();

private:
  vtkInsituAnalysisAdaptor(const vtkInsituAnalysisAdaptor&); // Not implemented.
  void operator=(const vtkInsituAnalysisAdaptor&); // Not implemented.
};

#endif
