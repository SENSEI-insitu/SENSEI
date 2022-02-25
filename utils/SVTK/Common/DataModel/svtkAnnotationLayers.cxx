/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAnnotationLayers.cxx

-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkAnnotationLayers.h"

#include "svtkAnnotation.h"
#include "svtkIdTypeArray.h"
#include "svtkInformation.h"
#include "svtkInformationVector.h"
#include "svtkObjectFactory.h"
#include "svtkSelection.h"
#include "svtkSelectionNode.h"
#include "svtkSmartPointer.h"

#include <algorithm>
#include <vector>

svtkStandardNewMacro(svtkAnnotationLayers);
svtkCxxSetObjectMacro(svtkAnnotationLayers, CurrentAnnotation, svtkAnnotation);

class svtkAnnotationLayers::Internals
{
public:
  std::vector<svtkSmartPointer<svtkAnnotation> > Annotations;
};

svtkAnnotationLayers::svtkAnnotationLayers()
  : Implementation(new Internals())
{
  this->CurrentAnnotation = svtkAnnotation::New();

  // Start with an empty index selection
  svtkSmartPointer<svtkSelection> sel = svtkSmartPointer<svtkSelection>::New();
  svtkSmartPointer<svtkSelectionNode> node = svtkSmartPointer<svtkSelectionNode>::New();
  node->SetContentType(svtkSelectionNode::INDICES);
  svtkSmartPointer<svtkIdTypeArray> ids = svtkSmartPointer<svtkIdTypeArray>::New();
  node->SetSelectionList(ids);
  sel->AddNode(node);
  this->CurrentAnnotation->SetSelection(sel);
}

svtkAnnotationLayers::~svtkAnnotationLayers()
{
  delete this->Implementation;
  if (this->CurrentAnnotation)
  {
    this->CurrentAnnotation->Delete();
  }
}

void svtkAnnotationLayers::SetCurrentSelection(svtkSelection* sel)
{
  if (this->CurrentAnnotation)
  {
    this->CurrentAnnotation->SetSelection(sel);
    this->Modified();
  }
}

svtkSelection* svtkAnnotationLayers::GetCurrentSelection()
{
  if (this->CurrentAnnotation)
  {
    return this->CurrentAnnotation->GetSelection();
  }
  return nullptr;
}

unsigned int svtkAnnotationLayers::GetNumberOfAnnotations()
{
  return static_cast<unsigned int>(this->Implementation->Annotations.size());
}

svtkAnnotation* svtkAnnotationLayers::GetAnnotation(unsigned int idx)
{
  if (idx >= this->Implementation->Annotations.size())
  {
    return nullptr;
  }
  return this->Implementation->Annotations[idx];
}

void svtkAnnotationLayers::AddAnnotation(svtkAnnotation* annotation)
{
  this->Implementation->Annotations.push_back(annotation);
  this->Modified();
}

void svtkAnnotationLayers::RemoveAnnotation(svtkAnnotation* annotation)
{
  this->Implementation->Annotations.erase(std::remove(this->Implementation->Annotations.begin(),
                                            this->Implementation->Annotations.end(), annotation),
    this->Implementation->Annotations.end());
  this->Modified();
}

void svtkAnnotationLayers::Initialize()
{
  this->Implementation->Annotations.clear();
  this->Modified();
}

void svtkAnnotationLayers::ShallowCopy(svtkDataObject* other)
{
  this->Superclass::ShallowCopy(other);
  svtkAnnotationLayers* obj = svtkAnnotationLayers::SafeDownCast(other);
  if (!obj)
  {
    return;
  }
  this->Implementation->Annotations.clear();
  for (unsigned int a = 0; a < obj->GetNumberOfAnnotations(); ++a)
  {
    svtkAnnotation* ann = obj->GetAnnotation(a);
    this->AddAnnotation(ann);
  }
  this->SetCurrentAnnotation(obj->GetCurrentAnnotation());
}

void svtkAnnotationLayers::DeepCopy(svtkDataObject* other)
{
  this->Superclass::DeepCopy(other);
  svtkAnnotationLayers* obj = svtkAnnotationLayers::SafeDownCast(other);
  if (!obj)
  {
    return;
  }
  this->Implementation->Annotations.clear();
  for (unsigned int a = 0; a < obj->GetNumberOfAnnotations(); ++a)
  {
    svtkSmartPointer<svtkAnnotation> ann = svtkSmartPointer<svtkAnnotation>::New();
    ann->DeepCopy(obj->GetAnnotation(a));
    this->AddAnnotation(ann);
  }
}

svtkMTimeType svtkAnnotationLayers::GetMTime()
{
  svtkMTimeType mtime = this->Superclass::GetMTime();
  for (unsigned int a = 0; a < this->GetNumberOfAnnotations(); ++a)
  {
    svtkAnnotation* ann = this->GetAnnotation(a);
    if (ann)
    {
      svtkMTimeType atime = ann->GetMTime();
      if (atime > mtime)
      {
        mtime = atime;
      }
    }
  }
  svtkAnnotation* s = this->GetCurrentAnnotation();
  if (s)
  {
    svtkMTimeType stime = this->GetCurrentAnnotation()->GetMTime();
    if (stime > mtime)
    {
      mtime = stime;
    }
  }
  return mtime;
}

void svtkAnnotationLayers::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);

  svtkIndent next = indent.GetNextIndent();
  for (unsigned int a = 0; a < this->GetNumberOfAnnotations(); ++a)
  {
    os << next << "Annotation " << a << ":";
    svtkAnnotation* ann = this->GetAnnotation(a);
    if (ann)
    {
      os << "\n";
      ann->PrintSelf(os, next.GetNextIndent());
    }
    else
    {
      os << "(none)\n";
    }
  }
  os << indent << "CurrentAnnotation: ";
  if (this->CurrentAnnotation)
  {
    os << "\n";
    this->CurrentAnnotation->PrintSelf(os, indent.GetNextIndent());
  }
  else
  {
    os << "(none)\n";
  }
}

svtkAnnotationLayers* svtkAnnotationLayers::GetData(svtkInformation* info)
{
  return info ? svtkAnnotationLayers::SafeDownCast(info->Get(DATA_OBJECT())) : nullptr;
}

svtkAnnotationLayers* svtkAnnotationLayers::GetData(svtkInformationVector* v, int i)
{
  return svtkAnnotationLayers::GetData(v->GetInformationObject(i));
}
