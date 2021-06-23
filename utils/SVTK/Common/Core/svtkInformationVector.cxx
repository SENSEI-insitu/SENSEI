/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkInformationVector.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkInformationVector.h"

#include "svtkGarbageCollector.h"
#include "svtkInformation.h"
#include "svtkObjectFactory.h"

#include <vector>

svtkStandardNewMacro(svtkInformationVector);

class svtkInformationVectorInternals
{
public:
  std::vector<svtkInformation*> Vector;

  ~svtkInformationVectorInternals();
};

//----------------------------------------------------------------------------
svtkInformationVectorInternals::~svtkInformationVectorInternals()
{
  // Delete all the information objects.
  for (std::vector<svtkInformation*>::iterator i = this->Vector.begin(); i != this->Vector.end();
       ++i)
  {
    if (svtkInformation* info = *i)
    {
      info->Delete();
    }
  }
}

//----------------------------------------------------------------------------
svtkInformationVector::svtkInformationVector()
{
  this->Internal = new svtkInformationVectorInternals;
  this->NumberOfInformationObjects = 0;
}

//----------------------------------------------------------------------------
svtkInformationVector::~svtkInformationVector()
{
  delete this->Internal;
}

//----------------------------------------------------------------------------
void svtkInformationVector::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Number of Information Objects: " << this->NumberOfInformationObjects << "\n";
  os << indent << "Information Objects:\n";
  for (int i = 0; i < this->NumberOfInformationObjects; ++i)
  {
    svtkInformation* info = this->GetInformationObject(i);
    svtkIndent nextIndent = indent.GetNextIndent();
    os << nextIndent << info->GetClassName() << "(" << info << "):\n";
    info->PrintSelf(os, nextIndent.GetNextIndent());
  }
}

//----------------------------------------------------------------------------
void svtkInformationVector::SetNumberOfInformationObjects(int newNumber)
{
  // Adjust the number of objects.
  int oldNumber = this->NumberOfInformationObjects;
  if (newNumber > oldNumber)
  {
    // Create new information objects.
    this->Internal->Vector.resize(newNumber, nullptr);
    for (int i = oldNumber; i < newNumber; ++i)
    {
      this->Internal->Vector[i] = svtkInformation::New();
    }
    this->NumberOfInformationObjects = newNumber;
  }
  else if (newNumber < oldNumber)
  {
    // Delete old information objects.
    for (int i = newNumber; i < oldNumber; ++i)
    {
      if (svtkInformation* info = this->Internal->Vector[i])
      {
        // Set the pointer to nullptr first to avoid reporting of the
        // entry if deleting the information object causes a garbage
        // collection reference walk.
        this->Internal->Vector[i] = nullptr;
        info->Delete();
      }
    }
    this->Internal->Vector.resize(newNumber);
    this->NumberOfInformationObjects = newNumber;
  }
}

//----------------------------------------------------------------------------
void svtkInformationVector::SetInformationObject(int index, svtkInformation* newInfo)
{
  if (newInfo && index >= 0 && index < this->NumberOfInformationObjects)
  {
    // Replace an existing information object.
    svtkInformation* oldInfo = this->Internal->Vector[index];
    if (oldInfo != newInfo)
    {
      newInfo->Register(this);
      this->Internal->Vector[index] = newInfo;
      oldInfo->UnRegister(this);
    }
  }
  else if (newInfo && index >= this->NumberOfInformationObjects)
  {
    // If a hole will be created fill it with empty objects.
    if (index > this->NumberOfInformationObjects)
    {
      this->SetNumberOfInformationObjects(index);
    }

    // Store the information object in a new entry.
    newInfo->Register(this);
    this->Internal->Vector.push_back(newInfo);
    this->NumberOfInformationObjects++;
  }
  else if (!newInfo && index >= 0 && index < this->NumberOfInformationObjects - 1)
  {
    // We do not allow nullptr information objects.  Create an empty one
    // to fill in the hole.
    svtkInformation* oldInfo = this->Internal->Vector[index];
    this->Internal->Vector[index] = svtkInformation::New();
    oldInfo->UnRegister(this);
  }
  else if (!newInfo && index >= 0 && index == this->NumberOfInformationObjects - 1)
  {
    // Remove the last information object.
    this->SetNumberOfInformationObjects(index);
  }
}

//----------------------------------------------------------------------------
svtkInformation* svtkInformationVector::GetInformationObject(int index)
{
  if (index >= 0 && index < this->NumberOfInformationObjects)
  {
    return this->Internal->Vector[index];
  }
  return nullptr;
}

//----------------------------------------------------------------------------
void svtkInformationVector::Append(svtkInformation* info)
{
  // Setting an entry beyond the end will automatically append.
  this->SetInformationObject(this->NumberOfInformationObjects, info);
}

//----------------------------------------------------------------------------
void svtkInformationVector::Remove(svtkInformation* info)
{
  // Search for the information object and remove it.
  for (int i = 0; i < this->NumberOfInformationObjects; ++i)
  {
    if (this->Internal->Vector[i] == info)
    {
      this->Internal->Vector.erase(this->Internal->Vector.begin() + i);
      info->UnRegister(this);
      this->NumberOfInformationObjects--;
    }
  }
}

//----------------------------------------------------------------------------
void svtkInformationVector::Remove(int i)
{
  if (i < this->NumberOfInformationObjects)
  {
    if (this->Internal->Vector[i])
    {
      this->Internal->Vector[i]->UnRegister(this);
    }
    this->Internal->Vector.erase(this->Internal->Vector.begin() + i);
    this->NumberOfInformationObjects--;
  }
}

//----------------------------------------------------------------------------
void svtkInformationVector::Copy(svtkInformationVector* from, int deep)
{
  // if deep we can reuse existing info objects
  if (deep)
  {
    this->SetNumberOfInformationObjects(from->GetNumberOfInformationObjects());
    for (int i = 0; i < from->GetNumberOfInformationObjects(); ++i)
    {
      this->Internal->Vector[i]->Copy(from->GetInformationObject(i), deep);
    }
    return;
  }

  // otherwise it is a shallow copy and we must copy pointers
  this->SetNumberOfInformationObjects(0);
  // copy the data
  for (int i = 0; i < from->GetNumberOfInformationObjects(); ++i)
  {
    svtkInformation* fromI = from->GetInformationObject(i);
    this->SetInformationObject(i, fromI);
  }
}

//----------------------------------------------------------------------------
void svtkInformationVector::Register(svtkObjectBase* o)
{
  this->RegisterInternal(o, 1);
}

//----------------------------------------------------------------------------
void svtkInformationVector::UnRegister(svtkObjectBase* o)
{
  this->UnRegisterInternal(o, 1);
}

//----------------------------------------------------------------------------
void svtkInformationVector::ReportReferences(svtkGarbageCollector* collector)
{
  this->Superclass::ReportReferences(collector);
  for (int i = 0; i < this->NumberOfInformationObjects; ++i)
  {
    svtkGarbageCollectorReport(collector, this->Internal->Vector[i], "Entry");
  }
}
