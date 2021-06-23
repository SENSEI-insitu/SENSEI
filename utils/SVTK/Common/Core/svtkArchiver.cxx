/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkArchiver.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkArchiver.h"

#include <svtkObjectFactory.h>
#include <svtksys/SystemTools.hxx>

#include <fstream>
#include <sstream>

//----------------------------------------------------------------------------
svtkStandardNewMacro(svtkArchiver);

//----------------------------------------------------------------------------
svtkArchiver::svtkArchiver()
{
  this->ArchiveName = nullptr;
}

//----------------------------------------------------------------------------
svtkArchiver::~svtkArchiver()
{
  this->SetArchiveName(nullptr);
}

//----------------------------------------------------------------------------
void svtkArchiver::OpenArchive()
{
  if (this->ArchiveName == nullptr)
  {
    svtkErrorMacro(<< "Please specify ArchiveName to use");
    return;
  }

  if (!svtksys::SystemTools::MakeDirectory(this->ArchiveName))
  {
    svtkErrorMacro(<< "Can not create directory " << this->ArchiveName);
    return;
  }
}

//----------------------------------------------------------------------------
void svtkArchiver::CloseArchive() {}

//----------------------------------------------------------------------------
void svtkArchiver::InsertIntoArchive(
  const std::string& relativePath, const char* data, std::size_t size)
{
  std::stringstream path;
  path << this->ArchiveName << "/" << relativePath;

  svtksys::SystemTools::MakeDirectory(svtksys::SystemTools::GetFilenamePath(path.str()));

  std::ofstream out(path.str().c_str(), std::ios::out | std::ios::binary);
  out.write(data, static_cast<std::streamsize>(size));
  out.close();
}

//----------------------------------------------------------------------------
bool svtkArchiver::Contains(const std::string& relativePath)
{
  std::stringstream path;
  path << this->ArchiveName << "/" << relativePath;

  return svtksys::SystemTools::FileExists(svtksys::SystemTools::GetFilenamePath(path.str()), true);
}

//----------------------------------------------------------------------------
void svtkArchiver::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
