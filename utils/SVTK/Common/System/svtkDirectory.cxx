/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkDirectory.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkDirectory.h"
#include "svtkStringArray.h"

#include "svtkDebugLeaks.h"
#include "svtkObjectFactory.h"

#include <svtksys/SystemTools.hxx>

svtkStandardNewMacro(svtkDirectory);

svtkDirectory::svtkDirectory()
  : Path(nullptr)
{
  this->Files = svtkStringArray::New();
}

void svtkDirectory::CleanUpFilesAndPath()
{
  this->Files->Reset();
  delete[] this->Path;
  this->Path = nullptr;
}

svtkDirectory::~svtkDirectory()
{
  this->CleanUpFilesAndPath();
  this->Files->Delete();
  this->Files = nullptr;
}

void svtkDirectory::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "Files:  (" << this->Files << ")\n";
  if (!this->Path)
  {
    os << indent << "Directory not open\n";
    return;
  }

  os << indent << "Directory for: " << this->Path << "\n";
  os << indent << "Contains the following files:\n";
  indent = indent.GetNextIndent();
  for (int i = 0; i < this->Files->GetNumberOfValues(); i++)
  {
    os << indent << this->Files->GetValue(i) << "\n";
  }
}

// First microsoft and borland compilers

#if defined(_MSC_VER) || defined(__BORLANDC__) || defined(__MINGW32__)
#include "svtkWindows.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <sys/types.h>

int svtkDirectory::Open(const char* name)
{
  // clean up from any previous open
  this->CleanUpFilesAndPath();

  char* buf = 0;
  int n = static_cast<int>(strlen(name));
  if (name[n - 1] == '/')
  {
    buf = new char[n + 1 + 1];
    snprintf(buf, n + 1 + 1, "%s*", name);
  }
  else
  {
    buf = new char[n + 2 + 1];
    snprintf(buf, n + 2 + 1, "%s/*", name);
  }
  struct _finddata_t data; // data of current file

  // First count the number of files in the directory
  intptr_t srchHandle;

  srchHandle = _findfirst(buf, &data);

  if (srchHandle == -1)
  {
    _findclose(srchHandle);
    delete[] buf;
    return 0;
  }

  delete[] buf;

  // Loop through names
  do
  {
    this->Files->InsertNextValue(data.name);
  } while (_findnext(srchHandle, &data) != -1);

  this->Path = strcpy(new char[strlen(name) + 1], name);

  return _findclose(srchHandle) != -1;
}

const char* svtkDirectory::GetCurrentWorkingDirectory(char* buf, unsigned int len)
{
  return _getcwd(buf, len);
}

#else

// Now the POSIX style directory access

#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>

// PGI with glibc has trouble with dirent and large file support:
//  http://www.pgroup.com/userforum/viewtopic.php?
//  p=1992&sid=f16167f51964f1a68fe5041b8eb213b6
// Work around the problem by mapping dirent the same way as readdir.
#if defined(__PGI) && defined(__GLIBC__)
#define svtkdirectory_dirent_readdir dirent
#define svtkdirectory_dirent_readdir64 dirent64
#define svtkdirectory_dirent svtkdirectory_dirent_lookup(readdir)
#define svtkdirectory_dirent_lookup(x) svtkdirectory_dirent_lookup_delay(x)
#define svtkdirectory_dirent_lookup_delay(x) svtkdirectory_dirent_##x
#else
#define svtkdirectory_dirent dirent
#endif

int svtkDirectory::Open(const char* name)
{
  // clean up from any previous open
  this->CleanUpFilesAndPath();

  DIR* dir = opendir(name);

  if (!dir)
  {
    return 0;
  }

  svtkdirectory_dirent* d = nullptr;

  for (d = readdir(dir); d; d = readdir(dir))
  {
    this->Files->InsertNextValue(d->d_name);
  }
  this->Path = strcpy(new char[strlen(name) + 1], name);

  closedir(dir);

  return 1;
}

const char* svtkDirectory::GetCurrentWorkingDirectory(char* buf, unsigned int len)
{
  return getcwd(buf, len);
}

#endif

//----------------------------------------------------------------------------
int svtkDirectory::MakeDirectory(const char* dir)
{
  return svtksys::SystemTools::MakeDirectory(dir);
}

const char* svtkDirectory::GetFile(svtkIdType index)
{
  if (index >= this->Files->GetNumberOfValues() || index < 0)
  {
    svtkErrorMacro(<< "Bad index for GetFile on svtkDirectory\n");
    return nullptr;
  }

  return this->Files->GetValue(index).c_str();
}

svtkIdType svtkDirectory::GetNumberOfFiles()
{
  return this->Files->GetNumberOfValues();
}

//----------------------------------------------------------------------------
int svtkDirectory::FileIsDirectory(const char* name)
{
  // The svtksys::SystemTools::FileIsDirectory()
  // does not equal the following code (it probably should),
  // and it will broke KWWidgets. Reverse back to 1.30
  // return svtksys::SystemTools::FileIsDirectory(name);

  if (name == nullptr)
  {
    return 0;
  }

  int absolutePath = 0;
#if defined(_WIN32)
  if (name[0] == '/' || name[0] == '\\')
  {
    absolutePath = 1;
  }
  else
  {
    for (int i = 0; name[i] != '\0'; i++)
    {
      if (name[i] == ':')
      {
        absolutePath = 1;
        break;
      }
      else if (name[i] == '/' || name[i] == '\\')
      {
        break;
      }
    }
  }
#else
  if (name[0] == '/')
  {
    absolutePath = 1;
  }
#endif

  char* fullPath;

  int n = 0;
  if (!absolutePath && this->Path)
  {
    n = static_cast<int>(strlen(this->Path));
  }

  int m = static_cast<int>(strlen(name));

  fullPath = new char[n + m + 2];

  if (!absolutePath && this->Path)
  {
    strcpy(fullPath, this->Path);
#if defined(_WIN32)
    if (fullPath[n - 1] != '/' && fullPath[n - 1] != '\\')
    {
#if !defined(__CYGWIN__)
      fullPath[n++] = '\\';
#else
      fullPath[n++] = '/';
#endif
    }
#else
    if (fullPath[n - 1] != '/')
    {
      fullPath[n++] = '/';
    }
#endif
  }

  strcpy(&fullPath[n], name);

  int result = 0;
  svtksys::SystemTools::Stat_t fs;
  if (svtksys::SystemTools::Stat(fullPath, &fs) == 0)
  {
#if defined(_WIN32)
    result = ((fs.st_mode & _S_IFDIR) != 0);
#else
    result = S_ISDIR(fs.st_mode);
#endif
  }

  delete[] fullPath;

  return result;
}

int svtkDirectory::DeleteDirectory(const char* dir)
{
  return svtksys::SystemTools::RemoveADirectory(dir);
}

int svtkDirectory::Rename(const char* oldname, const char* newname)
{
  return 0 == rename(oldname, newname);
}
