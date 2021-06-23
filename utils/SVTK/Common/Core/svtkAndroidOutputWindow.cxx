/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAndroidOutputWindow.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkAndroidOutputWindow.h"

#include "svtkCommand.h"
#include "svtkObjectFactory.h"
#include <sstream>

#include <android/log.h>

svtkStandardNewMacro(svtkAndroidOutputWindow);

//----------------------------------------------------------------------------
svtkAndroidOutputWindow::svtkAndroidOutputWindow() {}

//----------------------------------------------------------------------------
svtkAndroidOutputWindow::~svtkAndroidOutputWindow() {}

//----------------------------------------------------------------------------
void svtkAndroidOutputWindow::DisplayErrorText(const char* someText)
{
  if (!someText)
  {
    return;
  }

  std::istringstream stream(someText);
  std::string line;
  while (std::getline(stream, line))
  {
    __android_log_print(ANDROID_LOG_ERROR, "SVTK", "%s", line.c_str());
  }
  this->InvokeEvent(svtkCommand::ErrorEvent, (void*)someText);
}

//----------------------------------------------------------------------------
void svtkAndroidOutputWindow::DisplayWarningText(const char* someText)
{
  if (!someText)
  {
    return;
  }

  std::istringstream stream(someText);
  std::string line;
  while (std::getline(stream, line))
  {
    __android_log_print(ANDROID_LOG_WARN, "SVTK", "%s", line.c_str());
  }
  this->InvokeEvent(svtkCommand::WarningEvent, (void*)someText);
}

//----------------------------------------------------------------------------
void svtkAndroidOutputWindow::DisplayGenericWarningText(const char* someText)
{
  if (!someText)
  {
    return;
  }

  std::istringstream stream(someText);
  std::string line;
  while (std::getline(stream, line))
  {
    __android_log_print(ANDROID_LOG_WARN, "SVTK", "%s", line.c_str());
  }
}

//----------------------------------------------------------------------------
void svtkAndroidOutputWindow::DisplayDebugText(const char* someText)
{
  if (!someText)
  {
    return;
  }

  std::istringstream stream(someText);
  std::string line;
  while (std::getline(stream, line))
  {
    __android_log_print(ANDROID_LOG_DEBUG, "SVTK", "%s", line.c_str());
  }
}

//----------------------------------------------------------------------------
void svtkAndroidOutputWindow::DisplayText(const char* someText)
{
  if (!someText)
  {
    return;
  }

  std::istringstream stream(someText);
  std::string line;
  while (std::getline(stream, line))
  {
    __android_log_print(ANDROID_LOG_INFO, "SVTK", "%s", line.c_str());
  }
}

//----------------------------------------------------------------------------
void svtkAndroidOutputWindow::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
