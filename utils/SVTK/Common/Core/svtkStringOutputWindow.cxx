/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStringOutputWindow.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkStringOutputWindow.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkStringOutputWindow);

svtkStringOutputWindow::svtkStringOutputWindow()
{
  this->OStream.str("");
  this->OStream.clear();
}

svtkStringOutputWindow::~svtkStringOutputWindow() = default;

void svtkStringOutputWindow::Initialize()
{
  this->OStream.str("");
  this->OStream.clear();
}

void svtkStringOutputWindow::DisplayText(const char* text)
{
  if (!text)
  {
    return;
  }

  this->OStream << text << endl;
}

void svtkStringOutputWindow::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
