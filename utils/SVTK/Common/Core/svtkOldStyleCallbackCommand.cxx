/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkOldStyleCallbackCommand.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkOldStyleCallbackCommand.h"

#include "svtkObject.h"
#include "svtkSetGet.h"

#include <cctype>
#include <cstring>

//----------------------------------------------------------------
svtkOldStyleCallbackCommand::svtkOldStyleCallbackCommand()
{
  this->ClientData = nullptr;
  this->Callback = nullptr;
  this->ClientDataDeleteCallback = nullptr;
}

svtkOldStyleCallbackCommand::~svtkOldStyleCallbackCommand()
{
  if (this->ClientDataDeleteCallback)
  {
    this->ClientDataDeleteCallback(this->ClientData);
  }
}

void svtkOldStyleCallbackCommand::Execute(svtkObject*, unsigned long, void*)
{
  if (this->Callback)
  {
    this->Callback(this->ClientData);
  }
}
