/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCallbackCommand.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkCallbackCommand.h"

//----------------------------------------------------------------
svtkCallbackCommand::svtkCallbackCommand()
{
  this->ClientData = nullptr;
  this->Callback = nullptr;
  this->ClientDataDeleteCallback = nullptr;
  this->AbortFlagOnExecute = 0;
}

//----------------------------------------------------------------
svtkCallbackCommand::~svtkCallbackCommand()
{
  if (this->ClientDataDeleteCallback)
  {
    this->ClientDataDeleteCallback(this->ClientData);
  }
}

//----------------------------------------------------------------
void svtkCallbackCommand::Execute(svtkObject* caller, unsigned long event, void* callData)
{
  if (this->Callback)
  {
    this->Callback(caller, event, this->ClientData, callData);
    if (this->AbortFlagOnExecute)
    {
      this->AbortFlagOn();
    }
  }
}
