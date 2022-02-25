/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkOldStyleCallbackCommand.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkOldStyleCallbackCommand
 * @brief   supports legacy function callbacks for SVTK
 *
 * svtkOldStyleCallbackCommand is a callback that supports the legacy callback
 * methods found in SVTK. For example, the legacy method
 * svtkProcessObject::SetStartMethod() is actually invoked using the
 * command/observer design pattern of SVTK, and the svtkOldStyleCallbackCommand
 * is used to provide the legacy functionality. The callback function should
 * have the form void func(void *clientdata), where clientdata is special data
 * that should is associated with this instance of svtkCallbackCommand.
 *
 * @warning
 * This is legacy glue. Please do not use; it will be eventually eliminated.
 *
 * @sa
 * svtkCommand svtkCallbackCommand
 */

#ifndef svtkOldStyleCallbackCommand_h
#define svtkOldStyleCallbackCommand_h

#include "svtkCommand.h"
#include "svtkCommonCoreModule.h" // For export macro

// the old style void fund(void *) callbacks
class SVTKCOMMONCORE_EXPORT svtkOldStyleCallbackCommand : public svtkCommand
{
public:
  svtkTypeMacro(svtkOldStyleCallbackCommand, svtkCommand);

  static svtkOldStyleCallbackCommand* New() { return new svtkOldStyleCallbackCommand; }

  /**
   * Satisfy the superclass API for callbacks.
   */
  void Execute(svtkObject* invoker, unsigned long eid, void* calldata) override;

  //@{
  /**
   * Methods to set and get client and callback information.
   */
  void SetClientData(void* cd) { this->ClientData = cd; }
  void SetCallback(void (*f)(void* clientdata)) { this->Callback = f; }
  void SetClientDataDeleteCallback(void (*f)(void*)) { this->ClientDataDeleteCallback = f; }
  //@}

  void* ClientData;
  void (*Callback)(void*);
  void (*ClientDataDeleteCallback)(void*);

protected:
  svtkOldStyleCallbackCommand();
  ~svtkOldStyleCallbackCommand() override;
};

#endif /* svtkOldStyleCallbackCommand_h */

// SVTK-HeaderTest-Exclude: svtkOldStyleCallbackCommand.h
