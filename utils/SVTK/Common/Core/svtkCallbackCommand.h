/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCallbackCommand.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCallbackCommand
 * @brief   supports function callbacks
 *
 * Use svtkCallbackCommand for generic function callbacks. That is, this class
 * can be used when you wish to execute a function (of the signature
 * described below) using the Command/Observer design pattern in SVTK.
 * The callback function should have the form
 * <pre>
 * void func(svtkObject*, unsigned long eid, void* clientdata, void *calldata)
 * </pre>
 * where the parameter svtkObject* is the object invoking the event; eid is
 * the event id (see svtkCommand.h); clientdata is special data that should
 * is associated with this instance of svtkCallbackCommand; and calldata is
 * data that the svtkObject::InvokeEvent() may send with the callback. For
 * example, the invocation of the ProgressEvent sends along the progress
 * value as calldata.
 *
 *
 * @sa
 * svtkCommand svtkOldStyleCallbackCommand
 */

#ifndef svtkCallbackCommand_h
#define svtkCallbackCommand_h

#include "svtkCommand.h"
#include "svtkCommonCoreModule.h" // For export macro

class SVTKCOMMONCORE_EXPORT svtkCallbackCommand : public svtkCommand
{
public:
  svtkTypeMacro(svtkCallbackCommand, svtkCommand);

  static svtkCallbackCommand* New() { return new svtkCallbackCommand; }

  /**
   * Satisfy the superclass API for callbacks. Recall that the caller is
   * the instance invoking the event; eid is the event id (see
   * svtkCommand.h); and calldata is information sent when the callback
   * was invoked (e.g., progress value in the svtkCommand::ProgressEvent).
   */
  void Execute(svtkObject* caller, unsigned long eid, void* callData) override;

  /**
   * Methods to set and get client and callback information, and the callback
   * function.
   */
  virtual void SetClientData(void* cd) { this->ClientData = cd; }
  virtual void* GetClientData() { return this->ClientData; }
  virtual void SetCallback(
    void (*f)(svtkObject* caller, unsigned long eid, void* clientdata, void* calldata))
  {
    this->Callback = f;
  }
  virtual void SetClientDataDeleteCallback(void (*f)(void*)) { this->ClientDataDeleteCallback = f; }

  /**
   * Set/Get the abort flag on execute. If this is set to true the AbortFlag
   * will be set to On automatically when the Execute method is triggered *and*
   * a callback is set.
   */
  void SetAbortFlagOnExecute(int f) { this->AbortFlagOnExecute = f; }
  int GetAbortFlagOnExecute() { return this->AbortFlagOnExecute; }
  void AbortFlagOnExecuteOn() { this->SetAbortFlagOnExecute(1); }
  void AbortFlagOnExecuteOff() { this->SetAbortFlagOnExecute(0); }

  void (*Callback)(svtkObject*, unsigned long, void*, void*);
  void (*ClientDataDeleteCallback)(void*);

protected:
  int AbortFlagOnExecute;
  void* ClientData;

  svtkCallbackCommand();
  ~svtkCallbackCommand() override;
};

#endif

// SVTK-HeaderTest-Exclude: svtkCallbackCommand.h
