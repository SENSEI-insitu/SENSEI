/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkEventForwarderCommand.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkEventForwarderCommand
 * @brief   a simple event forwarder command
 *
 * Use svtkEventForwarderCommand to forward an event to a new object.
 * This command will intercept the event, and use InvokeEvent
 * on a 'target' as if that object was the one that invoked the event instead
 * of the object this command was attached to using AddObserver.
 *
 * @sa
 * svtkCommand
 */

#ifndef svtkEventForwarderCommand_h
#define svtkEventForwarderCommand_h

#include "svtkCommand.h"
#include "svtkCommonCoreModule.h" // For export macro

class SVTKCOMMONCORE_EXPORT svtkEventForwarderCommand : public svtkCommand
{
public:
  svtkTypeMacro(svtkEventForwarderCommand, svtkCommand);

  static svtkEventForwarderCommand* New() { return new svtkEventForwarderCommand; }

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
  virtual void SetTarget(svtkObject* obj) { this->Target = obj; }
  virtual void* GetTarget() { return this->Target; }

protected:
  svtkObject* Target;

  svtkEventForwarderCommand();
  ~svtkEventForwarderCommand() override {}
};

#endif /* svtkEventForwarderCommand_h */

// SVTK-HeaderTest-Exclude: svtkEventForwarderCommand.h
