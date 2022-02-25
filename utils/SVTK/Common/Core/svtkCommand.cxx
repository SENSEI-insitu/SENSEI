/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCommand.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkCommand.h"
#include "svtkDebugLeaks.h"

#ifdef SVTK_DEBUG_LEAKS
static const char* leakname = "svtkCommand or subclass";
#endif

//----------------------------------------------------------------
svtkCommand::svtkCommand()
  : AbortFlag(0)
  , PassiveObserver(0)
{
#ifdef SVTK_DEBUG_LEAKS
  svtkDebugLeaks::ConstructClass(leakname);
#endif
}

//----------------------------------------------------------------
void svtkCommand::UnRegister()
{
  int refcount = this->GetReferenceCount() - 1;
  this->SetReferenceCount(refcount);
  if (refcount <= 0)
  {
#ifdef SVTK_DEBUG_LEAKS
    svtkDebugLeaks::DestructClass(leakname);
#endif
    delete this;
  }
}

//----------------------------------------------------------------
const char* svtkCommand::GetStringFromEventId(unsigned long event)
{
  switch (event)
  {
// clang-format off
#define _svtk_add_event(Enum)                                                                       \
  case Enum:                                                                                       \
    return #Enum;

  svtkAllEventsMacro()

#undef _svtk_add_event
      // clang-format on

      case UserEvent : return "UserEvent";

    case NoEvent:
      return "NoEvent";
  }

  // Unknown event. Original code was returning NoEvent, so I'll stick with
  // that.
  return "NoEvent";
}

//----------------------------------------------------------------
unsigned long svtkCommand::GetEventIdFromString(const char* event)
{
  if (event)
  {

// clang-format off
#define _svtk_add_event(Enum)                                                                       \
  if (strcmp(event, #Enum) == 0)                                                                   \
  {                                                                                                \
    return Enum;                                                                                   \
  }

    svtkAllEventsMacro()

#undef _svtk_add_event

    if (strcmp("UserEvent",event) == 0)
    {
      return svtkCommand::UserEvent;
    }
    // clang-format on
  }

  return svtkCommand::NoEvent;
}

bool svtkCommand::EventHasData(unsigned long event)
{
  switch (event)
  {
    case svtkCommand::Button3DEvent:
    case svtkCommand::Move3DEvent:
      return true;
    default:
      return false;
  }
}
