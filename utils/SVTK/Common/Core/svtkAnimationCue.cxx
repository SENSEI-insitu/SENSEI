/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAnimationCue.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#include "svtkAnimationCue.h"
#include "svtkCommand.h"
#include "svtkObjectFactory.h"

svtkStandardNewMacro(svtkAnimationCue);

//----------------------------------------------------------------------------
svtkAnimationCue::svtkAnimationCue()
{
  this->StartTime = this->EndTime = 0.0;
  this->CueState = svtkAnimationCue::UNINITIALIZED;
  this->TimeMode = TIMEMODE_RELATIVE;
  this->AnimationTime = 0;
  this->DeltaTime = 0;
  this->ClockTime = 0;
}

//----------------------------------------------------------------------------
svtkAnimationCue::~svtkAnimationCue() = default;

//----------------------------------------------------------------------------
void svtkAnimationCue::StartCueInternal()
{
  svtkAnimationCue::AnimationCueInfo info;
  info.StartTime = this->StartTime;
  info.EndTime = this->EndTime;
  info.AnimationTime = 0.0;
  info.DeltaTime = 0.0;
  info.ClockTime = 0.0;
  this->InvokeEvent(svtkCommand::StartAnimationCueEvent, &info);
}

//----------------------------------------------------------------------------
void svtkAnimationCue::EndCueInternal()
{
  svtkAnimationCue::AnimationCueInfo info;
  info.StartTime = this->StartTime;
  info.EndTime = this->EndTime;
  info.AnimationTime = this->EndTime;
  info.DeltaTime = 0.0;
  info.ClockTime = 0.0;
  this->InvokeEvent(svtkCommand::EndAnimationCueEvent, &info);
}

//----------------------------------------------------------------------------
void svtkAnimationCue::TickInternal(double currenttime, double deltatime, double clocktime)
{
  svtkAnimationCue::AnimationCueInfo info;
  info.StartTime = this->StartTime;
  info.EndTime = this->EndTime;
  info.DeltaTime = deltatime;
  info.AnimationTime = currenttime;
  info.ClockTime = clocktime;

  this->AnimationTime = currenttime;
  this->DeltaTime = deltatime;
  this->ClockTime = clocktime;

  this->InvokeEvent(svtkCommand::AnimationCueTickEvent, &info);

  this->AnimationTime = 0;
  this->DeltaTime = 0;
  this->ClockTime = 0;
}

//----------------------------------------------------------------------------
void svtkAnimationCue::Tick(double currenttime, double deltatime, double clocktime)
{
  // Check to see if we have crossed the Cue start.
  if (currenttime >= this->StartTime && this->CueState == svtkAnimationCue::UNINITIALIZED)
  {
    this->CueState = svtkAnimationCue::ACTIVE;
    this->StartCueInternal();
  }

  // Note that Tick event is sent for both start time and
  // end time.
  if (this->CueState == svtkAnimationCue::ACTIVE)
  {
    if (currenttime <= this->EndTime)
    {
      this->TickInternal(currenttime, deltatime, clocktime);
    }
    if (currenttime >= this->EndTime)
    {
      this->EndCueInternal();
      this->CueState = svtkAnimationCue::INACTIVE;
    }
  }
}

//----------------------------------------------------------------------------
void svtkAnimationCue::SetTimeMode(int mode)
{
  this->TimeMode = mode;
}

//----------------------------------------------------------------------------
void svtkAnimationCue::Initialize()
{
  this->CueState = svtkAnimationCue::UNINITIALIZED;
}

//----------------------------------------------------------------------------
void svtkAnimationCue::Finalize()
{
  if (this->CueState == svtkAnimationCue::ACTIVE)
  {
    this->EndCueInternal();
  }
  this->CueState = svtkAnimationCue::INACTIVE;
}

//----------------------------------------------------------------------------
void svtkAnimationCue::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "StartTime: " << this->StartTime << endl;
  os << indent << "EndTime: " << this->EndTime << endl;
  os << indent << "CueState: " << this->CueState << endl;
  os << indent << "TimeMode: " << this->TimeMode << endl;
  os << indent << "AnimationTime: " << this->AnimationTime << endl;
  os << indent << "DeltaTime: " << this->DeltaTime << endl;
  os << indent << "ClockTime: " << this->ClockTime << endl;
}
