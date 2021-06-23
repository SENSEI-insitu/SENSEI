/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAnimationScene.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "svtkAnimationScene.h"

#include "svtkCollection.h"
#include "svtkCollectionIterator.h"
#include "svtkCommand.h"
#include "svtkObjectFactory.h"
#include "svtkTimerLog.h"

svtkStandardNewMacro(svtkAnimationScene);

//----------------------------------------------------------------------------
svtkAnimationScene::svtkAnimationScene()
{
  this->PlayMode = PLAYMODE_SEQUENCE;
  this->FrameRate = 10.0;
  this->Loop = 0;
  this->InPlay = 0;
  this->StopPlay = 0;

  this->AnimationCues = svtkCollection::New();
  this->AnimationCuesIterator = this->AnimationCues->NewIterator();
  this->AnimationTimer = svtkTimerLog::New();
}

//----------------------------------------------------------------------------
svtkAnimationScene::~svtkAnimationScene()
{
  if (this->InPlay)
  {
    this->Stop();
  }
  this->AnimationCues->Delete();
  this->AnimationCuesIterator->Delete();
  this->AnimationTimer->Delete();
}

//----------------------------------------------------------------------------
void svtkAnimationScene::AddCue(svtkAnimationCue* cue)
{
  if (this->AnimationCues->IsItemPresent(cue))
  {
    svtkErrorMacro("Animation cue already present in the scene");
    return;
  }
  if (this->TimeMode == svtkAnimationCue::TIMEMODE_NORMALIZED &&
    cue->GetTimeMode() != svtkAnimationCue::TIMEMODE_NORMALIZED)
  {
    svtkErrorMacro("A cue with relative time mode cannot be added to a scene "
                  "with normalized time mode.");
    return;
  }
  this->AnimationCues->AddItem(cue);
}

//----------------------------------------------------------------------------
void svtkAnimationScene::RemoveCue(svtkAnimationCue* cue)
{
  this->AnimationCues->RemoveItem(cue);
}

//----------------------------------------------------------------------------
void svtkAnimationScene::RemoveAllCues()
{
  this->AnimationCues->RemoveAllItems();
}
//----------------------------------------------------------------------------
int svtkAnimationScene::GetNumberOfCues()
{
  return this->AnimationCues->GetNumberOfItems();
}
//----------------------------------------------------------------------------
void svtkAnimationScene::SetTimeMode(int mode)
{
  if (mode == svtkAnimationCue::TIMEMODE_NORMALIZED)
  {
    // If normalized time mode is being set on the scene,
    // ensure that none of the contained cues need relative times.
    svtkCollectionIterator* it = this->AnimationCuesIterator;
    for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextItem())
    {
      svtkAnimationCue* cue = svtkAnimationCue::SafeDownCast(it->GetCurrentObject());
      if (cue && cue->GetTimeMode() != svtkAnimationCue::TIMEMODE_NORMALIZED)
      {
        svtkErrorMacro("Scene contains a cue in relative mode. It must be removed "
                      "or changed to normalized mode before changing the scene time mode");
        return;
      }
    }
  }
  this->Superclass::SetTimeMode(mode);
}

//----------------------------------------------------------------------------
void svtkAnimationScene::InitializeChildren()
{
  // run through all the cues and init them.
  svtkCollectionIterator* it = this->AnimationCuesIterator;
  for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextItem())
  {
    svtkAnimationCue* cue = svtkAnimationCue::SafeDownCast(it->GetCurrentObject());
    if (cue)
    {
      cue->Initialize();
    }
  }
}

//----------------------------------------------------------------------------
void svtkAnimationScene::FinalizeChildren()
{
  svtkCollectionIterator* it = this->AnimationCuesIterator;
  for (it->InitTraversal(); !it->IsDoneWithTraversal(); it->GoToNextItem())
  {
    svtkAnimationCue* cue = svtkAnimationCue::SafeDownCast(it->GetCurrentObject());
    if (cue)
    {
      cue->Finalize();
    }
  }
}

//----------------------------------------------------------------------------
void svtkAnimationScene::Play()
{
  if (this->InPlay)
  {
    return;
  }

  if (this->TimeMode == svtkAnimationCue::TIMEMODE_NORMALIZED)
  {
    svtkErrorMacro("Cannot play a scene with normalized time mode");
    return;
  }
  if (this->EndTime <= this->StartTime)
  {
    svtkErrorMacro("Scene start and end times are not suitable for playing");
    return;
  }

  this->InvokeEvent(svtkCommand::StartEvent);

  this->InPlay = 1;
  this->StopPlay = 0;
  this->FrameRate = (this->FrameRate == 0.0) ? 1.0 : this->FrameRate;
  // the actual play loop, check for StopPlay flag.

  double currenttime = this->AnimationTime;
  // adjust currenttime to a valid time.
  currenttime =
    (currenttime < this->StartTime || currenttime >= this->EndTime) ? this->StartTime : currenttime;

  double time_per_frame = (this->PlayMode == PLAYMODE_SEQUENCE) ? (1.0 / this->FrameRate) : 1;
  do
  {
    this->Initialize(); // Set the Scene in uninitialized mode.
    this->AnimationTimer->StartTimer();
    double timer_start_time = currenttime;
    double deltatime = 0.0;
    do
    {
      this->Tick(currenttime, deltatime, currenttime);

      // needed to compute delta times.
      double previous_tick_time = currenttime;

      switch (this->PlayMode)
      {
        case PLAYMODE_REALTIME:
          this->AnimationTimer->StopTimer();
          currenttime = this->AnimationTimer->GetElapsedTime() + timer_start_time;
          break;

        case PLAYMODE_SEQUENCE:
          currenttime += time_per_frame;
          break;

        default:
          svtkErrorMacro("Invalid Play Mode");
          this->StopPlay = 1;
      }

      deltatime = currenttime - previous_tick_time;
      deltatime = (deltatime < 0) ? -1 * deltatime : deltatime;
    } while (!this->StopPlay && this->CueState != svtkAnimationCue::INACTIVE);
    // End of loop for 1 cycle.

    // restart the loop.
    currenttime = this->StartTime;
  } while (this->Loop && !this->StopPlay);

  this->StopPlay = 0;
  this->InPlay = 0;

  this->InvokeEvent(svtkCommand::EndEvent);
}

//----------------------------------------------------------------------------
void svtkAnimationScene::Stop()
{
  if (!this->InPlay)
  {
    return;
  }
  this->StopPlay = 1;
}

//----------------------------------------------------------------------------
void svtkAnimationScene::TickInternal(double currenttime, double deltatime, double clocktime)
{
  this->AnimationTime = currenttime;
  this->ClockTime = clocktime;

  svtkCollectionIterator* iter = this->AnimationCuesIterator;
  for (iter->InitTraversal(); !iter->IsDoneWithTraversal(); iter->GoToNextItem())
  {
    svtkAnimationCue* cue = svtkAnimationCue::SafeDownCast(iter->GetCurrentObject());
    if (cue)
    {
      switch (cue->GetTimeMode())
      {
        case svtkAnimationCue::TIMEMODE_RELATIVE:
          cue->Tick(currenttime - this->StartTime, deltatime, clocktime);
          break;

        case svtkAnimationCue::TIMEMODE_NORMALIZED:
          cue->Tick((currenttime - this->StartTime) / (this->EndTime - this->StartTime),
            deltatime / (this->EndTime - this->StartTime), clocktime);
          break;

        default:
          svtkErrorMacro("Invalid cue time mode");
      }
    }
  }

  this->Superclass::TickInternal(currenttime, deltatime, clocktime);
}

//----------------------------------------------------------------------------
void svtkAnimationScene::StartCueInternal()
{
  this->Superclass::StartCueInternal();
  this->InitializeChildren();
}

//----------------------------------------------------------------------------
void svtkAnimationScene::EndCueInternal()
{
  this->FinalizeChildren();
  this->Superclass::EndCueInternal();
}

//----------------------------------------------------------------------------
void svtkAnimationScene::SetAnimationTime(double currenttime)
{
  if (this->InPlay)
  {
    svtkErrorMacro("SetAnimationTime cannot be called while playing");
    return;
  }
  this->Initialize();
  this->Tick(currenttime, 0.0, currenttime);
  if (this->CueState == svtkAnimationCue::INACTIVE)
  {
    this->Finalize();
  }
}

//----------------------------------------------------------------------------
void svtkAnimationScene::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
  os << indent << "PlayMode: " << this->PlayMode << endl;
  os << indent << "FrameRate: " << this->FrameRate << endl;
  os << indent << "Loop: " << this->Loop << endl;
  os << indent << "InPlay: " << this->InPlay << endl;
  os << indent << "StopPlay: " << this->StopPlay << endl;
}
