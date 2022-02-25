/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAnimationScene.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkAnimationScene
 * @brief   the animation scene manager.
 *
 * svtkAnimationCue and svtkAnimationScene provide the framework to support
 * animations in SVTK. svtkAnimationCue represents an entity that changes/
 * animates with time, while svtkAnimationScene represents scene or setup
 * for the animation, which consists of individual cues or other scenes.
 *
 * A scene can be played in real time mode, or as a sequence of frames
 * 1/frame rate apart in time.
 * @sa
 * svtkAnimationCue
 */

#ifndef svtkAnimationScene_h
#define svtkAnimationScene_h

#include "svtkAnimationCue.h"
#include "svtkCommonDataModelModule.h" // For export macro

class svtkAnimationCue;
class svtkCollection;
class svtkCollectionIterator;
class svtkTimerLog;

class SVTKCOMMONDATAMODEL_EXPORT svtkAnimationScene : public svtkAnimationCue
{
public:
  svtkTypeMacro(svtkAnimationScene, svtkAnimationCue);
  void PrintSelf(ostream& os, svtkIndent indent) override;
  static svtkAnimationScene* New();

  //@{
  /**
   * Get/Set the PlayMode for running/playing the animation scene.
   * In the Sequence mode, all the frames are generated one after the other.
   * The time reported to each Tick of the constituent cues (during Play) is
   * incremented by 1/frame rate, irrespective of the current time.
   * In the RealTime mode, time indicates the instance in time.
   */
  svtkSetMacro(PlayMode, int);
  void SetModeToSequence() { this->SetPlayMode(PLAYMODE_SEQUENCE); }
  void SetModeToRealTime() { this->SetPlayMode(PLAYMODE_REALTIME); }
  svtkGetMacro(PlayMode, int);
  //@}

  //@{
  /**
   * Get/Set the frame rate (in frames per second).
   * This parameter affects only in the Sequence mode. The time interval
   * indicated to each cue on every tick is progressed by 1/frame-rate seconds.
   */
  svtkSetMacro(FrameRate, double);
  svtkGetMacro(FrameRate, double);
  //@}

  //@{
  /**
   * Add/Remove an AnimationCue to/from the Scene.
   * It's an error to add a cue twice to the Scene.
   */
  void AddCue(svtkAnimationCue* cue);
  void RemoveCue(svtkAnimationCue* cue);
  void RemoveAllCues();
  int GetNumberOfCues();
  //@}

  /**
   * Starts playing the animation scene. Fires a svtkCommand::StartEvent
   * before play beings and svtkCommand::EndEvent after play ends.
   */
  virtual void Play();

  /**
   * Stops the animation scene that is running.
   */
  void Stop();

  //@{
  /**
   * Enable/Disable animation loop.
   */
  svtkSetMacro(Loop, int);
  svtkGetMacro(Loop, int);
  //@}

  /**
   * Makes the state of the scene same as the given time.
   */
  void SetAnimationTime(double time);

  /**
   * Overridden to allow change to Normalized mode only
   * if none of the constituent cues is in Relative time mode.
   */
  void SetTimeMode(int mode) override;

  /**
   * Returns if the animation is being played.
   */
  int IsInPlay() { return this->InPlay; }

  enum PlayModes
  {
    PLAYMODE_SEQUENCE = 0,
    PLAYMODE_REALTIME = 1
  };

protected:
  svtkAnimationScene();
  ~svtkAnimationScene() override;

  //@{
  /**
   * Called on every valid tick.
   * Calls ticks on all the contained cues.
   */
  void TickInternal(double currenttime, double deltatime, double clocktime) override;
  void StartCueInternal() override;
  void EndCueInternal() override;
  //@}

  void InitializeChildren();
  void FinalizeChildren();

  int PlayMode;
  double FrameRate;
  int Loop;
  int InPlay;
  int StopPlay;

  svtkCollection* AnimationCues;
  svtkCollectionIterator* AnimationCuesIterator;
  svtkTimerLog* AnimationTimer;

private:
  svtkAnimationScene(const svtkAnimationScene&) = delete;
  void operator=(const svtkAnimationScene&) = delete;
};

#endif
