/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkEventData.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @brief   platform-independent event data structures
 */

#ifndef svtkEventData_h
#define svtkEventData_h

#include "svtkCommand.h"

// enumeration of possible devices
enum class svtkEventDataDevice
{
  Unknown = -1,
  HeadMountedDisplay,
  RightController,
  LeftController,
  GenericTracker,
  NumberOfDevices
};

const int svtkEventDataNumberOfDevices = (static_cast<int>(svtkEventDataDevice::NumberOfDevices));

// enumeration of possible device inputs
enum class svtkEventDataDeviceInput
{
  Unknown = -1,
  Trigger,
  TrackPad,
  Joystick,
  Grip,
  ApplicationMenu,
  NumberOfInputs
};

const int svtkEventDataNumberOfInputs = (static_cast<int>(svtkEventDataDeviceInput::NumberOfInputs));

// enumeration of actions that can happen
enum class svtkEventDataAction
{
  Unknown = -1,
  Press,
  Release,
  Touch,
  Untouch,
  NumberOfActions
};

class svtkEventDataForDevice;
class svtkEventDataDevice3D;

class svtkEventData : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkEventData, svtkObjectBase);

  int GetType() const { return this->Type; }

  // are two events equivalent
  bool operator==(const svtkEventData& a) const
  {
    return this->Type == a.Type && this->Equivalent(&a);
  }

  // some convenience downcasts
  virtual svtkEventDataForDevice* GetAsEventDataForDevice() { return nullptr; }
  virtual svtkEventDataDevice3D* GetAsEventDataDevice3D() { return nullptr; }

protected:
  svtkEventData() {}
  ~svtkEventData() override {}

  // subclasses override this to define their
  // definition of equivalent
  virtual bool Equivalent(const svtkEventData* ed) const = 0;

  int Type;

private:
  svtkEventData(const svtkEventData& c) = delete;
};

// a subclass for events that may have one or more of
// device, input, and action
class svtkEventDataForDevice : public svtkEventData
{
public:
  svtkTypeMacro(svtkEventDataForDevice, svtkEventData);
  static svtkEventDataForDevice* New()
  {
    svtkEventDataForDevice* ret = new svtkEventDataForDevice;
    ret->InitializeObjectBase();
    return ret;
  }

  svtkEventDataDevice GetDevice() const { return this->Device; }
  svtkEventDataDeviceInput GetInput() const { return this->Input; }
  svtkEventDataAction GetAction() const { return this->Action; }

  void SetDevice(svtkEventDataDevice v) { this->Device = v; }
  void SetInput(svtkEventDataDeviceInput v) { this->Input = v; }
  void SetAction(svtkEventDataAction v) { this->Action = v; }

  svtkEventDataForDevice* GetAsEventDataForDevice() override { return this; }

protected:
  svtkEventDataDevice Device;
  svtkEventDataDeviceInput Input;
  svtkEventDataAction Action;

  bool Equivalent(const svtkEventData* e) const override
  {
    const svtkEventDataForDevice* edd = static_cast<const svtkEventDataForDevice*>(e);
    return this->Device == edd->Device && this->Input == edd->Input && this->Action == edd->Action;
  }

  svtkEventDataForDevice()
  {
    this->Device = svtkEventDataDevice::Unknown;
    this->Input = svtkEventDataDeviceInput::Unknown;
    this->Action = svtkEventDataAction::Unknown;
  }
  ~svtkEventDataForDevice() override {}

private:
  svtkEventDataForDevice(const svtkEventData& c) = delete;
  void operator=(const svtkEventDataForDevice&) = delete;
};

// a subclass for events that have a 3D world position
// direction and orientation.
class svtkEventDataDevice3D : public svtkEventDataForDevice
{
public:
  svtkTypeMacro(svtkEventDataDevice3D, svtkEventDataForDevice);

  svtkEventDataDevice3D* GetAsEventDataDevice3D() override { return this; }

  void GetWorldPosition(double v[3]) const
  {
    v[0] = this->WorldPosition[0];
    v[1] = this->WorldPosition[1];
    v[2] = this->WorldPosition[2];
  }
  const double* GetWorldPosition() const SVTK_SIZEHINT(3) { return this->WorldPosition; }
  void SetWorldPosition(const double p[3])
  {
    this->WorldPosition[0] = p[0];
    this->WorldPosition[1] = p[1];
    this->WorldPosition[2] = p[2];
  }

  void GetWorldDirection(double v[3]) const
  {
    v[0] = this->WorldDirection[0];
    v[1] = this->WorldDirection[1];
    v[2] = this->WorldDirection[2];
  }
  const double* GetWorldDirection() const SVTK_SIZEHINT(3) { return this->WorldDirection; }
  void SetWorldDirection(const double p[3])
  {
    this->WorldDirection[0] = p[0];
    this->WorldDirection[1] = p[1];
    this->WorldDirection[2] = p[2];
  }

  void GetWorldOrientation(double v[4]) const
  {
    v[0] = this->WorldOrientation[0];
    v[1] = this->WorldOrientation[1];
    v[2] = this->WorldOrientation[2];
    v[3] = this->WorldOrientation[3];
  }
  const double* GetWorldOrientation() const SVTK_SIZEHINT(4) { return this->WorldOrientation; }
  void SetWorldOrientation(const double p[4])
  {
    this->WorldOrientation[0] = p[0];
    this->WorldOrientation[1] = p[1];
    this->WorldOrientation[2] = p[2];
    this->WorldOrientation[3] = p[3];
  }

  void GetTrackPadPosition(double v[2]) const
  {
    v[0] = this->TrackPadPosition[0];
    v[1] = this->TrackPadPosition[1];
  }
  const double* GetTrackPadPosition() const SVTK_SIZEHINT(2) { return this->TrackPadPosition; }
  void SetTrackPadPosition(const double p[2])
  {
    this->TrackPadPosition[0] = p[0];
    this->TrackPadPosition[1] = p[1];
  }
  void SetTrackPadPosition(double x, double y)
  {
    this->TrackPadPosition[0] = x;
    this->TrackPadPosition[1] = y;
  }

protected:
  double WorldPosition[3];
  double WorldOrientation[4];
  double WorldDirection[3];
  double TrackPadPosition[2];

  svtkEventDataDevice3D() {}
  ~svtkEventDataDevice3D() override {}

private:
  svtkEventDataDevice3D(const svtkEventDataDevice3D& c) = delete;
  void operator=(const svtkEventDataDevice3D&) = delete;
};

// subclass for button event 3d
class svtkEventDataButton3D : public svtkEventDataDevice3D
{
public:
  svtkTypeMacro(svtkEventDataButton3D, svtkEventDataDevice3D);
  static svtkEventDataButton3D* New()
  {
    svtkEventDataButton3D* ret = new svtkEventDataButton3D;
    ret->InitializeObjectBase();
    return ret;
  }

protected:
  svtkEventDataButton3D() { this->Type = svtkCommand::Button3DEvent; }
  ~svtkEventDataButton3D() override {}

private:
  svtkEventDataButton3D(const svtkEventDataButton3D& c) = delete;
  void operator=(const svtkEventDataButton3D&) = delete;
};

// subclass for move event 3d
class svtkEventDataMove3D : public svtkEventDataDevice3D
{
public:
  svtkTypeMacro(svtkEventDataMove3D, svtkEventDataDevice3D);
  static svtkEventDataMove3D* New()
  {
    svtkEventDataMove3D* ret = new svtkEventDataMove3D;
    ret->InitializeObjectBase();
    return ret;
  }

protected:
  svtkEventDataMove3D() { this->Type = svtkCommand::Move3DEvent; }
  ~svtkEventDataMove3D() override {}

private:
  svtkEventDataMove3D(const svtkEventDataMove3D& c) = delete;
  void operator=(const svtkEventDataMove3D&) = delete;
};

#endif

// SVTK-HeaderTest-Exclude: svtkEventData.h
