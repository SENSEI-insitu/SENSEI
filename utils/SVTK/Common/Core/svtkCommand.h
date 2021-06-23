/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkCommand.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkCommand
 * @brief   superclass for callback/observer methods
 *
 * svtkCommand is an implementation of the observer/command design
 * pattern.  In this design pattern, any instance of svtkObject can be
 * "observed" for any events it might invoke. For example,
 * svtkRenderer invokes a StartEvent as it begins to render and a
 * EndEvent when it finishes rendering. Filters (subclasses of
 * svtkProcessObject) invoke StartEvent, ProgressEvent, and EndEvent as
 * the filter processes data. Observers of events are added with the
 * AddObserver() method found in svtkObject.  AddObserver(), besides
 * requiring an event id or name, also takes an instance of svtkCommand
 * (or a subclasses). Note that svtkCommand is meant to be subclassed,
 * so that you can package the information necessary to support your
 * callback.
 *
 * Event processing can be organized in priority lists, so it is
 * possible to truncate the processing of a particular event by
 * setting the AbortFlag variable. The priority is set using the
 * AddObserver() method.  By default the priority is 0, events of the
 * same priority are processed in last-in-first-processed order. The
 * ordering/aborting of events is important for things like 3D
 * widgets, which handle an event if the widget is selected (and then
 * aborting further processing of that event).  Otherwise. the event
 * is passed along for further processing.
 *
 * When an instance of svtkObject invokes an event, it also passes an optional
 * void pointer to a callData. This callData is nullptr most of the time.
 * The callData is not specific to a type of event but specific to a type
 * of svtkObject invoking a specific event. For instance, svtkCommand::PickEvent
 * is invoked by svtkProp with a nullptr callData but is invoked by
 * svtkInteractorStyleImage with a pointer to the svtkInteractorStyleImage object
 * itself.
 *
 * Here is the list of events that may be invoked with a non-nullptr callData.
 * - svtkCommand::ProgressEvent
 *  - most of the objects return a pointer to a double value ranged between
 * 0.0 and 1.0
 *  - Infovis/svtkFixedWidthTextReader returns a pointer to a float value equal
 * to the number of lines read so far.
 * - svtkCommand::ErrorEvent
 *  - an error message as a const char * string
 * - svtkCommand::WarningEvent
 *  - a warning message as a const char * string
 * - svtkCommand::StartAnimationCueEvent
 *  - a pointer to a svtkAnimationCue::AnimationCueInfo object
 * - svtkCommand::EndAnimationCueEvent
 *  - a pointer to a svtkAnimationCue::AnimationCueInfo object
 * - svtkCommand::AnimationCueTickEvent
 *  - a pointer to a svtkAnimationCue::AnimationCueInfo object
 * - svtkCommand::PickEvent
 *  - Common/svtkProp returns nullptr
 *  - Rendering/svtkInteractorStyleImage returns a pointer to itself
 * - svtkCommand::StartPickEvent
 *  - Rendering/svtkPropPicker returns nullptr
 *  - Rendering/svtkInteractorStyleImage returns a pointer to itself
 * - svtkCommand::EndPickEvent
 *  - Rendering/svtkPropPicker returns nullptr
 *  - Rendering/svtkInteractorStyleImage returns a pointer to itself
 * - svtkCommand::WrongTagEvent
 *  - Parallel/svtkSocketCommunicator returns a received tag as a char *
 * - svtkCommand::SelectionChangedEvent
 *  - Views/svtkView returns nullptr
 *  - Views/svtkDataRepresentation returns a pointer to a svtkSelection
 *  - Rendering/svtkInteractorStyleRubberBand2D returns an array of 5 unsigned
 * integers (p1x, p1y, p2x, p2y, mode), where mode is
 * svtkInteractorStyleRubberBand2D::SELECT_UNION or
 * svtkInteractorStyleRubberBand2D::SELECT_NORMAL
 * - svtkCommand::AnnotationChangedEvent
 *  - GUISupport/Qt/svtkQtAnnotationView returns a pointer to a
 * svtkAnnotationLayers
 * - svtkCommand::PlacePointEvent
 *  - Widgets/svtkSeedWidget returns a pointer to an int, being the current
 * handle number
 * - svtkCommand::DeletePointEvent
 *  - Widgets/svtkSeedWidget returns a pointer to an int, being the
 * handle number of the deleted point
 * - svtkCommand::ResetWindowLevelEvent
 *  - Widgets/svtkImagePlaneWidget returns an array of 2 double values (window
 * and level)
 *  - Rendering/svtkInteractorStyleImage returns a pointer to itself
 * - svtkCommand::StartWindowLevelEvent
 *  - Widgets/svtkImagePlaneWidget returns an array of 2 double values (window
 * and level)
 *  - Rendering/svtkInteractorStyleImage returns a pointer to itself
 * - svtkCommand::EndWindowLevelEvent
 *  - Widgets/svtkImagePlaneWidget returns an array of 2 double values (window
 * and level)
 *  - Rendering/svtkInteractorStyleImage returns a pointer to itself
 * - svtkCommand::WindowLevelEvent
 *  - Widgets/svtkImagePlaneWidget returns an array of 2 double values (window
 * and level)
 *  - Rendering/svtkInteractorStyleImage returns a pointer to itself
 * - svtkCommand::CharEvent
 *  - most of the objects return nullptr
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QKeyEvent *
 * - svtkCommand::TimerEvent
 *  - most of the objects return a to an int representing a timer id
 *  - Widgets/svtkHoverWidget returns nullptr
 * - svtkCommand::CreateTimerEvent
 *  - Rendering/svtkGenericRenderWindowInteractor returns a to an int
 * representing a timer id
 * - svtkCommand::DestroyTimerEvent
 *  - Rendering/svtkGenericRenderWindowInteractor returns a to an int
 * representing a timer id
 * - svtkCommand::UserEvent
 *  - most of the objects return nullptr
 *  - Infovis/svtkInteractorStyleTreeMapHover returns a pointer to a svtkIdType
 * representing a pedigree id
 * - svtkCommand::KeyPressEvent
 *  - most of the objects return nullptr
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QKeyEvent*
 * - svtkCommand::KeyReleaseEvent
 *  - most of the objects return nullptr
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QKeyEvent*
 * - svtkCommand::LeftButtonPressEvent
 *  - most of the objects return nullptr
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QMouseEvent*
 * - svtkCommand::LeftButtonReleaseEvent
 *  - most of the objects return nullptr
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QMouseEvent*
 * - svtkCommand::MouseMoveEvent
 *  - most of the objects return nullptr
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QMouseEvent*
 * - svtkCommand::MouseWheelForwardEvent
 *  - most of the objects return nullptr
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QWheelEvent*
 * - svtkCommand::MouseWheelBackwardEvent
 *  - most of the objects return nullptr
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QWheelEvent*
 * - svtkCommand::RightButtonPressEvent
 *  - most of the objects return nullptr
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QMouseEvent*
 * - svtkCommand::RightButtonReleaseEvent
 *  - most of the objects return nullptr
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QMouseEvent*
 * - svtkCommand::MiddleButtonPressEvent
 *  - most of the objects return nullptr
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QMouseEvent*
 * - svtkCommand::MiddleButtonReleaseEvent
 *  - most of the objects return nullptr
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QMouseEvent*
 * - svtkCommand::CursorChangedEvent
 *  - most of the objects return a pointer to an int representing a shape
 *  - Rendering/svtkInteractorObserver returns nullptr
 * - svtkCommand::ResetCameraEvent
 *  - Rendering/svtkRenderer returns a pointer to itself
 * - svtkCommand::ResetCameraClippingRangeEvent
 *  - Rendering/svtkRenderer returns a pointer to itself
 * - svtkCommand::ActiveCameraEvent
 *  - Rendering/svtkRenderer returns a pointer to the active camera
 * - svtkCommand::CreateCameraEvent
 *  - Rendering/svtkRenderer returns a pointer to the created camera
 * - svtkCommand::EnterEvent
 *  - most of the objects return nullptr
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QEvent*
 * - svtkCommand::LeaveEvent
 *  - most of the objects return nullptr
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QEvent*
 * - svtkCommand::RenderWindowMessageEvent
 *  - Rendering/svtkWin32OpenGLRenderWindow return a pointer to a UINT message
 * - svtkCommand::ComputeVisiblePropBoundsEvent
 *  - Rendering/svtkRenderer returns a pointer to itself
 * - QSVTKOpenGLStereoWidget::ContextMenuEvent
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QContextMenuEvent*
 * - QSVTKOpenGLStereoWidget::DragEnterEvent
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QDragEnterEvent*
 * - QSVTKOpenGLStereoWidget::DragMoveEvent
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QDragMoveEvent*
 * - QSVTKOpenGLStereoWidget::DragLeaveEvent
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QDragLeaveEvent*
 * - QSVTKOpenGLStereoWidget::DropEvent
 *  - GUISupport/Qt/QSVTKOpenGLStereoWidget returns a QDropEvent*
 * - svtkCommand::ViewProgressEvent
 *  - View/svtkView returns a ViewProgressEventCallData*
 * - svtkCommand::VolumeMapperRenderProgressEvent
 *  - A pointer to a double value between 0.0 and 1.0
 * - svtkCommand::VolumeMapperComputeGradientsProgressEvent
 *  - A pointer to a double value between 0.0 and 1.0
 * - svtkCommand::TDxMotionEvent (TDx=3DConnexion)
 *  - A svtkTDxMotionEventInfo*
 * - svtkCommand::TDxButtonPressEvent
 *  - A int* being the number of the button
 * - svtkCommand::TDxButtonReleaseEvent
 *  - A int* being the number of the button
 * - svtkCommand::UpdateShaderEvent
 *  - A svtkOpenGLHelper* currently being used
 * - svtkCommand::FourthButtonPressEvent
 *  - most of the objects return nullptr
 * - svtkCommand::FourthButtonReleaseEvent
 *  - most of the objects return nullptr
 * - svtkCommand::FifthButtonPressEvent
 *  - most of the objects return nullptr
 * - svtkCommand::FifthButtonReleaseEvent
 *  - most of the objects return nullptr
 * - svtkCommand::ErrorEvent
 *  - svtkOutputWindow fires this with `char char*` for the error message
 * - svtkCommand::WarningEvent
 *  - svtkOutputWindow fires this with `char char*` for the warning message
 * - svtkCommand::MessageEvent
 *  - svtkOutputWindow fires this with `char char*` for the message text
 * - svtkCommand::TextEvent
 *  - svtkOutputWindow fires this with `char char*` for the text
 *
 * @sa
 * svtkObject svtkCallbackCommand svtkOldStyleCallbackCommand
 * svtkInteractorObserver svtk3DWidget
 */

#ifndef svtkCommand_h
#define svtkCommand_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"           // Need svtkTypeMacro
#include "svtkObjectBase.h"

// clang-format off
// Define all types of events here.
// Using this macro makes it possible to avoid mismatches between the event
// enums and their string counterparts.
#define svtkAllEventsMacro()                                                                        \
    _svtk_add_event(AnyEvent)                                                                       \
    _svtk_add_event(DeleteEvent)                                                                    \
    _svtk_add_event(StartEvent)                                                                     \
    _svtk_add_event(EndEvent)                                                                       \
    _svtk_add_event(RenderEvent)                                                                    \
    _svtk_add_event(ProgressEvent)                                                                  \
    _svtk_add_event(PickEvent)                                                                      \
    _svtk_add_event(StartPickEvent)                                                                 \
    _svtk_add_event(EndPickEvent)                                                                   \
    _svtk_add_event(AbortCheckEvent)                                                                \
    _svtk_add_event(ExitEvent)                                                                      \
    _svtk_add_event(LeftButtonPressEvent)                                                           \
    _svtk_add_event(LeftButtonReleaseEvent)                                                         \
    _svtk_add_event(MiddleButtonPressEvent)                                                         \
    _svtk_add_event(MiddleButtonReleaseEvent)                                                       \
    _svtk_add_event(RightButtonPressEvent)                                                          \
    _svtk_add_event(RightButtonReleaseEvent)                                                        \
    _svtk_add_event(EnterEvent)                                                                     \
    _svtk_add_event(LeaveEvent)                                                                     \
    _svtk_add_event(KeyPressEvent)                                                                  \
    _svtk_add_event(KeyReleaseEvent)                                                                \
    _svtk_add_event(CharEvent)                                                                      \
    _svtk_add_event(ExposeEvent)                                                                    \
    _svtk_add_event(ConfigureEvent)                                                                 \
    _svtk_add_event(TimerEvent)                                                                     \
    _svtk_add_event(MouseMoveEvent)                                                                 \
    _svtk_add_event(MouseWheelForwardEvent)                                                         \
    _svtk_add_event(MouseWheelBackwardEvent)                                                        \
    _svtk_add_event(ActiveCameraEvent)                                                              \
    _svtk_add_event(CreateCameraEvent)                                                              \
    _svtk_add_event(ResetCameraEvent)                                                               \
    _svtk_add_event(ResetCameraClippingRangeEvent)                                                  \
    _svtk_add_event(ModifiedEvent)                                                                  \
    _svtk_add_event(WindowLevelEvent)                                                               \
    _svtk_add_event(StartWindowLevelEvent)                                                          \
    _svtk_add_event(EndWindowLevelEvent)                                                            \
    _svtk_add_event(ResetWindowLevelEvent)                                                          \
    _svtk_add_event(SetOutputEvent)                                                                 \
    _svtk_add_event(ErrorEvent)                                                                     \
    _svtk_add_event(WarningEvent)                                                                   \
    _svtk_add_event(StartInteractionEvent)                                                          \
    _svtk_add_event(DropFilesEvent)                                                                 \
    _svtk_add_event(UpdateDropLocationEvent)                                                        \
        /*^ mainly used by svtkInteractorObservers*/                                                \
    _svtk_add_event(InteractionEvent)                                                               \
    _svtk_add_event(EndInteractionEvent)                                                            \
    _svtk_add_event(EnableEvent)                                                                    \
    _svtk_add_event(DisableEvent)                                                                   \
    _svtk_add_event(CreateTimerEvent)                                                               \
    _svtk_add_event(DestroyTimerEvent)                                                              \
    _svtk_add_event(PlacePointEvent)                                                                \
    _svtk_add_event(DeletePointEvent)                                                               \
    _svtk_add_event(PlaceWidgetEvent)                                                               \
    _svtk_add_event(CursorChangedEvent)                                                             \
    _svtk_add_event(ExecuteInformationEvent)                                                        \
    _svtk_add_event(RenderWindowMessageEvent)                                                       \
    _svtk_add_event(WrongTagEvent)                                                                  \
    _svtk_add_event(StartAnimationCueEvent)                                                         \
    _svtk_add_event(ResliceAxesChangedEvent)                                                        \
        /*^ used by svtkAnimationCue*/                                                              \
    _svtk_add_event(AnimationCueTickEvent)                                                          \
    _svtk_add_event(EndAnimationCueEvent)                                                           \
    _svtk_add_event(VolumeMapperRenderEndEvent)                                                     \
    _svtk_add_event(VolumeMapperRenderProgressEvent)                                                \
    _svtk_add_event(VolumeMapperRenderStartEvent)                                                   \
    _svtk_add_event(VolumeMapperComputeGradientsEndEvent)                                           \
    _svtk_add_event(VolumeMapperComputeGradientsProgressEvent)                                      \
    _svtk_add_event(VolumeMapperComputeGradientsStartEvent)                                         \
    _svtk_add_event(WidgetModifiedEvent)                                                            \
    _svtk_add_event(WidgetValueChangedEvent)                                                        \
    _svtk_add_event(WidgetActivateEvent)                                                            \
    _svtk_add_event(ConnectionCreatedEvent)                                                         \
    _svtk_add_event(ConnectionClosedEvent)                                                          \
    _svtk_add_event(DomainModifiedEvent)                                                            \
    _svtk_add_event(PropertyModifiedEvent)                                                          \
    _svtk_add_event(UpdateEvent)                                                                    \
    _svtk_add_event(RegisterEvent)                                                                  \
    _svtk_add_event(UnRegisterEvent)                                                                \
    _svtk_add_event(UpdateInformationEvent)                                                         \
    _svtk_add_event(AnnotationChangedEvent)                                                         \
    _svtk_add_event(SelectionChangedEvent)                                                          \
    _svtk_add_event(UpdatePropertyEvent)                                                            \
    _svtk_add_event(ViewProgressEvent)                                                              \
    _svtk_add_event(UpdateDataEvent)                                                                \
    _svtk_add_event(CurrentChangedEvent)                                                            \
    _svtk_add_event(ComputeVisiblePropBoundsEvent)                                                  \
    _svtk_add_event(TDxMotionEvent)                                                                 \
      /*^ 3D Connexion device event */                                                             \
    _svtk_add_event(TDxButtonPressEvent)                                                            \
      /*^ 3D Connexion device event */                                                             \
    _svtk_add_event(TDxButtonReleaseEvent)                                                          \
      /* 3D Connexion device event */                                                              \
    _svtk_add_event(HoverEvent)                                                                     \
    _svtk_add_event(LoadStateEvent)                                                                 \
    _svtk_add_event(SaveStateEvent)                                                                 \
    _svtk_add_event(StateChangedEvent)                                                              \
    _svtk_add_event(WindowMakeCurrentEvent)                                                         \
    _svtk_add_event(WindowIsCurrentEvent)                                                           \
    _svtk_add_event(WindowFrameEvent)                                                               \
    _svtk_add_event(HighlightEvent)                                                                 \
    _svtk_add_event(WindowSupportsOpenGLEvent)                                                      \
    _svtk_add_event(WindowIsDirectEvent)                                                            \
    _svtk_add_event(WindowStereoTypeChangedEvent)                                                   \
    _svtk_add_event(WindowResizeEvent)                                                              \
    _svtk_add_event(UncheckedPropertyModifiedEvent)                                                 \
    _svtk_add_event(UpdateShaderEvent)                                                              \
    _svtk_add_event(MessageEvent)                                                                   \
    _svtk_add_event(StartSwipeEvent)                                                                \
    _svtk_add_event(SwipeEvent)                                                                     \
    _svtk_add_event(EndSwipeEvent)                                                                  \
    _svtk_add_event(StartPinchEvent)                                                                \
    _svtk_add_event(PinchEvent)                                                                     \
    _svtk_add_event(EndPinchEvent)                                                                  \
    _svtk_add_event(StartRotateEvent)                                                               \
    _svtk_add_event(RotateEvent)                                                                    \
    _svtk_add_event(EndRotateEvent)                                                                 \
    _svtk_add_event(StartPanEvent)                                                                  \
    _svtk_add_event(PanEvent)                                                                       \
    _svtk_add_event(EndPanEvent)                                                                    \
    _svtk_add_event(TapEvent)                                                                       \
    _svtk_add_event(LongTapEvent)                                                                   \
    _svtk_add_event(FourthButtonPressEvent)                                                         \
    _svtk_add_event(FourthButtonReleaseEvent)                                                       \
    _svtk_add_event(FifthButtonPressEvent)                                                          \
    _svtk_add_event(FifthButtonReleaseEvent)                                                        \
    _svtk_add_event(Move3DEvent)                                                                    \
    _svtk_add_event(Button3DEvent)                                                                  \
    _svtk_add_event(TextEvent)                                                                      \
    _svtk_add_event(LeftButtonDoubleClickEvent)                                                     \
    _svtk_add_event(RightButtonDoubleClickEvent)
// clang-format on

#define svtkEventDeclarationMacro(_enum_name)                                                       \
  enum _enum_name                                                                                  \
  {                                                                                                \
    NoEvent = 0,                                                                                   \
    svtkAllEventsMacro() UserEvent = 1000                                                           \
  }

// The superclass that all commands should be subclasses of
class SVTKCOMMONCORE_EXPORT svtkCommand : public svtkObjectBase
{
public:
  svtkBaseTypeMacro(svtkCommand, svtkObjectBase);

  /**
   * Decrease the reference count (release by another object). This has
   * the same effect as invoking Delete() (i.e., it reduces the reference
   * count by 1).
   */
  void UnRegister();
  void UnRegister(svtkObjectBase*) override { this->UnRegister(); }

  /**
   * All derived classes of svtkCommand must implement this
   * method. This is the method that actually does the work of the
   * callback. The caller argument is the object invoking the event,
   * the eventId parameter is the id of the event, and callData
   * parameter is data that can be passed into the execute
   * method. (Note: svtkObject::InvokeEvent() takes two parameters: the
   * event id (or name) and call data. Typically call data is nullptr,
   * but the user can package data and pass it this
   * way. Alternatively, a derived class of svtkCommand can be used to
   * pass data.)
   */
  virtual void Execute(svtkObject* caller, unsigned long eventId, void* callData) = 0;

  //@{
  /**
   * Convenience methods for translating between event names and event
   * ids.
   */
  static const char* GetStringFromEventId(unsigned long event);
  static unsigned long GetEventIdFromString(const char* event);
  //@}

  /**
   * Does this event type contain svtkEventData
   */
  static bool EventHasData(unsigned long event);

  /**
   * Set/Get the abort flag. If this is set to true no further
   * commands are executed.
   */
  void SetAbortFlag(int f) { this->AbortFlag = f; }
  int GetAbortFlag() { return this->AbortFlag; }
  void AbortFlagOn() { this->SetAbortFlag(1); }
  void AbortFlagOff() { this->SetAbortFlag(0); }

  /**
   * Set/Get the passive observer flag. If this is set to true, this
   * indicates that this command does not change the state of the
   * system in any way. Passive observers are processed first, and
   * are not called even when another command has focus.
   */
  void SetPassiveObserver(int f) { this->PassiveObserver = f; }
  int GetPassiveObserver() { return this->PassiveObserver; }
  void PassiveObserverOn() { this->SetPassiveObserver(1); }
  void PassiveObserverOff() { this->SetPassiveObserver(0); }

  /**
   * All the currently defined events are listed here.  Developers can
   * use -- svtkCommand::UserEvent + int to specify their own event
   * ids.
   * Add new events by updating svtkAllEventsMacro.
   */
#define _svtk_add_event(Enum) Enum,
  svtkEventDeclarationMacro(EventIds);
#undef _svtk_add_event

protected:
  int AbortFlag;
  int PassiveObserver;

  svtkCommand();
  ~svtkCommand() override {}

  friend class svtkSubjectHelper;

  svtkCommand(const svtkCommand& c)
    : svtkObjectBase(c)
  {
  }
  void operator=(const svtkCommand&) {}
};

#endif /* svtkCommand_h */

// SVTK-HeaderTest-Exclude: svtkCommand.h
