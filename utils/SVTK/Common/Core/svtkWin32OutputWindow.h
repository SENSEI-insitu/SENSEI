/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkWin32OutputWindow.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkWin32OutputWindow
 * @brief   Win32 Specific output window class
 *
 * This class is used for error and debug message output on the Windows
 * platform.   It creates a read only EDIT control to display the
 * output.   This class should not be used directly.   It should
 * only be used through the interface of svtkOutputWindow.  This class
 * only handles one output window per process.  If the window is destroyed,
 * the svtkObject::GlobalWarningDisplayOff() function is called.  The
 * window is created the next time text is written to the window.
 *
 * In its constructor, svtkWin32OutputWindow changes the default
 * `svtkOutputWindow::DisplayMode` to
 * `svtkOutputWindow::NEVER` unless running on a dashboard machine,
 * in which cause it's left as `svtkOutputWindow::DEFAULT`.
 */

#ifndef svtkWin32OutputWindow_h
#define svtkWin32OutputWindow_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkOutputWindow.h"

class SVTKCOMMONCORE_EXPORT svtkWin32OutputWindow : public svtkOutputWindow
{
public:
  // Methods from svtkObject
  svtkTypeMacro(svtkWin32OutputWindow, svtkOutputWindow);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Create a svtkWin32OutputWindow.
   */
  static svtkWin32OutputWindow* New();

  /**
   * New lines are converted to carriage return new lines.
   */
  void DisplayText(const char*) override;

  //@{
  /**
   * Set or get whether the svtkWin32OutputWindow should also send its output
   * to stderr / cerr.
   *
   * @deprecated in SVTK 9.0. Please use `svtkOutputWindow::SetDisplayMode` instead.
   */
  SVTK_LEGACY(void SetSendToStdErr(bool));
  SVTK_LEGACY(bool GetSendToStdErr());
  SVTK_LEGACY(void SendToStdErrOn());
  SVTK_LEGACY(void SendToStdErrOff());
  //@}

protected:
  svtkWin32OutputWindow();
  ~svtkWin32OutputWindow() override;

  void PromptText(const char* text);
  static void AddText(const char*);
  static int Initialize();

private:
  svtkWin32OutputWindow(const svtkWin32OutputWindow&) = delete;
  void operator=(const svtkWin32OutputWindow&) = delete;
};

#endif
