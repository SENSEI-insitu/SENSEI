/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkAndroidOutputWindow.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkAndroidOutputWindow
 * @brief   Win32 Specific output window class
 *
 * This class is used for error and debug message output on the windows
 * platform.   It creates a read only EDIT control to display the
 * output.   This class should not be used directly.   It should
 * only be used through the interface of svtkOutputWindow.  This class
 * only handles one output window per process.  If the window is destroyed,
 * the svtkObject::GlobalWarningDisplayOff() function is called.  The
 * window is created the next time text is written to the window.
 */

#ifndef svtkAndroidOutputWindow_h
#define svtkAndroidOutputWindow_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkOutputWindow.h"

class SVTKCOMMONCORE_EXPORT svtkAndroidOutputWindow : public svtkOutputWindow
{
public:
  // Methods from svtkObject
  svtkTypeMacro(svtkAndroidOutputWindow, svtkOutputWindow);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Create a svtkAndroidOutputWindow.
   */
  static svtkAndroidOutputWindow* New();

  //@{
  /**
   * New lines are converted to carriage return new lines.
   */
  void DisplayText(const char*) override;
  virtual void DisplayErrorText(const char*);
  virtual void DisplayWarningText(const char*);
  virtual void DisplayGenericWarningText(const char*);
  //@}

  virtual void DisplayDebugText(const char*);

protected:
  svtkAndroidOutputWindow();
  ~svtkAndroidOutputWindow() override;

private:
  svtkAndroidOutputWindow(const svtkAndroidOutputWindow&) = delete;
  void operator=(const svtkAndroidOutputWindow&) = delete;
};

#endif
