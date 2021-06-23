/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkWin32ProcessOutputWindow.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkWin32ProcessOutputWindow
 * @brief   Win32-specific output window class
 *
 * svtkWin32ProcessOutputWindow executes a process and sends messages
 * to its standard input pipe.  This is useful to have a separate
 * process display SVTK errors so that if a SVTK application crashes,
 * the error messages are still available.
 */

#ifndef svtkWin32ProcessOutputWindow_h
#define svtkWin32ProcessOutputWindow_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkOutputWindow.h"

class SVTKCOMMONCORE_EXPORT svtkWin32ProcessOutputWindow : public svtkOutputWindow
{
public:
  svtkTypeMacro(svtkWin32ProcessOutputWindow, svtkOutputWindow);
  static svtkWin32ProcessOutputWindow* New();
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Send text to the output window process.
   */
  void DisplayText(const char*) override;

protected:
  svtkWin32ProcessOutputWindow();
  ~svtkWin32ProcessOutputWindow();

  int Initialize();
  void Write(const char* data, size_t length);

  // The write end of the pipe to the child process.
  svtkWindowsHANDLE OutputPipe;

  // Whether the pipe has been broken.
  int Broken;

  // Count the number of times a new child has been initialized.
  unsigned int Count;

private:
  svtkWin32ProcessOutputWindow(const svtkWin32ProcessOutputWindow&) = delete;
  void operator=(const svtkWin32ProcessOutputWindow&) = delete;
};

#endif
