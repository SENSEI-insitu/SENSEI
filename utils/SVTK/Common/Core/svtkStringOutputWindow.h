/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkStringOutputWindow.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkStringOutputWindow
 * @brief   File Specific output window class
 *
 * Writes debug/warning/error output to a log file instead of the console.
 * To use this class, instantiate it and then call SetInstance(this).
 *
 */

#ifndef svtkStringOutputWindow_h
#define svtkStringOutputWindow_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkOutputWindow.h"
#include <sstream> // for ivar

class SVTKCOMMONCORE_EXPORT svtkStringOutputWindow : public svtkOutputWindow
{
public:
  svtkTypeMacro(svtkStringOutputWindow, svtkOutputWindow);

  static svtkStringOutputWindow* New();

  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Put the text into the log file.
   * New lines are converted to carriage return new lines.
   */
  void DisplayText(const char*) override;

  /**
   * Get the current output as a string
   */
  std::string GetOutput() { return this->OStream.str(); }

protected:
  svtkStringOutputWindow();
  ~svtkStringOutputWindow() override;
  void Initialize();

  std::ostringstream OStream;

private:
  svtkStringOutputWindow(const svtkStringOutputWindow&) = delete;
  void operator=(const svtkStringOutputWindow&) = delete;
};

#endif
