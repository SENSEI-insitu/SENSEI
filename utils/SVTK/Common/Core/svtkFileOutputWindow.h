/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkFileOutputWindow.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkFileOutputWindow
 * @brief   File Specific output window class
 *
 * Writes debug/warning/error output to a log file instead of the console.
 * To use this class, instantiate it and then call SetInstance(this).
 *
 */

#ifndef svtkFileOutputWindow_h
#define svtkFileOutputWindow_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkOutputWindow.h"

class SVTKCOMMONCORE_EXPORT svtkFileOutputWindow : public svtkOutputWindow
{
public:
  svtkTypeMacro(svtkFileOutputWindow, svtkOutputWindow);

  static svtkFileOutputWindow* New();

  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Put the text into the log file.
   * New lines are converted to carriage return new lines.
   */
  void DisplayText(const char*) override;

  //@{
  /**
   * Sets the name for the log file.
   */
  svtkSetStringMacro(FileName);
  svtkGetStringMacro(FileName);
  //@}

  //@{
  /**
   * Turns on buffer flushing for the output
   * to the log file.
   */
  svtkSetMacro(Flush, svtkTypeBool);
  svtkGetMacro(Flush, svtkTypeBool);
  svtkBooleanMacro(Flush, svtkTypeBool);
  //@}

  //@{
  /**
   * Setting append will cause the log file to be
   * opened in append mode.  Otherwise, if the log file exists,
   * it will be overwritten each time the svtkFileOutputWindow
   * is created.
   */
  svtkSetMacro(Append, svtkTypeBool);
  svtkGetMacro(Append, svtkTypeBool);
  svtkBooleanMacro(Append, svtkTypeBool);
  //@}

protected:
  svtkFileOutputWindow();
  ~svtkFileOutputWindow() override;
  void Initialize();

  char* FileName;
  ostream* OStream;
  svtkTypeBool Flush;
  svtkTypeBool Append;

private:
  svtkFileOutputWindow(const svtkFileOutputWindow&) = delete;
  void operator=(const svtkFileOutputWindow&) = delete;
};

#endif
