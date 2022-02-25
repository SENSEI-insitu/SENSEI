/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkXMLFileOutputWindow.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkXMLFileOutputWindow
 * @brief   XML File Specific output window class
 *
 * Writes debug/warning/error output to an XML file. Uses prefined XML
 * tags for each text display method. The text is processed to replace
 * XML markup characters.
 *
 *   DisplayText - <Text>
 *
 *   DisplayErrorText - <Error>
 *
 *   DisplayWarningText - <Warning>
 *
 *   DisplayGenericWarningText - <GenericWarning>
 *
 *   DisplayDebugText - <Debug>
 *
 * The method DisplayTag outputs the text unprocessed. To use this
 * class, instantiate it and then call SetInstance(this).
 */

#ifndef svtkXMLFileOutputWindow_h
#define svtkXMLFileOutputWindow_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkFileOutputWindow.h"

class SVTKCOMMONCORE_EXPORT svtkXMLFileOutputWindow : public svtkFileOutputWindow
{
public:
  svtkTypeMacro(svtkXMLFileOutputWindow, svtkFileOutputWindow);

  static svtkXMLFileOutputWindow* New();

  //@{
  /**
   * Put the text into the log file. The text is processed to
   * replace &, <, > with &amp, &lt, and &gt.
   * Each display method outputs a different XML tag.
   */
  void DisplayText(const char*) override;
  void DisplayErrorText(const char*) override;
  void DisplayWarningText(const char*) override;
  void DisplayGenericWarningText(const char*) override;
  void DisplayDebugText(const char*) override;
  //@}

  /**
   * Put the text into the log file without processing it.
   */
  virtual void DisplayTag(const char*);

protected:
  svtkXMLFileOutputWindow() {}
  ~svtkXMLFileOutputWindow() override {}

  void Initialize();
  virtual void DisplayXML(const char*, const char*);

private:
  svtkXMLFileOutputWindow(const svtkXMLFileOutputWindow&) = delete;
  void operator=(const svtkXMLFileOutputWindow&) = delete;
};

#endif
// SVTK-HeaderTest-Exclude: svtkXMLFileOutputWindow.h
