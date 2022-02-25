/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTimeStamp.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/**
 * @class   svtkTimeStamp
 * @brief   record modification and/or execution time
 *
 * svtkTimeStamp records a unique time when the method Modified() is
 * executed. This time is guaranteed to be monotonically increasing.
 * Classes use this object to record modified and/or execution time.
 * There is built in support for the binary < and > comparison
 * operators between two svtkTimeStamp objects.
 */

#ifndef svtkTimeStamp_h
#define svtkTimeStamp_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkSystemIncludes.h"

class SVTKCOMMONCORE_EXPORT svtkTimeStamp
{
public:
  svtkTimeStamp() { this->ModifiedTime = 0; }
  static svtkTimeStamp* New();
  void Delete() { delete this; }

  /**
   * Set this objects time to the current time. The current time is
   * just a monotonically increasing unsigned long integer. It is
   * possible for this number to wrap around back to zero.
   * This should only happen for processes that have been running
   * for a very long time, while constantly changing objects
   * within the program. When this does occur, the typical consequence
   * should be that some filters will update themselves when really
   * they don't need to.
   */
  void Modified();

  /**
   * Return this object's Modified time.
   */
  svtkMTimeType GetMTime() const { return this->ModifiedTime; }

  //@{
  /**
   * Support comparisons of time stamp objects directly.
   */
  bool operator>(svtkTimeStamp& ts) { return (this->ModifiedTime > ts.ModifiedTime); }
  bool operator<(svtkTimeStamp& ts) { return (this->ModifiedTime < ts.ModifiedTime); }
  //@}

  /**
   * Allow for typecasting to unsigned long.
   */
  operator svtkMTimeType() const { return this->ModifiedTime; }

private:
  svtkMTimeType ModifiedTime;
};

#endif
// SVTK-HeaderTest-Exclude: svtkTimeStamp.h
