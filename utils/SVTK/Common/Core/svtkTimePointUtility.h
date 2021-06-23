/*=========================================================================

  Program:   Visualization Toolkit
  Module:    svtkTimePointUtility.h

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
/*-------------------------------------------------------------------------
  Copyright 2008 Sandia Corporation.
  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
  the U.S. Government retains certain rights in this software.
-------------------------------------------------------------------------*/
/**
 * @class   svtkTimePointUtility
 * @brief   performs common time operations
 *
 *
 * svtkTimePointUtility is provides methods to perform common time operations.
 */

#ifndef svtkTimePointUtility_h
#define svtkTimePointUtility_h

#include "svtkCommonCoreModule.h" // For export macro
#include "svtkObject.h"

class SVTKCOMMONCORE_EXPORT svtkTimePointUtility : public svtkObject
{
public:
  static svtkTimePointUtility* New();
  svtkTypeMacro(svtkTimePointUtility, svtkObject);
  void PrintSelf(ostream& os, svtkIndent indent) override;

  /**
   * Return the time point for 12:00am on a specified day.
   */
  static svtkTypeUInt64 DateToTimePoint(int year, int month, int day);

  /**
   * Return the time point for a time of day (the number of milliseconds from 12:00am.
   * The hour should be from 0-23.
   */
  static svtkTypeUInt64 TimeToTimePoint(int hour, int minute, int second, int millis = 0);

  /**
   * Return the time point for a date and time.
   */
  static svtkTypeUInt64 DateTimeToTimePoint(
    int year, int month, int day, int hour, int minute, int sec, int millis = 0);

  /**
   * Retrieve the year, month, and day of a time point.
   * Everything but the first argument are output parameters.
   */
  static void GetDate(svtkTypeUInt64 time, int& year, int& month, int& day);

  /**
   * Retrieve the hour, minute, second, and milliseconds of a time point.
   * Everything but the first argument are output parameters.
   */
  static void GetTime(svtkTypeUInt64 time, int& hour, int& minute, int& second, int& millis);

  /**
   * Retrieve the date and time of a time point.
   * Everything but the first argument are output parameters.
   */
  static void GetDateTime(svtkTypeUInt64 time, int& year, int& month, int& day, int& hour,
    int& minute, int& second, int& millis);

  /**
   * Retrieve the year from a time point.
   */
  static int GetYear(svtkTypeUInt64 time);

  /**
   * Retrieve the month from a time point.
   */
  static int GetMonth(svtkTypeUInt64 time);

  /**
   * Retrieve the day of the month from a time point.
   */
  static int GetDay(svtkTypeUInt64 time);

  /**
   * Retrieve the hour of the day from the time point.
   */
  static int GetHour(svtkTypeUInt64 time);

  /**
   * Retrieve the number of minutes from the start of the last hour.
   */
  static int GetMinute(svtkTypeUInt64 time);

  /**
   * Retrieve the number of seconds from the start of the last minute.
   */
  static int GetSecond(svtkTypeUInt64 time);

  /**
   * Retrieve the milliseconds from the start of the last second.
   */
  static int GetMillisecond(svtkTypeUInt64 time);

  enum
  {
    ISO8601_DATETIME_MILLIS = 0,
    ISO8601_DATETIME = 1,
    ISO8601_DATE = 2,
    ISO8601_TIME_MILLIS = 3,
    ISO8601_TIME = 4
  };

  static const int MILLIS_PER_SECOND;
  static const int MILLIS_PER_MINUTE;
  static const int MILLIS_PER_HOUR;
  static const int MILLIS_PER_DAY;
  static const int SECONDS_PER_MINUTE;
  static const int SECONDS_PER_HOUR;
  static const int SECONDS_PER_DAY;
  static const int MINUTES_PER_HOUR;
  static const int MINUTES_PER_DAY;
  static const int HOURS_PER_DAY;

  /**
   * Converts a ISO8601 string into a SVTK timepoint.
   * The string must follow one of the ISO8601 formats described
   * in ToISO8601.  To check for a valid format, pass a bool* as
   * the second argument.  The value will be set to true if the
   * string was parsed successfully, false otherwise.
   */
  static svtkTypeUInt64 ISO8601ToTimePoint(const char* str, bool* ok = nullptr);

  /**
   * Converts a SVTK timepoint into one of the following ISO8601
   * formats.  The default format is ISO8601_DATETIME_MILLIS.

   * <PRE>
   * Type                      Format / Example
   * 0 ISO8601_DATETIME_MILLIS [YYYY]-[MM]-[DD]T[hh]:[mm]:[ss].[SSS]
   * 2006-01-02T03:04:05.678
   * 1 ISO8601_DATETIME        [YYYY]-[MM]-[DD]T[hh]:[mm]:[ss]
   * 2006-01-02T03:04:05
   * 2 ISO8601_DATE            [YYYY]-[MM]-[DD]
   * 2006-01-02
   * 3 ISO8601_TIME_MILLIS     [hh]:[mm]:[ss].[SSS]
   * 03:04:05.678
   * 4 ISO8601_TIME            [hh]:[mm]:[ss]
   * 03:04:05
   * </PRE>
   */
  static const char* TimePointToISO8601(svtkTypeUInt64, int format = ISO8601_DATETIME_MILLIS);

protected:
  svtkTimePointUtility() {}
  ~svtkTimePointUtility() override {}

private:
  svtkTimePointUtility(const svtkTimePointUtility&) = delete;
  void operator=(const svtkTimePointUtility&) = delete;
};

#endif
