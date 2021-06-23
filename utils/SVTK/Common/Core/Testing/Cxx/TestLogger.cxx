/*=========================================================================

  Program:   Visualization Toolkit
  Module:    TestLogger.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
// Test some generic features of svtkLogger.

#include "svtkLogger.h"
#include "svtkObject.h"

#include <string>
#include <vector>

namespace
{
void log_handler(void* user_data, const svtkLogger::Message& message)
{
  auto lines = reinterpret_cast<std::string*>(user_data);
  (*lines) += "\n";
  (*lines) += message.message;
}
}

int TestLogger(int, char*[])
{
  std::string lines;
  svtkLogF(INFO, "changing verbosity to %d", svtkLogger::VERBOSITY_TRACE);
  svtkLogger::AddCallback("sonnet-grabber", log_handler, &lines, svtkLogger::VERBOSITY_2);
  svtkLogger::SetStderrVerbosity(svtkLogger::VERBOSITY_TRACE);
  svtkLogScopeFunction(TRACE);
  {
    svtkLogScopeF(TRACE, "Sonnet 18");
    auto whom = "thee";
    svtkLog(2, "Shall I compare " << whom << " to a summer's day?");

    auto what0 = "lovely";
    auto what1 = "temperate";
    svtkLogF(2, "Thou art more %s and more %s:", what0, what1);

    auto month = "May";
    svtkLogIf(2, true, << "Rough winds do shake the darling buds of " << month << ",");
    svtkLogIfF(2, true, "And %s’s lease hath all too short a date;", "summers");
  }

  cerr << "--------------------------------------------" << endl
       << lines << endl
       << endl
       << "--------------------------------------------" << endl;

  svtkGenericWarningMacro("testing generic warning -- should only show up in the log");

  // remove callback since the user-data becomes invalid out of this function.
  svtkLogger::RemoveCallback("sonnet-grabber");

  // test out explicit scope start and end markers.
  {
    svtkLogStartScope(INFO, "scope-0");
  }
  svtkLogStartScopeF(INFO, "scope-1", "scope %d", 1);
  svtkLog(INFO, "some text");
  svtkLogEndScope("scope-1");
  {
    svtkLogEndScope("scope-0");
  }
  return EXIT_SUCCESS;
}
