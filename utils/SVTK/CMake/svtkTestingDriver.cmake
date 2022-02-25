SET(CMAKE_TESTDRIVER_BEFORE_TESTMAIN
"
  svtksys::SystemInformation::SetStackTraceOnError(1);

  // turn on windows stack traces if applicable
  svtkWindowsTestUtilitiesSetupForTesting();

  // init logging.
  svtkLogger::Init(ac, av);
")

SET(CMAKE_TESTDRIVER_AFTER_TESTMAIN "")
