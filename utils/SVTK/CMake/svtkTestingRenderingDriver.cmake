SET(CMAKE_TESTDRIVER_BEFORE_TESTMAIN
"
    svtksys::SystemInformation::SetStackTraceOnError(1);
#ifndef NDEBUG
    svtkFloatingPointExceptions::Enable();
#endif

    // Set defaults
    svtkTestingInteractor::ValidBaseline = \"Use_-V_for_Baseline\";
    svtkTestingInteractor::TempDirectory =
      std::string(\"${_svtk_build_TEST_OUTPUT_DIRECTORY}\");
    svtkTestingInteractor::DataDirectory = std::string(\"Use_-D_for_Data\");

    int interactive = 0;
    for (int ii = 0; ii < ac; ++ii)
      {
      if (strcmp(av[ii], \"-I\") == 0)
        {
        interactive = 1;
        continue;
        }
      if (ii < ac-1 && strcmp(av[ii], \"-V\") == 0)
        {
        svtkTestingInteractor::ValidBaseline = std::string(av[++ii]);
        continue;
        }
      if (ii < ac-1 && strcmp(av[ii], \"-T\") == 0)
        {
        svtkTestingInteractor::TempDirectory = std::string(av[++ii]);
        continue;
        }
      if (ii < ac-1 && strcmp(av[ii], \"-D\") == 0)
        {
        svtkTestingInteractor::DataDirectory = std::string(av[++ii]);
        continue;
        }
      if (ii < ac-1 && strcmp(av[ii], \"-E\") == 0)
        {
        svtkTestingInteractor::ErrorThreshold =
            static_cast<double>(atof(av[++ii]));
        continue;
        }
      if (ii < ac-1 && strcmp(av[ii], \"-v\") == 0)
        {
        svtkLogger::SetStderrVerbosity(static_cast<svtkLogger::Verbosity>(atoi(av[++ii])));
        continue;
        }
      }

    // init logging
    svtkLogger::Init(ac, av, nullptr);

    // turn on windows stack traces if applicable
    svtkWindowsTestUtilitiesSetupForTesting();

    svtkSmartPointer<svtkTestingObjectFactory> factory = svtkSmartPointer<svtkTestingObjectFactory>::New();
    if (!interactive)
      {
      // Disable any other overrides before registering our factory.
      svtkObjectFactoryCollection *collection = svtkObjectFactory::GetRegisteredFactories();
      collection->InitTraversal();
      svtkObjectFactory *f = collection->GetNextItem();
      while (f)
        {
        f->Disable(\"svtkRenderWindowInteractor\");
        f = collection->GetNextItem();
        }
      svtkObjectFactory::RegisterFactory(factory);
      }
"
)

SET(CMAKE_TESTDRIVER_AFTER_TESTMAIN
"
   if (result == SVTK_SKIP_RETURN_CODE)
     {
     printf(\"Unsupported runtime configuration: Test returned \"
            \"SVTK_SKIP_RETURN_CODE. Skipping test.\\n\");
     return result;
     }

   if (!interactive)
     {
     if (svtkTestingInteractor::TestReturnStatus != -1)
        {
        if (svtkTestingInteractor::TestReturnStatus != svtkTesting::PASSED)
          {
          result = EXIT_FAILURE;
          }
        else
          {
          result = EXIT_SUCCESS;
          }
        }
      svtkObjectFactory::UnRegisterFactory(factory);
      }
"
)
