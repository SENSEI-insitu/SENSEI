# This file  attempts to  convert an  old pipeline filter  to a  new pipeline
# filter. Run it with a  -DCLASS=classname it will use that class name
# for processing

IF (NOT DEFINED CLASS)
  MESSAGE ("You did not specify the class to process. Usage: cmake -DCLASS=svtkMyClass -P NewPipeConvert" FATAL_ERROR)
ENDIF ()

FILE (GLOB H_FILE ${CLASS}.h)
FILE (GLOB CXX_FILE ${CLASS}.cxx)

# read in both files
FILE (READ ${H_FILE} H_CONTENTS)
FILE (READ ${CXX_FILE} CXX_CONTENTS)

#================================================================
# First do the H file
#================================================================

# convert svtkImageToImageFilter subclasses to subclass off of
# svtkImageAlgorithm, if it is threaded use threaded one
IF ("${CXX_CONTENTS}" MATCHES ".*ThreadedExecute.*")
  STRING (REGEX REPLACE
    "svtkImageToImageFilter"
    "svtkThreadedImageAlgorithm"
    H_CONTENTS "${H_CONTENTS}")
  STRING (REGEX REPLACE
    "svtkImageTwoInputFilter"
    "svtkThreadedImageAlgorithm"
    H_CONTENTS "${H_CONTENTS}")
ELSE ()
  STRING (REGEX REPLACE
    "svtkImageToImageFilter"
    "svtkImageAlgorithm"
    H_CONTENTS "${H_CONTENTS}")
  STRING (REGEX REPLACE
    "svtkImageSource"
    "svtkImageAlgorithm"
    H_CONTENTS "${H_CONTENTS}")
  STRING (REGEX REPLACE
    "svtkImageTwoInputFilter"
    "svtkImageAlgorithm"
    H_CONTENTS "${H_CONTENTS}")
  STRING (REGEX REPLACE
    "svtkDataSetToImageFilter"
    "svtkImageAlgorithm"
    H_CONTENTS "${H_CONTENTS}")
ENDIF ()


# polyDataAlgorithm
STRING (REGEX REPLACE
  "svtkPolyDataToPolyDataFilter"
  "svtkPolyDataAlgorithm"
  H_CONTENTS "${H_CONTENTS}")

STRING (REGEX REPLACE
  "ExecuteInformation[ \t]*\\([^,\)]*,[^\)]*\\)"
  "ExecuteInformation (svtkInformation *, svtkInformationVector **, svtkInformationVector *)"
  H_CONTENTS "${H_CONTENTS}")

STRING (REGEX REPLACE
  "void ExecuteInformation[ \t]*\\([ \t]*\\)[ \t\n]*{[^}]*};"
  ""
  H_CONTENTS "${H_CONTENTS}")

STRING (REGEX REPLACE
  "ExecuteInformation[ \t]*\\([ \t]*\\)"
  "ExecuteInformation (svtkInformation *, svtkInformationVector **, svtkInformationVector *)"
  H_CONTENTS "${H_CONTENTS}")

STRING (REGEX REPLACE
  "ComputeInputUpdateExtent[ \t]*\\([^,]*,[^,\)]*\\)"
  "RequestUpdateExtent (svtkInformation *, svtkInformationVector **, svtkInformationVector *)"
  H_CONTENTS "${H_CONTENTS}")

FILE (WRITE ${H_FILE} "${H_CONTENTS}")


#================================================================
# Now do the CXX files
#================================================================

STRING (REGEX REPLACE
  "::ExecuteInformation[ \t]*\\([^{]*{"
  "::ExecuteInformation (\n  svtkInformation * svtkNotUsed(request),\n  svtkInformationVector **inputVector,\n  svtkInformationVector *outputVector)\n{"
  CXX_CONTENTS "${CXX_CONTENTS}")

# add outInfo only once
IF (NOT "${CXX_CONTENTS}" MATCHES ".*::ExecuteInformation[^{]*{\n  // get the info objects.*")
  STRING (REGEX REPLACE
    "::ExecuteInformation[ \t]*\\([^{]*{"
    "::ExecuteInformation (\n  svtkInformation * svtkNotUsed(request),\n  svtkInformationVector **inputVector,\n  svtkInformationVector *outputVector)\n{\n  // get the info objects\n  svtkInformation* outInfo = outputVector->GetInformationObject(0);\n  svtkInformation *inInfo = inputVector[0]->GetInformationObject(0);\n"
    CXX_CONTENTS "${CXX_CONTENTS}")
ENDIF ()


STRING (REGEX REPLACE
  "::ComputeInputUpdateExtent[ \t]*\\([^,\)]*,[^,\)]*\\)"
  "::RequestUpdateExtent (\n  svtkInformation * svtkNotUsed(request),\n  svtkInformationVector **inputVector,\n  svtkInformationVector *outputVector)"
  CXX_CONTENTS "${CXX_CONTENTS}")

# add outInfo only once
IF (NOT "${CXX_CONTENTS}" MATCHES ".*::RequestUpdateExtent[^{]*{\n  // get the info objects.*")
  STRING (REGEX REPLACE
    "::RequestUpdateExtent[ \t]*\\([^{]*{"
    "::RequestUpdateExtent (\n  svtkInformation * svtkNotUsed(request),\n  svtkInformationVector **inputVector,\n  svtkInformationVector *outputVector)\n{\n  // get the info objects\n  svtkInformation* outInfo = outputVector->GetInformationObject(0);\n  svtkInformation *inInfo = inputVector[0]->GetInformationObject(0);\n"
    CXX_CONTENTS "${CXX_CONTENTS}")
ENDIF ()

STRING (REGEX REPLACE
  "this->GetInput\\(\\)->GetWholeExtent\\("
  "inInfo->Get(svtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),"
  CXX_CONTENTS "${CXX_CONTENTS}")
STRING (REGEX REPLACE
  "input->GetWholeExtent\\("
  "inInfo->Get(svtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),"
  CXX_CONTENTS "${CXX_CONTENTS}")
STRING (REGEX REPLACE
  "inData->GetWholeExtent\\("
  "inInfo->Get(svtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),"
  CXX_CONTENTS "${CXX_CONTENTS}")
STRING (REGEX REPLACE
  "this->GetOutput\\(\\)->SetWholeExtent[ \t\n]*\\(([^)]*)"
  "outInfo->Set(svtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),\\1,6"
  CXX_CONTENTS "${CXX_CONTENTS}")
STRING (REGEX REPLACE
  "output->SetWholeExtent[ \t\n]*\\(([^)]*)"
  "outInfo->Set(svtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),\\1,6"
  CXX_CONTENTS "${CXX_CONTENTS}")
STRING (REGEX REPLACE
  "outData->SetWholeExtent[ \t\n]*\\(([^)]*)"
  "outInfo->Set(svtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(),\\1,6"
  CXX_CONTENTS "${CXX_CONTENTS}")

STRING (REGEX REPLACE
  "this->GetOutput\\(\\)->SetOrigin[ \t\n]*\\(([^)]*)"
  "outInfo->Set(svtkDataObject::ORIGIN(),\\1,3"
  CXX_CONTENTS "${CXX_CONTENTS}")
STRING (REGEX REPLACE
  "output->SetOrigin[ \t\n]*\\(([^)]*)"
  "outInfo->Set(svtkDataObject::ORIGIN(),\\1,3"
  CXX_CONTENTS "${CXX_CONTENTS}")
STRING (REGEX REPLACE
  "outData->SetOrigin[ \t\n]*\\(([^)]*)"
  "outInfo->Set(svtkDataObject::ORIGIN(),\\1,3"
  CXX_CONTENTS "${CXX_CONTENTS}")

STRING (REGEX REPLACE
  "this->GetOutput\\(\\)->SetSpacing[ \t\n]*\\(([^)]*)"
  "outInfo->Set(svtkDataObject::SPACING(),\\1,3"
  CXX_CONTENTS "${CXX_CONTENTS}")
STRING (REGEX REPLACE
  "output->SetSpacing[ \t\n]*\\(([^)]*)"
  "outInfo->Set(svtkDataObject::SPACING(),\\1,3"
  CXX_CONTENTS "${CXX_CONTENTS}")
STRING (REGEX REPLACE
  "outData->SetSpacing[ \t\n]*\\(([^)]*)"
  "outInfo->Set(svtkDataObject::SPACING(),\\1,3"
  CXX_CONTENTS "${CXX_CONTENTS}")

STRING (REGEX REPLACE
  "output->SetScalarType[ \t\n]*\\(([^)]*)"
  "outInfo->Set(svtkDataObject::SCALAR_TYPE(),\\1"
  CXX_CONTENTS "${CXX_CONTENTS}")
STRING (REGEX REPLACE
  "outData->SetScalarType[ \t\n]*\\(([^)]*)"
  "outInfo->Set(svtkDataObject::SCALAR_TYPE(),\\1"
  CXX_CONTENTS "${CXX_CONTENTS}")
STRING (REGEX REPLACE
  "output->SetNumberOfScalarComponents[ \t\n]*\\(([^)]*)"
  "outInfo->Set(svtkDataObject::SCALAR_NUMBER_OF_COMPONENTS(),\\1"
  CXX_CONTENTS "${CXX_CONTENTS}")
STRING (REGEX REPLACE
  "outData->SetNumberOfScalarComponents[ \t\n]*\\(([^)]*)"
  "outInfo->Set(svtkDataObject::SCALAR_NUMBER_OF_COMPONENTS(),\\1"
  CXX_CONTENTS "${CXX_CONTENTS}")

# add some useful include files if needed
IF ("${CXX_CONTENTS}" MATCHES ".*svtkInformation.*")
  # do not do these replacements multiple times
  IF (NOT "${CXX_CONTENTS}" MATCHES ".*svtkInformation.h.*")
    STRING (REGEX REPLACE
      "svtkObjectFactory.h"
      "svtkInformation.h\"\n#include \"svtkInformationVector.h\"\n#include \"svtkObjectFactory.h\"\n#include \"svtkStreamingDemandDrivenPipeline.h"
      CXX_CONTENTS "${CXX_CONTENTS}")
  ENDIF ()
ENDIF ()

FILE (WRITE ${CXX_FILE} "${CXX_CONTENTS}")
