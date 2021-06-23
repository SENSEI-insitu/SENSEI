# This file  attempts to  convert an  old pipeline filter  to a  new pipeline
# filter. Run it with a  -DCLASS=classname it will use that class name
# for processing

IF (NOT DEFINED CLASS)
  MESSAGE ("You did not specify the class to process. Usage: cmake -DCLASS=svtkMyClass -P NewPipeConvertPointSet" FATAL_ERROR)
ENDIF ()

FILE (GLOB H_FILE ${CLASS}.h)
FILE (GLOB CXX_FILE ${CLASS}.cxx)

# read in both files
FILE (READ ${H_FILE} H_CONTENTS)
FILE (READ ${CXX_FILE} CXX_CONTENTS)

#================================================================
# First do the H file
#================================================================

STRING (REGEX REPLACE
  "svtkPointSetToPointSetFilter"
  "svtkPointSetAlgorithm"
  H_CONTENTS "${H_CONTENTS}")

STRING (REGEX REPLACE
  "void[ \t]+Execute[ \t]*\\([ \t]*\\)"
  "int RequestData(svtkInformation *, svtkInformationVector **, svtkInformationVector *)"
  H_CONTENTS "${H_CONTENTS}")

STRING (REGEX REPLACE
  "void[ \t]+ExecuteInformation[ \t]*\\([ \t]*\\)"
  "int RequestInformation(svtkInformation *, svtkInformationVector **, svtkInformationVector *)"
  H_CONTENTS "${H_CONTENTS}")

STRING (REGEX REPLACE
  "void[ \t]+ComputeInputUpdateExtents[ \t]*\\([ \t]*[^)]*\\)"
  "int RequestUpdateExtent(svtkInformation *, svtkInformationVector **, svtkInformationVector *)"
  H_CONTENTS "${H_CONTENTS}")

FILE (WRITE ${H_FILE} "${H_CONTENTS}")

#================================================================
# Now do the CXX files
#================================================================

STRING (REGEX REPLACE
  "::Execute[ \t]*\\([^{]*{"
  "::RequestData(\n  svtkInformation *svtkNotUsed(request),\n  svtkInformationVector **inputVector,\n  svtkInformationVector *outputVector)\n{\n  // get the info objects\n  svtkInformation *inInfo = inputVector[0]->GetInformationObject(0);\n  svtkInformation *outInfo = outputVector->GetInformationObject(0);\n\n  // get the input and output\n  svtkPointSet *input = svtkPointSet::SafeDownCast(\n    inInfo->Get(svtkDataObject::DATA_OBJECT()));\n  svtkPointSet *output = svtkPointSet::SafeDownCast(\n    outInfo->Get(svtkDataObject::DATA_OBJECT()));\n"
  CXX_CONTENTS "${CXX_CONTENTS}")

STRING (REGEX REPLACE
  "::ExecuteInformation[ \t]*\\([^{]*{"
  "::RequestInformation(\n  svtkInformation *svtkNotUsed(request),\n  svtkInformationVector **inputVector,\n  svtkInformationVector *outputVector)\n{\n  // get the info objects\n  svtkInformation *inInfo = inputVector[0]->GetInformationObject(0);\n  svtkInformation *outInfo = outputVector->GetInformationObject(0);\n"
  CXX_CONTENTS "${CXX_CONTENTS}")

STRING (REGEX REPLACE
  "::ComputeInputUpdateExtents[ \t]*\\([^{]*{"
  "::RequestUpdateExtent(\n  svtkInformation *svtkNotUsed(request),\n  svtkInformationVector **inputVector,\n  svtkInformationVector *outputVector)\n{\n  // get the info objects\n  svtkInformation *inInfo = inputVector[0]->GetInformationObject(0);\n  svtkInformation *outInfo = outputVector->GetInformationObject(0);\n"
  CXX_CONTENTS "${CXX_CONTENTS}")

# add some useful include files if needed
IF ("${CXX_CONTENTS}" MATCHES ".*svtkInformation.*")
  # do not do these replacements multiple times
  IF (NOT "${CXX_CONTENTS}" MATCHES ".*svtkInformation.h.*")
    STRING (REGEX REPLACE
      "svtkObjectFactory.h"
      "svtkInformation.h\"\n#include \"svtkInformationVector.h\"\n#include \"svtkObjectFactory.h"
      CXX_CONTENTS "${CXX_CONTENTS}")
  ENDIF ()
ENDIF ()

FILE (WRITE ${CXX_FILE} "${CXX_CONTENTS}")
