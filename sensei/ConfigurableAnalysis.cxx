#include <svtkObjectFactory.h>
#include <svtkSmartPointer.h>
#include <svtkNew.h>
#include <svtkDataObject.h>

#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <errno.h>

#include "ConfigurableAnalysis.h"
#include "senseiConfig.h"
#include "Error.h"
#include "Profiler.h"
#include "SVTKUtils.h"
#include "XMLUtils.h"
#include "STLUtils.h"
#include "DataRequirements.h"

#include "Autocorrelation.h"
#include "Histogram.h"
#ifdef ENABLE_VTK_IO
#include "VTKPosthocIO.h"
#ifdef ENABLE_VTK_MPI
#include "VTKAmrWriter.h"
#endif
#endif
#ifdef ENABLE_VTK_ACCELERATORS
#include "VTKmContourAnalysis.h"
#endif
#ifdef ENABLE_VTKM
#include "VTKmVolumeReductionAnalysis.h"
#include "VTKmCDFAnalysis.h"
#endif
#ifdef ENABLE_ADIOS1
#include "ADIOS1AnalysisAdaptor.h"
#endif
#ifdef ENABLE_ADIOS2
#include "ADIOS2AnalysisAdaptor.h"
#endif
#ifdef ENABLE_HDF5
#include "HDF5AnalysisAdaptor.h"
#endif
#ifdef ENABLE_CATALYST
#include "CatalystAnalysisAdaptor.h"
#include "CatalystParticle.h"
#include "CatalystSlice.h"
#include <vtkNew.h>
#endif
#ifdef ENABLE_ASCENT
#include "AscentAnalysisAdaptor.h"
#endif
#ifdef ENABLE_LIBSIM
#include "LibsimAnalysisAdaptor.h"
#include "LibsimImageProperties.h"
#endif
#ifdef ENABLE_OSPRAY
#include "OSPRayAnalysisAdaptor.h"
#endif
#ifdef ENABLE_PYTHON
#include "PythonAnalysis.h"
#endif
#if defined(ENABLE_VTK_IO) && defined(ENABLE_VTK_FILTERS)
#define ENABLE_SLICE_EXTRACT
#include "SliceExtract.h"
#endif
#if defined(ENABLE_VTK_FILTERS)
#include "Calculator.h"
#endif

using AnalysisAdaptorPtr = svtkSmartPointer<sensei::AnalysisAdaptor>;
using AnalysisAdaptorVector = std::vector<AnalysisAdaptorPtr>;

namespace sensei
{
using namespace STLUtils; // for operator<< overloads


struct ConfigurableAnalysis::InternalsType
{
  InternalsType()
    : Comm(MPI_COMM_NULL)
  {
  }

  // Initializes the adaptor by calling the initializer functor,
  // optionally timing how long initialization takes.
  // If no \a initializer is passed, then no initialization is
  // required but a timer entry will be created for consistency.
  int TimeInitialization(
    AnalysisAdaptorPtr adaptor,
    std::function<int()> initializer = []() { return 0; });

  // creates, initializes from xml, and adds the analysis
  // if it has been compiled into the build and is enabled.
  // a status message indicating success/failure is printed
  // by rank 0
  int AddHistogram(pugi::xml_node node);
  int AddVTKmContour(pugi::xml_node node);
  int AddVTKmVolumeReduction(pugi::xml_node node);
  int AddVTKmCDF(pugi::xml_node node);
  int AddAdios1(pugi::xml_node node);
  int AddAdios2(pugi::xml_node node);
  int AddHDF5(pugi::xml_node node);
  int AddAscent(pugi::xml_node node);
  int AddCatalyst(pugi::xml_node node);
  int AddLibsim(pugi::xml_node node);
  int AddAutoCorrelation(pugi::xml_node node);
  int AddOSPRay(pugi::xml_node node);
  int AddPosthocIO(pugi::xml_node node);
  int AddVTKAmrWriter(pugi::xml_node node);
  int AddPythonAnalysis(pugi::xml_node node);
  int AddSliceExtract(pugi::xml_node node);
  int AddCalculator(pugi::xml_node node);

public:
  // list of all analyses. api calls are forwareded to each
  // analysis in the list
  AnalysisAdaptorVector Analyses;

  // special analyses. these apear in the above list, however
  // they require special treatment which is simplified by
  // storing an additional pointer.
#ifdef ENABLE_LIBSIM
  svtkSmartPointer<LibsimAnalysisAdaptor> LibsimAdaptor;
#endif
#ifdef ENABLE_CATALYST
  svtkSmartPointer<CatalystAnalysisAdaptor> CatalystAdaptor;
#endif

  // the communicator that is used to initialize new analyses.
  // When this is MPI_COMM_NULL, the default, each analysis uses
  // it's default, a duplicate of COMM_WORLD. thus if the user
  // doesn't set a communicator correct behavior is insured
  // and superfluous Comm_dup's are avoided.
  MPI_Comm Comm;

  std::vector<std::string> LogEventNames;
};

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::TimeInitialization(
  AnalysisAdaptorPtr adaptor, std::function<int()> initializer)
{
  const char* analysisName = nullptr;
  bool logEnabled = Profiler::Enabled();
  if (logEnabled)
    {
    std::ostringstream initName;
    std::ostringstream execName;
    std::ostringstream finiName;
    auto analysisNumber = this->Analyses.size();
    initName << adaptor->GetClassName() << "::" << analysisNumber << "::Initialize";
    execName << adaptor->GetClassName() << "::" << analysisNumber << "::Execute";
    finiName << adaptor->GetClassName() << "::" << analysisNumber << "::Finalize";
    this->LogEventNames.push_back(initName.str());
    this->LogEventNames.push_back(execName.str());
    this->LogEventNames.push_back(finiName.str());
    analysisName = this->LogEventNames[3 * analysisNumber].c_str();
    Profiler::StartEvent(analysisName);
    }

  int result = initializer();

  if (logEnabled)
    Profiler::EndEvent(analysisName);

  return result;
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddHistogram(pugi::xml_node node)
{
  if (XMLUtils::RequireAttribute(node, "mesh") || XMLUtils::RequireAttribute(node, "array"))
    {
    SENSEI_ERROR("Failed to initialize Histogram");
    return -1;
    }

  int association = 0;
  std::string assocStr = node.attribute("association").as_string("point");
  if (SVTKUtils::GetAssociation(assocStr, association))
    {
    SENSEI_ERROR("Failed to initialize Histogram");
    return -1;
    }

  std::string mesh = node.attribute("mesh").value();
  std::string array = node.attribute("array").value();
  int bins = node.attribute("bins").as_int(10);
  std::string fileName = node.attribute("file").value();

  auto histogram = svtkSmartPointer<Histogram>::New();

  if (this->Comm != MPI_COMM_NULL)
    histogram->SetCommunicator(this->Comm);

  this->TimeInitialization(histogram, [&]() {
      histogram->Initialize(bins, mesh, association, array, fileName);
      return 0;
    });
  this->Analyses.push_back(histogram.GetPointer());

  SENSEI_STATUS("Configured histogram with " << bins
    << " bins on " << assocStr << " data array \"" << array
    << "\" on mesh \"" << mesh << "\" writing output to "
    << (fileName.empty() ? "cout" : "file"))

  return 0;
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddVTKmContour(pugi::xml_node node)
{
#ifndef ENABLE_VTK_ACCELERATORS
  (void)node;
  SENSEI_ERROR("svtkAcceleratorsVtkm was requested but is disabled in this build")
  return -1;
#else

  if (XMLUtils::RequireAttribute(node, "mesh") || XMLUtils::RequireAttribute(node, "array"))
    {
    SENSEI_ERROR("Failed to initialize VTKmContourAnalysis");
    return -1;
    }

  auto mesh = node.attribute("mesh");
  auto array = node.attribute("array");

  double value = node.attribute("value").as_double(0.0);
  bool writeOutput = node.attribute("write_output").as_bool(false);

  auto contour = svtkSmartPointer<VTKmContourAnalysis>::New();

  if (this->Comm != MPI_COMM_NULL)
    contour->SetCommunicator(this->Comm);

  this->TimeInitialization(contour, [&]() {
    contour->Initialize(mesh.value(), array.value(), value, writeOutput);
    return 0;
    });
  this->Analyses.push_back(contour.GetPointer());

  SENSEI_STATUS("Configured VTKmContourAnalysis " << array.value())

  return 0;
#endif
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddVTKmVolumeReduction(pugi::xml_node node)
{
#ifndef ENABLE_VTKM
  (void)node;
  SENSEI_ERROR("VTK-m analysis was requested but is disabled in this build")
  return -1;
#else
  if (XMLUtils::RequireAttribute(node, "mesh") || XMLUtils::RequireAttribute(node, "field") ||
    XMLUtils::RequireAttribute(node, "association") || XMLUtils::RequireAttribute(node, "reduction"))
    {
    SENSEI_ERROR("Failed to initialize VTKmVolumeReductionAnalysis");
    return -1;
    }

  auto mesh = node.attribute("mesh").as_string();
  auto field = node.attribute("field").as_string();
  auto assoc = node.attribute("association").as_string();
  auto reduction = node.attribute("reduction").as_int();

  std::string workDir = node.attribute("working-directory").as_string(".");

  auto reducer = svtkSmartPointer<VTKmVolumeReductionAnalysis>::New();
  this->TimeInitialization(reducer, [&]() {
    reducer->Initialize(mesh, field, assoc, workDir, reduction, this->Comm);
    return 0;
  });
  this->Analyses.push_back(reducer.GetPointer());

  SENSEI_STATUS("Configured VTKmVolumeReductionAnalysis " << mesh << "/" << field)

  return 0;
#endif
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddVTKmCDF(pugi::xml_node node)
{
#ifndef ENABLE_VTKM
  (void)node;
  SENSEI_ERROR("VTK-m analysis was requested but is disabled in this build")
  return -1;
#else
  if (
    XMLUtils::RequireAttribute(node, "mesh") ||
    XMLUtils::RequireAttribute(node, "field") ||
    XMLUtils::RequireAttribute(node, "association")
    )
    {
    SENSEI_ERROR("Failed to initialize VTKmCDFAnalysis");
    return -1;
    }

  auto mesh = node.attribute("mesh").as_string();
  auto field = node.attribute("field").as_string();
  auto assoc = node.attribute("association").as_string();
  auto quantiles = node.attribute("quantiles").as_int(10);
  auto exchangeSize = node.attribute("exchange-size").as_int(quantiles);

  bool haveWorkDir = !!node.attribute("working-directory");
  std::string workDir =  haveWorkDir ? node.attribute("working-directory").as_string() : ".";

  auto analysis = svtkSmartPointer<VTKmCDFAnalysis>::New();
  this->TimeInitialization(analysis, [&]() {
    analysis->Initialize(mesh, field, assoc, workDir, quantiles, exchangeSize, this->Comm);
    return 0;
  });
  this->Analyses.push_back(analysis.GetPointer());

  SENSEI_STATUS("Configured VTKmCDFAnalysis " << mesh << "/" << field)

  return 0;
#endif
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddAscent(pugi::xml_node node)
{
#ifndef ENABLE_ASCENT
  (void)node;
  SENSEI_ERROR("Ascent was requested but is disabled in this build")
  return( -1 );
#else
  svtkNew<AscentAnalysisAdaptor> ascent;

  if (this->Comm != MPI_COMM_NULL)
    ascent->SetCommunicator(this->Comm);

  // get the data requirements for this run
  DataRequirements req;
  if (req.Initialize(node) || req.Empty())
    {
    SENSEI_ERROR("Failed to initialize the AscentAnalysisAdaptor."
      << " Missing data requirements.")
    return -1;
    }
  ascent->SetDataRequirements(req);

  // get the json files used to configure ascent
  std::string options_file;
  std::string actions_file;

  // Check if the xml file has the ascent options filename.
  if(node.attribute("options"))
    options_file = node.attribute("options").value();

  // Check if the xml file has the ascent actions filename.
  if (XMLUtils::RequireAttribute(node, "actions"))
    {
    SENSEI_ERROR("Failed to initialize the AscentAnalysisAdaptor");
    return -1;
    }

  actions_file = node.attribute("actions").value();

  if (ascent->Initialize(actions_file, options_file))
    {
    SENSEI_ERROR("Failed to initialize ascent using the actions \""
      << actions_file << "\" and options \"" << options_file << "\"")
    return -1;
    }

  this->Analyses.push_back(ascent.GetPointer());

  SENSEI_STATUS("Configured the AscentAnalysisAdaptor with actions \""
    << actions_file << "\" and options \"" << options_file << "\"")

  return 0;
#endif
}


// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddAdios1(pugi::xml_node node)
{
#ifndef ENABLE_ADIOS1
  (void)node;
  SENSEI_ERROR("ADIOS 1 was requested but is disabled in this build")
  return -1;
#else
  auto adios = svtkSmartPointer<ADIOS1AnalysisAdaptor>::New();

  if (this->Comm != MPI_COMM_NULL)
    adios->SetCommunicator(this->Comm);

  pugi::xml_attribute filename = node.attribute("filename");
  if (filename)
    adios->SetFileName(filename.value());

  pugi::xml_attribute method = node.attribute("method");
  if (method)
    adios->SetMethod(method.value());

  unsigned long maxBufSize =
    node.attribute("max_buffer_size").as_ullong(0);
  adios->SetMaxBufferSize(maxBufSize);

  DataRequirements req;
  if (req.Initialize(node))
    {
    SENSEI_ERROR("Failed to initialize ADIOS 1.")
    return -1;
    }
  adios->SetDataRequirements(req);

  this->TimeInitialization(adios);
  this->Analyses.push_back(adios.GetPointer());

  SENSEI_STATUS("Configured ADIOSAnalysisAdaptor filename=\""
    << filename.value() << "\" method " << method.value()
    << " max_buffer_size=" << maxBufSize)

  return 0;
#endif
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddAdios2(pugi::xml_node node)
{
#ifndef ENABLE_ADIOS2
  (void)node;
  SENSEI_ERROR("ADIOS 2 was requested but is disabled in this build")
  return -1;
#else
  auto adiosAdaptor = svtkSmartPointer<ADIOS2AnalysisAdaptor>::New();

  if (this->Comm != MPI_COMM_NULL)
    adiosAdaptor->SetCommunicator(this->Comm);

  if (adiosAdaptor->Initialize(node))
    {
    SENSEI_ERROR("Failed to configure the ADIOS2 adaptor from XML")
    return -1;
    }

  unsigned int frequency = node.attribute("frequency").as_uint(0);
  adiosAdaptor->SetFrequency(frequency);

  this->TimeInitialization(adiosAdaptor);
  this->Analyses.push_back(adiosAdaptor.GetPointer());

  return 0;
#endif
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddHDF5(pugi::xml_node node)
{
#ifndef ENABLE_HDF5
  (void)node;
  SENSEI_ERROR("HDF5 was requested but is disabled in this build");
  return -1;
#else
  auto dataE = svtkSmartPointer<HDF5AnalysisAdaptor>::New();

  if(this->Comm != MPI_COMM_NULL)
    dataE->SetCommunicator(this->Comm);

  pugi::xml_attribute filename = node.attribute("filename");
  pugi::xml_attribute methodAttr = node.attribute("method");

  if(filename)
    dataE->SetStreamName(filename.value());

  if(methodAttr)
    {
      std::string method = methodAttr.value();

      if(method.size() > 0)
        {
          bool doStreaming = ('s' == method[0]);
          bool doCollectiveTxf = ((method.size() > 1) && ('c' == method[1]));

          dataE->SetStreaming(doStreaming);
          dataE->SetCollective(doCollectiveTxf);
        }
    }

  DataRequirements req;
  if (req.Initialize(node))
    {
    SENSEI_ERROR("Failed to initialize HDF5 Transport.")
    return -1;
    }
  dataE->SetDataRequirements(req);


  this->TimeInitialization(dataE);
  this->Analyses.push_back(dataE.GetPointer());

  SENSEI_STATUS("Configured HDF5 AnalysisAdaptor \"" << filename.value())

  return 0;
#endif
}


// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddCatalyst(pugi::xml_node node)
{
#ifndef ENABLE_CATALYST
  (void)node;
  SENSEI_ERROR("Catalyst was requested but is disabled in this build")
  return -1;
#else
  // a single adaptor used with multiple pipelines
  if (!this->CatalystAdaptor)
    {
    this->CatalystAdaptor = svtkSmartPointer<CatalystAnalysisAdaptor>::New();

    if (this->Comm != MPI_COMM_NULL)
      this->CatalystAdaptor->SetCommunicator(this->Comm);

    this->TimeInitialization(this->CatalystAdaptor);
    this->Analyses.push_back(this->CatalystAdaptor);
    }

  // Add plugin xmls.
  if (node.attribute("plugin_xml"))
    {
    this->CatalystAdaptor->AddPluginXML(node.attribute("plugin_xml").value());
    }

  // Add the pipelines
  if (strcmp(node.attribute("pipeline").value(), "slice") == 0)
    {
    vtkNew<CatalystSlice> slice;

    double tmp[3];
    if (node.attribute("mesh"))
      {
      slice->SetInputMesh(node.attribute("mesh").value());
      }
    if (node.attribute("slice-normal") &&
      (std::sscanf(node.attribute("slice-normal").value(),
      "%lg,%lg,%lg", &tmp[0], &tmp[1], &tmp[2]) == 3))
      {
      slice->SetSliceNormal(tmp[0], tmp[1], tmp[2]);
      }

    if (node.attribute("slice-origin") &&
      (std::sscanf(node.attribute("slice-origin").value(),
      "%lg,%lg,%lg", &tmp[0], &tmp[1], &tmp[2]) == 3))
      {
      slice->SetSliceOrigin(tmp[0], tmp[1], tmp[2]);
      slice->SetAutoCenter(false);
      }
    else
      {
      slice->SetAutoCenter(true);
      }

    int association = 0;
    std::string assocStr = node.attribute("association").as_string("point");
    if (SVTKUtils::GetAssociation(assocStr, association))
      {
      SENSEI_ERROR("Failed to initialize Catalyst")
      return -1;
      }

    slice->ColorBy(association, node.attribute("array").value());
    if (node.attribute("color-range") &&
      (std::sscanf(node.attribute("color-range").value(), "%lg,%lg", &tmp[0], &tmp[1]) == 2))
      {
      slice->SetAutoColorRange(false);
      slice->SetColorRange(tmp[0], tmp[1]);
      }
    else
      {
      slice->SetAutoColorRange(true);
      }

    slice->SetUseLogScale(node.attribute("color-log").as_int(0) == 1);
    if (node.attribute("image-filename"))
      {
      slice->SetImageParameters(
        node.attribute("image-filename").value(),
        node.attribute("image-width").as_int(800),
        node.attribute("image-height").as_int(800));
      }

    this->CatalystAdaptor->AddPipeline(slice.GetPointer());
    }
  else if (strcmp(node.attribute("pipeline").value(), "particle") == 0)
    {
    vtkNew<CatalystParticle> particle;

    double tmp[3];
    if (node.attribute("mesh"))
      {
      particle->SetInputMesh(node.attribute("mesh").value());
      }
    if (node.attribute("particle-style"))
      {
      particle->SetParticleStyle(node.attribute("particle-style").value());
      }
    if (node.attribute("particle-radius") &&
      (std::sscanf(node.attribute("particle-radius").value(),
      "%lg", &tmp[0]) == 1))
      {
      particle->SetParticleRadius(tmp[0]);
      }
    if (node.attribute("camera-position") &&
      (std::sscanf(node.attribute("camera-position").value(),
      "%lg,%lg,%lg", &tmp[0], &tmp[1], &tmp[2]) == 3))
      {
      particle->SetCameraPosition(tmp);
      }

    if (node.attribute("camera-focus") &&
      (std::sscanf(node.attribute("camera-focus").value(),
      "%lg,%lg,%lg", &tmp[0], &tmp[1], &tmp[2]) == 3))
      {
      particle->SetCameraFocus(tmp);
      }

    int association = 0;
    std::string assocStr = node.attribute("association").as_string("point");
    if (SVTKUtils::GetAssociation(assocStr, association))
      {
      SENSEI_ERROR("Failed to initialize Catalyst")
      return -1;
      }

    particle->ColorBy(association, node.attribute("array").value());
    if (node.attribute("color-range") &&
      (std::sscanf(node.attribute("color-range").value(), "%lg,%lg", &tmp[0], &tmp[1]) == 2))
      {
      particle->SetAutoColorRange(false);
      particle->SetColorRange(tmp[0], tmp[1]);
      }
    else
      {
      particle->SetAutoColorRange(true);
      }

    particle->SetUseLogScale(node.attribute("color-log").as_int(0) == 1);
    if (node.attribute("image-filename"))
      {
      particle->SetImageParameters(
        node.attribute("image-filename").value(),
        node.attribute("image-width").as_int(800),
        node.attribute("image-height").as_int(800));
      }

    this->CatalystAdaptor->AddPipeline(particle.GetPointer());
    }
  else if (strcmp(node.attribute("pipeline").value(), "pythonscript") == 0)
    {
#ifndef ENABLE_CATALYST_PYTHON
    SENSEI_ERROR("Catalyst Python was requested but is disabled in this build")
#else
    if (node.attribute("filename"))
      {
      std::string fileName = node.attribute("filename").value();
      int scriptVersion = node.attribute("versionhint").as_int(2);

      std::string producer, mesh, steerable_source_type;
      if (auto resultnode = node.child("result"))
      {
        producer = resultnode.attribute("producer").as_string();
        steerable_source_type = resultnode.attribute("steerable_source_type").as_string();
        mesh = resultnode.attribute("mesh").as_string();
      }

      this->CatalystAdaptor->AddPythonScriptPipeline(fileName,
          producer, steerable_source_type, mesh, scriptVersion);
      }

    unsigned int frequency = node.attribute("frequency").as_uint(0);
    this->CatalystAdaptor->SetFrequency(frequency);

#endif
    }
  SENSEI_STATUS("Configured CatalystAnalysisAdaptor "
    << node.attribute("pipeline").value() << " "
    << (node.attribute("filename") ? node.attribute("filename").value() : ""))
#endif
  return 0;
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddLibsim(pugi::xml_node node)
{
#ifndef ENABLE_LIBSIM
  (void)node;
  SENSEI_ERROR("Libsim was requested but is disabled in this build")
  return -1;
#else

  // We keep around a single instance of the libsim adaptor and then tell it to
  // do different things.
  if (!this->LibsimAdaptor)
    {
    this->LibsimAdaptor = svtkSmartPointer<LibsimAnalysisAdaptor>::New();

    if (this->Comm != MPI_COMM_NULL)
      this->LibsimAdaptor->SetCommunicator(this->Comm);

    if(node.attribute("trace"))
      this->LibsimAdaptor->SetTraceFile(node.attribute("trace").value());

    if(node.attribute("options"))
      this->LibsimAdaptor->SetOptions(node.attribute("options").value());

    if(node.attribute("visitdir"))
      this->LibsimAdaptor->SetVisItDirectory(node.attribute("visitdir").value());

    if(node.attribute("mode"))
      this->LibsimAdaptor->SetMode(node.attribute("mode").value());

    this->LibsimAdaptor->SetComputeNesting(
      node.attribute("compute_nesting").as_int(0));

    this->TimeInitialization(this->LibsimAdaptor, [&]()
      {
      this->LibsimAdaptor->Initialize();
      return 0;
      });

    this->Analyses.push_back(this->LibsimAdaptor);
    }

  bool doExport = false;
  if(node.attribute("operation") != NULL)
    doExport = (node.attribute("operation").value() == std::string("export"));

  int frequency = 5;
  if(node.attribute("frequency") != NULL)
    frequency = node.attribute("frequency").as_int();

  if(frequency < 1)
    frequency = 1;

  std::string filename;
  LibsimImageProperties imageProps;
  if(node.attribute("filename") != NULL)
    {
    imageProps.SetFilename(node.attribute("filename").value());
    filename = node.attribute("filename").value();
    }
  if(node.attribute("image-filename") != NULL)
    {
    imageProps.SetFilename(node.attribute("image-filename").value());
    filename = node.attribute("image-filename").value();
    }

  if(node.attribute("image-width") != NULL)
    imageProps.SetWidth(node.attribute("image-width").as_int());

  if(node.attribute("image-height") != NULL)
    imageProps.SetHeight(node.attribute("image-height").as_int());

  std::string plots, plotVars;
  double origin[3] = {0.,0.,0.};
  double normal[3] = {1.,0.,0.};
  bool slice = false, project = false;

  std::string session;
  if(node.attribute("session") != NULL)
    session = node.attribute("session").value();

  if(node.attribute("plots") != NULL)
    plots = node.attribute("plots").value();

  if(node.attribute("plotvars") != NULL)
    plotVars = node.attribute("plotvars").value();

  if(node.attribute("slice-origin") != NULL)
    {
    double tmp[3];
    if(sscanf(node.attribute("slice-origin").value(),
       "%lg,%lg,%lg", &tmp[0], &tmp[1], &tmp[2]) == 3)
      {
      slice = true;
      origin[0] = tmp[0];
      origin[1] = tmp[1];
      origin[2] = tmp[2];
      }
    }
  if(node.attribute("slice-normal") != NULL)
    {
    double tmp[3];
    if(sscanf(node.attribute("slice-normal").value(),
      "%lg,%lg,%lg", &tmp[0], &tmp[1], &tmp[2]) == 3)
    {
      slice = true;
      normal[0] = tmp[0];
      normal[1] = tmp[1];
      normal[2] = tmp[2];
      }
    }

  if(node.attribute("slice-project") != NULL)
    project = node.attribute("slice-project").as_int() != 0;

  if(doExport)
  {
    // Add the export that we want to make.
    if(!this->LibsimAdaptor->AddExport(frequency, session, plots, plotVars,
      slice, project, origin, normal, filename))
      return -2;

    SENSEI_STATUS("configured LibsimAnalysisAdaptor export")
  }
  else
  {
    // Add the image that we want to make.
    if(!this->LibsimAdaptor->AddRender(frequency, session, plots, plotVars,
      slice, project, origin, normal, imageProps))
      return -2;

    SENSEI_STATUS("configured LibsimAnalysisAdaptor render")
  }

#endif
  return 0;
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddAutoCorrelation(pugi::xml_node node)
{
  if (XMLUtils::RequireAttribute(node, "mesh") || XMLUtils::RequireAttribute(node, "array"))
    {
    SENSEI_ERROR("Failed to initialize Autocorrelation");
    return -1;
    }

  std::string meshName = node.attribute("mesh").value();
  std::string arrayName = node.attribute("array").value();

  std::string assocStr = node.attribute("association").as_string("point");
  int assoc = 0;
  if (SVTKUtils::GetAssociation(assocStr, assoc))
    {
    SENSEI_ERROR("Failed to initialize Autocorrelation");
    return -1;
    }

  int window = node.attribute("window").as_int(10);
  int kMax = node.attribute("k-max").as_int(3);
  int numThreads = node.attribute("n-threads").as_int(1);

  auto adaptor = svtkSmartPointer<Autocorrelation>::New();

  if (this->Comm != MPI_COMM_NULL)
    adaptor->SetCommunicator(this->Comm);

  this->TimeInitialization(adaptor, [&]() {
    adaptor->Initialize(window, meshName, assoc, arrayName, kMax);
    return 0;
  });

  this->Analyses.push_back(adaptor.GetPointer());

  SENSEI_STATUS("Configured Autocorrelation " << assocStr
    << " data array \"" << arrayName << "\" on mesh \"" << meshName
    << "\" window " << window << " k-max " << kMax
    << " n-threads " << numThreads)

  return 0;
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddPosthocIO(pugi::xml_node node)
{
#ifndef ENABLE_VTK_IO
  (void)node;
  SENSEI_ERROR("VTK I/O was requested but is disabled in this build")
  return -1;
#else
  DataRequirements req;
  if (req.Initialize(node))
    {
    SENSEI_ERROR("Failed to initialize VTKPosthocIO.")
    return -1;
    }

  std::string outputDir = node.attribute("output_dir").as_string("./");
  std::string fileName = node.attribute("file_name").as_string("data");
  std::string mode = node.attribute("mode").as_string("visit");
  std::string writer = node.attribute("writer").as_string("xml");
  std::string ghostArrayName = node.attribute("ghost_array_name").as_string("");
  int verbose = node.attribute("verbose").as_int(0);
  unsigned int frequency = node.attribute("frequency").as_uint(0);

  auto adaptor = svtkSmartPointer<VTKPosthocIO>::New();

  if (this->Comm != MPI_COMM_NULL)
    adaptor->SetCommunicator(this->Comm);

  adaptor->SetGhostArrayName(ghostArrayName);
  adaptor->SetVerbose(verbose);
  adaptor->SetFrequency(frequency);

  if (adaptor->SetOutputDir(outputDir) || adaptor->SetMode(mode) ||
    adaptor->SetWriter(writer) || adaptor->SetDataRequirements(req))
    {
    SENSEI_ERROR("Failed to initialize the VTKPosthocIO analysis")
    return -1;
    }

  this->TimeInitialization(adaptor);
  this->Analyses.push_back(adaptor.GetPointer());

  SENSEI_STATUS("Configured VTKPosthocIO")

  return 0;
#endif
}


// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddOSPRay(pugi::xml_node node)
{
#ifndef ENABLE_OSPRAY
  (void)node;
  SENSEI_ERROR("OSPRay was requested but is disabled in this build")
  return -1;
#else

  struct Colormap {
    std::vector<float> points;
    std::vector<float> RGBPoints;
    std::vector<float> linearColors; // interpolated values from points
    std::vector<float> linearOpacities; // interpolated values from points
    int resolution; // number of colors/opacities
  };
  std::map<std::string, Colormap> colormaps;

  auto parseFloat = [&](std::stringstream& ss) {
    float result = -1.f;
    std::string val;
    std::getline(ss, val, ',');
    std::stringstream ss2(val);
    ss2 >> result;
    return result;
  };

  std::string colormap = "none";

  std::string output_dir = node.attribute("output_dir").value();
  std::string file_name = node.attribute("file_name").value();
  pugi::xml_node meshNode = node.child("mesh");
  if (meshNode.next_sibling("mesh")) {
    SENSEI_ERROR("OSPRay's Adapter only supports 1 mesh");
    return -1;
  }
  std::string meshName = meshNode.attribute("name").value();
  std::string pRadius = meshNode.child_value("radius");
  std::string pColor = meshNode.child_value("color");
  std::string renderAs = meshNode.attribute("render_as").value();
  std::string assoc = meshNode.attribute("array_assoc").value();
  std::string colormapStr = meshNode.attribute("colormap_name").value();
  std::string arrayName = meshNode.attribute("array_name").value();
  std::string useD3String = meshNode.attribute("use_D3").value();
  std::string backgroundColor = node.child_value("backgroundColor");
  auto colormapNodes = node.children("colormap");
  // colormaps are expected to be copied from exported paraview colormaps into
  // corresponding points (val/opacity) and rgb_points (val/r/g/b)
  for (auto colormapNode : colormapNodes) {
    std::string cmName = colormapNode.child_value("name");
    std::string cmPoints = colormapNode.child_value("points");
    std::string cmRGBPoints = colormapNode.child_value("rgb_points");
    std::string cmRange = colormapNode.child_value("range");
    Colormap colormap;
    // get rid of commas
    {
      std::stringstream ss(cmPoints);
      while (ss) {
        colormap.points.emplace_back(parseFloat(ss));
      }
    }
    {
      std::stringstream ss(cmRGBPoints);
      while (ss) {
        colormap.RGBPoints.emplace_back(parseFloat(ss));
      }
    }
    std::stringstream ss3(cmRange);
    double range[2];
    ss3 >> range[0] >> range[1];

    int numColors = 256;
    int numValues = colormap.RGBPoints.size()/4;
    int rgbIdx = 0;
    int aIdx = 0;
    // build colors
    float rangeScale = range[1] - range[0];
    auto& rgbPoints = colormap.RGBPoints;
    auto& points = colormap.points;
    for (int i = 0; i < numColors; i++) {
      float val = float(i) / float(numColors - 1) * rangeScale + range[0];
      int rgbIdx2 = rgbIdx + 1 < numValues ? rgbIdx + 1 : rgbIdx;
      if (rgbPoints[rgbIdx2 * 4] < val) {
        rgbIdx = std::min(numValues - 1, rgbIdx + 1);
        rgbIdx2 = rgbIdx + 1 < numValues ? rgbIdx + 1 : rgbIdx;
      }
      float rgb1[4] = {rgbPoints[rgbIdx * 4 + 0],
        rgbPoints[rgbIdx * 4 + 1],
        rgbPoints[rgbIdx * 4 + 2],
        rgbPoints[rgbIdx * 4 + 3]};
      float rgb2[4] = {rgbPoints[rgbIdx2 * 4 + 0],
        rgbPoints[rgbIdx2 * 4 + 1],
        rgbPoints[rgbIdx2 * 4 + 2],
        rgbPoints[rgbIdx2 * 4 + 3]};
      int aIdx2 = aIdx + 1 < numValues ? aIdx + 1 : aIdx;
      // *4 to skip over the range values paraview seems to insert every other value
      if (points[aIdx2 * 4] < val)
      {
        aIdx = std::min(numValues - 1, aIdx + 1);
        aIdx2 = aIdx + 1 < numValues ? aIdx + 1 : aIdx;
      }
      float a1[2] = {points[aIdx * 4 + 0], points[aIdx * 4 + 1]};
      float a2[2] = {points[aIdx2 * 4 + 0], points[aIdx2 * 4 + 1]};
      float rgba[4];
      float rgbRange = rgb2[0] - rgb1[0];
      if (rgbRange == 0.f)
        rgbRange = 1.f;
      float rgbFrac = 1.f - (val - rgb1[0]) / rgbRange;
      float aRange = a2[0] - a1[0];
      if (aRange == 0.f)
        aRange = 1.f;
      float aFrac = 1.f - (val - a1[0]) / aRange;
      for (int i = 0; i < 3; i++) {
        rgba[i] = rgb1[i + 1] * rgbFrac + rgb2[i + 1] * (1.f - rgbFrac);
      }
      rgba[3] = a1[1] * aFrac + a2[1] * (1.f - aFrac);
      colormap.linearColors.emplace_back(rgba[0]);
      colormap.linearColors.emplace_back(rgba[1]);
      colormap.linearColors.emplace_back(rgba[2]);
      colormap.linearOpacities.emplace_back(rgba[3]);
    }
    colormaps[cmName] = colormap;
  }

  pugi::xml_node cameraNode = node.child("camera");
  OSPRayAnalysisAdaptor::Camera camera;
  bool hasCamera = false;
  if (cameraNode) {
    if (cameraNode.next_sibling("camera")) {
      SENSEI_ERROR("OSPRay's Adapter only supports 1 camera");
      return -1;
    }
    hasCamera = true;
    std::string cPosition = cameraNode.child_value("position");
    std::string cDirection = cameraNode.child_value("direction");
    std::string cUp = cameraNode.child_value("up");
    std::string cFovy = cameraNode.child_value("fovy");
    std::string cFocusDistance = cameraNode.child_value("focusDistance");
    if (cPosition != "") {
      std::stringstream ss(cPosition);
      ss >> camera.Position[0] >> camera.Position[1] >> camera.Position[2];
    }
    if (cDirection != "") {
      std::stringstream ss(cDirection);
      ss >> camera.Direction[0] >> camera.Direction[1] >> camera.Direction[2];
    }
    if (cUp != "") {
      std::stringstream ss(cUp);
      ss >> camera.Up[0] >> camera.Up[1] >> camera.Up[2];
    }
    if (cFovy != "") {
      std::stringstream ss(cFovy);
      camera.Fovy = parseFloat(ss);
    }
    if (cFocusDistance != "") {
      std::stringstream ss(cFovy);
      camera.FocusDistance = parseFloat(ss);
    }
  }

  bool useD3 = useD3String == "True";

  if (colormapStr != "")
    colormap = colormapStr;

  auto adaptor = svtkSmartPointer<OSPRayAnalysisAdaptor>::New();
  adaptor->SetMeshName(meshName.c_str());
  adaptor->SetRenderAs(renderAs.c_str());
  adaptor->SetAssociation(assoc.c_str());
  adaptor->SetArrayName(arrayName.c_str());
  adaptor->SetWidth(512);
  adaptor->SetHeight(512);
  adaptor->SetDirectory(output_dir.c_str());
  adaptor->SetFileName(file_name.c_str());
  adaptor->SetUseD3(useD3);
  if (colormap != "none") {
    if (colormaps.find(colormap) != colormaps.end()) {
      auto& cm = colormaps[colormap];
      adaptor->SetColormap(cm.linearColors, cm.linearOpacities);
    } else
      std::cerr << "ERROR: (ospray config) colormap " << colormap << "not found\n";
  }
  if (pRadius != "") {
    std::stringstream ss(pRadius);
    float radius;
    ss >> radius;
    adaptor->SetParticleRadius(radius);
  }
  if (pColor != "") {
    std::stringstream ss(pColor);
    float color[3];
    ss >> color[0] >> color[1] >> color[2];
    adaptor->SetParticleColor(color);
  }
  if (backgroundColor != "") {
    std::stringstream ss(backgroundColor);
    float color[4];
    ss >> color[0] >> color[1] >> color[2] >> color[3];
    adaptor->SetBackgroundColor(color);
  }

  if (this->Comm != MPI_COMM_NULL) {
    adaptor->SetCommunicator(this->Comm);
  }
  if (hasCamera)
    adaptor->SetCamera(camera);

  this->TimeInitialization(adaptor, [&]() {
    adaptor->Initialize();
    return 0;
  });

  this->Analyses.push_back(adaptor.GetPointer());

  SENSEI_STATUS("Configured OSPRay"
    << " data array \"" << arrayName << "\" on mesh \"" << meshName)

  return 0;
#endif
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddVTKAmrWriter(pugi::xml_node node)
{
#if !defined(ENABLE_VTK_IO) || !defined(ENABLE_VTK_MPI)
  (void)node;
  SENSEI_ERROR("VTK AMR Writer was requested but is disabled in this build")
  return -1;
#else
  DataRequirements req;

  if (req.Initialize(node))
    {
    SENSEI_ERROR("Failed to initialize VTKAmrWriter.")
    return -1;
    }

  std::string outputDir = node.attribute("output_dir").as_string("./");
  std::string fileName = node.attribute("file_name").as_string("data");
  std::string mode = node.attribute("mode").as_string("visit");

  auto adapter = svtkSmartPointer<VTKAmrWriter>::New();

  if (this->Comm != MPI_COMM_NULL)
    adapter->SetCommunicator(this->Comm);

  if (adapter->SetOutputDir(outputDir) || adapter->SetMode(mode) ||
    adapter->SetDataRequirements(req) ||
    this->TimeInitialization(adapter, [&]() { return adapter->Initialize(); }))
    {
    SENSEI_ERROR("Failed to initialize the VTKAmrWriter analysis")
    return -1;
    }

  this->TimeInitialization(adapter);
  this->Analyses.push_back(adapter.GetPointer());

  SENSEI_STATUS("Configured VTKAmrWriter")

  return 0;
#endif
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddPythonAnalysis(pugi::xml_node node)
{
#if !defined(ENABLE_PYTHON)
  (void)node;
  SENSEI_ERROR("The PythonAnalysis was requested but is disabled in this build")
  return -1;
#else
  if (!node.attribute("script_file") && !node.attribute("script_module"))
    {
    SENSEI_ERROR("Failed to initialize PythonAnalysis. Missing "
      "a required attribute: script_file or script_module");
    return -1;
    }

  std::string scriptFile = node.attribute("script_file").value();
  std::string scriptModule = node.attribute("script_module").value();

  std::string initSource;
  pugi::xml_node inode = node.child("initialize_source");
  if (inode)
    initSource = inode.text().as_string();

  auto pyAnalysis = svtkSmartPointer<PythonAnalysis>::New();

  if (this->Comm != MPI_COMM_NULL)
    pyAnalysis->SetCommunicator(this->Comm);

  pyAnalysis->SetScriptFile(scriptFile);
  pyAnalysis->SetScriptModule(scriptModule);
  pyAnalysis->SetInitializeSource(initSource);

  if (this->TimeInitialization(pyAnalysis, [&]() {
      return pyAnalysis->Initialize(); }))
    {
    SENSEI_ERROR("Failed to initialize PythonAnalysis")
    return -1;
    }
  this->Analyses.push_back(pyAnalysis);

  const char *scriptType = scriptFile.empty() ? "module" : "file";

  const char *scriptName =
    scriptFile.empty() ?  scriptModule.c_str() : scriptFile.c_str();

  SENSEI_STATUS("Configured python with " << scriptType
    << " \"" << scriptName << "\"")

  return 0;
#endif
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddSliceExtract(pugi::xml_node node)
{
#ifndef ENABLE_SLICE_EXTRACT
  (void)node;
  SENSEI_ERROR("SliceExtract requested but is disabled in this build")
  return -1;
#else

  std::ostringstream oss;
  oss << "Configured SliceExtract ";

  // initialize the slice extract
  auto adaptor = svtkSmartPointer<SliceExtract>::New();

  if (this->Comm != MPI_COMM_NULL)
    adaptor->SetCommunicator(this->Comm);

  // parse data requirements
  DataRequirements req;
  if (req.Initialize(node) || adaptor->SetDataRequirements(req))
    return -1;

  // parse writer parameters
  pugi::xml_node writerNode = node.child("writer");
  if (writerNode)
    {
    std::string outputDir = writerNode.attribute("output_dir").as_string("./");
    std::string mode = writerNode.attribute("mode").as_string("visit");
    std::string writer = writerNode.attribute("writer").as_string("xml");

    if (adaptor->SetWriterOutputDir(outputDir) || adaptor->SetWriterMode(mode) ||
      adaptor->SetWriterWriter(writer))
      return -1;

    oss << " writer.mode=" << mode << " writer.outputDir=" << outputDir
      << " writer.writer=" << writer;
    }

  // operation specific parsing
  if (XMLUtils::RequireAttribute(node, "operation"))
    return -1;

  std::string operation = node.attribute("operation").as_string();

  if (adaptor->SetOperation(operation))
    return -1;

  oss << " operation=" << operation;

  if (operation == "planar_slice")
    {
    // parse point and normal
    pugi::xml_node pointNode = node.child("point");
    if (pointNode)
      {
      std::array<double,3> point{0.0,0.0,0.0};
      if (XMLUtils::ParseNumeric(node.child("point"), point) ||
        adaptor->SetPoint(point))
        return -1;

      oss << " point=" << point;
      }

    pugi::xml_node normalNode = node.child("normal");
    if (normalNode)
      {
      std::array<double,3> normal{0.0,0.0,0.0};
      if (XMLUtils::ParseNumeric(node.child("normal"), normal) ||
        adaptor->SetNormal(normal))
        return -1;

      oss << " normal=" << normal;
      }
    }
  else if (operation == "iso_surface")
    {
    // parse iso values parameters
    if (XMLUtils::RequireChild(node, "iso_values"))
      return -1;

    pugi::xml_node valsNode = node.child("iso_values");

    std::vector<double> isoVals;
    if (XMLUtils::ParseNumeric(valsNode, isoVals))
      return -1;

    if (XMLUtils::RequireAttribute(valsNode, "mesh_name") ||
      XMLUtils::RequireAttribute(valsNode, "array_name") ||
      XMLUtils::RequireAttribute(valsNode, "array_centering"))
      return -1;

    std::string meshName = valsNode.attribute("mesh_name").as_string();
    std::string arrayName = valsNode.attribute("array_name").as_string();
    std::string arrayCenStr = valsNode.attribute("array_centering").as_string();

    int arrayCen = 0;
    if (SVTKUtils::GetAssociation(arrayCenStr.c_str(), arrayCen))
      return -1;

    adaptor->SetIsoValues(meshName, arrayName, arrayCen, isoVals);

    oss << " mesh_name=" << meshName << " array_name=" << arrayName
      << " array_centering=" << arrayCenStr << " iso_values=" << isoVals;
    }
  else
    {
    SENSEI_ERROR("Invalid operation \"" << operation << "\"")
    return -1;
    }

  // get other settings
  int enablePart = node.attribute("enable_partitioner").as_int(1);
  adaptor->EnablePartitioner(enablePart);
  oss << " enable_partitioner=" <<  enablePart;

  int verbose = node.attribute("verbose").as_int(0);
  adaptor->SetVerbose(verbose);
  oss << " verbose=" << verbose;

  // call intialize and add to the pipeline
  this->TimeInitialization(adaptor);
  this->Analyses.push_back(adaptor.GetPointer());

  SENSEI_STATUS(<< oss.str())

  return 0;
#endif
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddCalculator(pugi::xml_node node)
{
#if !defined(ENABLE_VTK_FILTERS)
  (void)node;
  SENSEI_ERROR("Calculator requested but is disabled in this build")
  return -1;
#else
  if (XMLUtils::RequireAttribute(node, "mesh") || XMLUtils::RequireAttribute(node, "expression") ||
      XMLUtils::RequireAttribute(node, "result"))
    {
    SENSEI_ERROR("Failed to initialize Calculator");
    return -1;
    }

  int association = 0;
  std::string assocStr = node.attribute("association").as_string("point");
  if (SVTKUtils::GetAssociation(assocStr, association))
    {
    SENSEI_ERROR("Failed to initialize Calculator");
    return -1;
    }

  std::string mesh = node.attribute("mesh").value();
  std::string expression = node.attribute("expression").value();
  std::string result = node.attribute("result").value();

  auto calculator = svtkSmartPointer<Calculator>::New();

  if (this->Comm != MPI_COMM_NULL)
    calculator->SetCommunicator(this->Comm);

  this->TimeInitialization(calculator, [&]() {
      calculator->Initialize(mesh, association, expression, result);
      return 0;
    });
  this->Analyses.push_back(calculator.GetPointer());

  SENSEI_STATUS("Configured calculator with expression '" << expression
    << "' on mesh '" << mesh << "' to generate '" << result << "' on "
    << assocStr);

  return 0;
#endif
}

//----------------------------------------------------------------------------
senseiNewMacro(ConfigurableAnalysis);

//----------------------------------------------------------------------------
ConfigurableAnalysis::ConfigurableAnalysis()
  : Internals(new ConfigurableAnalysis::InternalsType())
{
}

//----------------------------------------------------------------------------
ConfigurableAnalysis::~ConfigurableAnalysis()
{
  delete this->Internals;
}

//----------------------------------------------------------------------------
int ConfigurableAnalysis::SetCommunicator(MPI_Comm comm)
{
  this->AnalysisAdaptor::SetCommunicator(comm);

  // save the communicator for use with adaptors created
  // during XML parsing
  this->Internals->Comm = comm;

  // set the communicator on existing adaptors, if there are any
  AnalysisAdaptorVector::iterator iter = this->Internals->Analyses.begin();
  AnalysisAdaptorVector::iterator end = this->Internals->Analyses.end();

  for (; iter != end; ++iter)
    (*iter)->SetCommunicator(comm);

  return 0;
}


//----------------------------------------------------------------------------
int ConfigurableAnalysis::Initialize(const std::string& filename)
{
  TimeEvent<128> event("ConfigurableAnalysis::Initialize");

  pugi::xml_document doc;
  if (XMLUtils::Parse(this->GetCommunicator(), filename, doc))
    {
    SENSEI_ERROR("Failed to load, parse, and share XML configuration")
    MPI_Abort(this->GetCommunicator(), -1);
    return -1;
    }

  pugi::xml_node root = doc.child("sensei");

  return Initialize(root);
}

//----------------------------------------------------------------------------
int ConfigurableAnalysis::Initialize(const pugi::xml_node &root)
{
  TimeEvent<128> event("ConfigurableAnalysis::Initialize");

  // create and configure analysis adaptors
  for (pugi::xml_node node = root.child("analysis");
    node; node = node.next_sibling("analysis"))
    {

    if (!node.attribute("enabled").as_bool(true))
      {
      SENSEI_STATUS("Skipping analysis of type \"" << node.attribute("type").value() << "\" as it is not enabled")    
      continue;
      }

    std::string type = node.attribute("type").value();
    if (!(((type == "histogram") && !this->Internals->AddHistogram(node))
      || ((type == "autocorrelation") && !this->Internals->AddAutoCorrelation(node))
      || ((type == "adios1") && !this->Internals->AddAdios1(node))
      || ((type == "adios2") && !this->Internals->AddAdios2(node))
      || ((type == "ascent") && !this->Internals->AddAscent(node))
      || ((type == "catalyst") && !this->Internals->AddCatalyst(node))
      || ((type == "hdf5") && !this->Internals->AddHDF5(node))
      || ((type == "libsim") && !this->Internals->AddLibsim(node))
      || ((type == "ospray") && !this->Internals->AddOSPRay(node))
      || ((type == "PosthocIO") && !this->Internals->AddPosthocIO(node))
      || ((type == "VTKAmrWriter") && !this->Internals->AddVTKAmrWriter(node))
      || ((type == "svtkmcontour") && !this->Internals->AddVTKmContour(node))
      || ((type == "svtkmhaar") && !this->Internals->AddVTKmVolumeReduction(node))
      || ((type == "cdf") && !this->Internals->AddVTKmCDF(node))
      || ((type == "python") && !this->Internals->AddPythonAnalysis(node))
      || ((type == "SliceExtract") && !this->Internals->AddSliceExtract(node))
      || ((type == "calculator") && !this->Internals->AddCalculator(node))))
      {
      SENSEI_ERROR("Failed to add \"" << type << "\" analysis")
      MPI_Abort(this->GetCommunicator(), -1);
      }
    }

  // create and configure transport analysis adaptors
  for (pugi::xml_node node = root.child("transport");
    node; node = node.next_sibling("transport"))
    {
    
    if (!node.attribute("enabled").as_bool(true))
      {
      SENSEI_STATUS("Skipping transport of type \"" << node.attribute("type").value() << "\" as it is not enabled")    
      continue;
      }

    std::string type = node.attribute("type").value();
    if (!(((type == "adios1") && !this->Internals->AddAdios1(node))
      || ((type == "adios2") && !this->Internals->AddAdios2(node))
      || ((type == "hdf5") && !this->Internals->AddHDF5(node))))
      {
      SENSEI_ERROR("Failed to add \"" << type << "\" transport")
      MPI_Abort(this->GetCommunicator(), -1);
      }
    }

  return 0;
}

//----------------------------------------------------------------------------
bool ConfigurableAnalysis::Execute(DataAdaptor* data, DataAdaptor** dataOut)
{
  // Currently, we'll assume that only 1 analysis adaptor will generate
  // non-null result to report as the result; in case of multiple, the last one wins.
  // In future, we can extend the XML
  // specification to support identifying which analysis generates results of
  // interest and potentially how results are propagated between analyses.

  TimeEvent<128> event("ConfigurableAnalysis::Execute");

  int ai = 0;
  AnalysisAdaptorVector::iterator iter = this->Internals->Analyses.begin();
  AnalysisAdaptorVector::iterator end = this->Internals->Analyses.end();
  for (; iter != end; ++iter, ++ai)
    {
    const char* analysisName = nullptr;
    bool logEnabled = Profiler::Enabled();
    if (logEnabled)
      {
      analysisName = this->Internals->LogEventNames[3 * ai + 1].c_str();
      Profiler::StartEvent(analysisName);
      }

    if (!(*iter)->Execute(data, dataOut))
      {
      SENSEI_ERROR("Failed to execute " << (*iter)->GetClassName())
      MPI_Abort(this->GetCommunicator(), -1);
      }

    if (logEnabled)
      Profiler::EndEvent(analysisName);
    }

  return true;
}

//----------------------------------------------------------------------------
int ConfigurableAnalysis::Finalize()
{
  TimeEvent<128> event("ConfigurableAnalysis::Finalize");

  int ai = 0;
  AnalysisAdaptorVector::iterator iter = this->Internals->Analyses.begin();
  AnalysisAdaptorVector::iterator end = this->Internals->Analyses.end();
  for (; iter != end; ++iter, ++ai)
    {
    bool logEnabled = Profiler::Enabled();
    const char* analysisName = nullptr;
    if (logEnabled)
      {
      analysisName = this->Internals->LogEventNames[3 * ai + 2].c_str();
      Profiler::StartEvent(analysisName);
      }

    if ((*iter)->Finalize())
      {
      SENSEI_ERROR("Failed to finalize " << (*iter)->GetClassName())
      MPI_Abort(this->GetCommunicator(), -1);
      }

    if (logEnabled)
      Profiler::EndEvent(analysisName);
    }

  return 0;
}

//----------------------------------------------------------------------------
void ConfigurableAnalysis::PrintSelf(ostream& os, svtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
