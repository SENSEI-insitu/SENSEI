#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkNew.h>
#include <vtkDataObject.h>

#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <errno.h>

#include "ConfigurableAnalysis.h"
#include "senseiConfig.h"
#include "Error.h"
#include "Timer.h"
#include "VTKUtils.h"
#include "XMLUtils.h"
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
#ifdef ENABLE_HDF5
#include "HDF5AnalysisAdaptor.h"
#endif
#ifdef ENABLE_CATALYST
#include "CatalystAnalysisAdaptor.h"
#include "CatalystParticle.h"
#include "CatalystSlice.h"
#endif
#ifdef ENABLE_LIBSIM
#include "LibsimAnalysisAdaptor.h"
#include "LibsimImageProperties.h"
#endif
#ifdef ENABLE_PYTHON
#include "PythonAnalysis.h"
#endif

using AnalysisAdaptorPtr = vtkSmartPointer<sensei::AnalysisAdaptor>;
using AnalysisAdaptorVector = std::vector<AnalysisAdaptorPtr>;

namespace sensei
{

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
  int AddAdios(pugi::xml_node node);
  int AddHDF5(pugi::xml_node node);
  int AddCatalyst(pugi::xml_node node);
  int AddLibsim(pugi::xml_node node);
  int AddAutoCorrelation(pugi::xml_node node);
  int AddPosthocIO(pugi::xml_node node);
  int AddVTKAmrWriter(pugi::xml_node node);
  int AddPythonAnalysis(pugi::xml_node node);

public:
  // list of all analyses. api calls are forwareded to each
  // analysis in the list
  AnalysisAdaptorVector Analyses;

  // special analyses. these apear in the above list, however
  // they require special treatment which is simplified by
  // storing an additional pointer.
#ifdef ENABLE_LIBSIM
  vtkSmartPointer<LibsimAnalysisAdaptor> LibsimAdaptor;
#endif
#ifdef ENABLE_CATALYST
  vtkSmartPointer<CatalystAnalysisAdaptor> CatalystAdaptor;
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
  bool logEnabled = timer::Enabled();
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
    timer::MarkStartEvent(analysisName);
  }

  int result = initializer();

  if (logEnabled)
    timer::MarkEndEvent(analysisName);

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
  if (VTKUtils::GetAssociation(assocStr, association))
    {
    SENSEI_ERROR("Failed to initialize Histogram");
    return -1;
    }

  std::string mesh = node.attribute("mesh").value();
  std::string array = node.attribute("array").value();
  int bins = node.attribute("bins").as_int(10);
  std::string fileName = node.attribute("file").value();

  auto histogram = vtkSmartPointer<Histogram>::New();

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
  SENSEI_ERROR("vtkAcceleratorsVtkm was requested but is disabled in this build")
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

  auto contour = vtkSmartPointer<VTKmContourAnalysis>::New();

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

  auto reducer = vtkSmartPointer<VTKmVolumeReductionAnalysis>::New();
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

  auto analysis = vtkSmartPointer<VTKmCDFAnalysis>::New();
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
int ConfigurableAnalysis::InternalsType::AddAdios(pugi::xml_node node)
{
#ifndef ENABLE_ADIOS1
  (void)node;
  SENSEI_ERROR("ADIOS 1 was requested but is disabled in this build")
  return -1;
#else
  auto adios = vtkSmartPointer<ADIOS1AnalysisAdaptor>::New();

  if (this->Comm != MPI_COMM_NULL)
    adios->SetCommunicator(this->Comm);

  pugi::xml_attribute filename = node.attribute("filename");
  if (filename)
    adios->SetFileName(filename.value());

  pugi::xml_attribute method = node.attribute("method");
  if (method)
    adios->SetMethod(method.value());

  DataRequirements req;
  if (req.Initialize(node))
    {
    SENSEI_ERROR("Failed to initialize ADIOS 1.")
    return -1;
    }
  adios->SetDataRequirements(req);

  this->TimeInitialization(adios);
  this->Analyses.push_back(adios.GetPointer());

  SENSEI_STATUS("Configured ADIOSAnalysisAdaptor \"" << filename.value()
    << "\" method " << method.value())

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
  auto dataE = vtkSmartPointer<HDF5AnalysisAdaptor>::New();

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
    this->CatalystAdaptor = vtkSmartPointer<CatalystAnalysisAdaptor>::New();

    if (this->Comm != MPI_COMM_NULL)
      this->CatalystAdaptor->SetCommunicator(this->Comm);

    this->TimeInitialization(this->CatalystAdaptor);
    this->Analyses.push_back(this->CatalystAdaptor);
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
    if (VTKUtils::GetAssociation(assocStr, association))
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
    if (VTKUtils::GetAssociation(assocStr, association))
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
      this->CatalystAdaptor->AddPythonScriptPipeline(fileName);
      }
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
    this->LibsimAdaptor = vtkSmartPointer<LibsimAnalysisAdaptor>::New();
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
    this->TimeInitialization(this->LibsimAdaptor, [&]() {
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
  if (VTKUtils::GetAssociation(assocStr, assoc))
    {
    SENSEI_ERROR("Failed to initialize Autocorrelation");
    return -1;
    }

  int window = node.attribute("window").as_int(10);
  int kMax = node.attribute("k-max").as_int(3);
  int numThreads = node.attribute("n-threads").as_int(1);

  auto adaptor = vtkSmartPointer<Autocorrelation>::New();

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

  auto adaptor = vtkSmartPointer<VTKPosthocIO>::New();

  if (this->Comm != MPI_COMM_NULL)
    adaptor->SetCommunicator(this->Comm);

  adaptor->SetGhostArrayName(ghostArrayName);

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

  auto adapter = vtkSmartPointer<VTKAmrWriter>::New();

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

  auto pyAnalysis = vtkSmartPointer<PythonAnalysis>::New();

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
  timer::MarkEvent event("ConfigurableAnalysis::Initialize");

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
  timer::MarkEvent event("ConfigurableAnalysis::Initialize");

  for (pugi::xml_node node = root.child("analysis");
    node; node = node.next_sibling("analysis"))
    {
    if (!node.attribute("enabled").as_int(0))
      continue;

    std::string type = node.attribute("type").value();
    if (!(((type == "histogram") && !this->Internals->AddHistogram(node))
      || ((type == "autocorrelation") && !this->Internals->AddAutoCorrelation(node))
      || ((type == "adios1") && !this->Internals->AddAdios(node))
      || ((type == "catalyst") && !this->Internals->AddCatalyst(node))
      || ((type == "hdf5") && !this->Internals->AddHDF5(node))
      || ((type == "libsim") && !this->Internals->AddLibsim(node))
      || ((type == "PosthocIO") && !this->Internals->AddPosthocIO(node))
      || ((type == "VTKAmrWriter") && !this->Internals->AddVTKAmrWriter(node))
      || ((type == "vtkmcontour") && !this->Internals->AddVTKmContour(node))
      || ((type == "vtkmhaar") && !this->Internals->AddVTKmVolumeReduction(node))
      || ((type == "cdf") && !this->Internals->AddVTKmCDF(node))
      || ((type == "python") && !this->Internals->AddPythonAnalysis(node))))
      {
      SENSEI_ERROR("Failed to add '" << type << "' analysis")
      MPI_Abort(this->GetCommunicator(), -1);
      }
    }

  return 0;
}

//----------------------------------------------------------------------------
bool ConfigurableAnalysis::Execute(DataAdaptor* data)
{
  timer::MarkEvent event("ConfigurableAnalysis::Execute");

  int ai = 0;
  AnalysisAdaptorVector::iterator iter = this->Internals->Analyses.begin();
  AnalysisAdaptorVector::iterator end = this->Internals->Analyses.end();
  for (; iter != end; ++iter, ++ai)
    {
    const char* analysisName = nullptr;
    bool logEnabled = timer::Enabled();
    if (logEnabled)
      {
      analysisName = this->Internals->LogEventNames[3 * ai + 1].c_str();
      timer::MarkStartEvent(analysisName);
      }

    if (!(*iter)->Execute(data))
      {
      SENSEI_ERROR("Failed to execute " << (*iter)->GetClassName())
      MPI_Abort(this->GetCommunicator(), -1);
      }

    if (logEnabled)
      timer::MarkEndEvent(analysisName);
    }

  return true;
}

//----------------------------------------------------------------------------
int ConfigurableAnalysis::Finalize()
{
  timer::MarkEvent event("ConfigurableAnalysis::Finalize");

  int ai = 0;
  AnalysisAdaptorVector::iterator iter = this->Internals->Analyses.begin();
  AnalysisAdaptorVector::iterator end = this->Internals->Analyses.end();
  for (; iter != end; ++iter, ++ai)
    {
    bool logEnabled = timer::Enabled();
    const char* analysisName = nullptr;
    if (logEnabled)
      {
      analysisName = this->Internals->LogEventNames[3 * ai + 2].c_str();
      timer::MarkStartEvent(analysisName);
      }

    if ((*iter)->Finalize())
      {
      SENSEI_ERROR("Failed to finalize " << (*iter)->GetClassName())
      MPI_Abort(this->GetCommunicator(), -1);
      }

    if (logEnabled)
      timer::MarkEndEvent(analysisName);
    }

  return 0;
}

//----------------------------------------------------------------------------
void ConfigurableAnalysis::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
