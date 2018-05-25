#include "ConfigurableAnalysis.h"
#include "senseiConfig.h"
#include "Error.h"
#include "VTKUtils.h"
#include "DataRequirements.h"

#include "Autocorrelation.h"
#include "Histogram.h"
#ifdef ENABLE_VTK_XMLP
#include "VTKPosthocIO.h"
#include "VTKAmrWriter.h"
#endif
#ifdef ENABLE_VTK_M
#include "VTKmContourAnalysis.h"
#endif
#ifdef ENABLE_ADIOS
#include "ADIOSAnalysisAdaptor.h"
#endif
#ifdef ENABLE_CATALYST
#include "CatalystAnalysisAdaptor.h"
#include "CatalystSlice.h"
#endif
#ifdef ENABLE_LIBSIM
#include "LibsimAnalysisAdaptor.h"
#include "LibsimImageProperties.h"
#endif

#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkNew.h>
#include <vtkDataObject.h>

#include <vector>
#include <pugixml.hpp>
#include <sstream>
#include <cstdio>
#include <errno.h>

using AnalysisAdaptorPtr = vtkSmartPointer<sensei::AnalysisAdaptor>;
using AnalysisAdaptorVector = std::vector<AnalysisAdaptorPtr>;

namespace sensei
{


static
int requireAttribute(pugi::xml_node &node, const char *attributeName)
{
  if (!node.attribute(attributeName))
    {
    SENSEI_ERROR(<< node.name()
      << " is missing required attribute " << attributeName)
    return -1;
    }
  return 0;
}


// --------------------------------------------------------------------------
static int parse(MPI_Comm comm, int rank,
   const std::string &filename, pugi::xml_document &doc)
{
  unsigned long nbytes = 0;
  char *buffer = nullptr;
  if(rank == 0)
    {
    FILE *f = fopen(filename.c_str(), "rb");
    if (f)
      {
      setvbuf(f, nullptr, _IONBF, 0);
      fseek(f, 0, SEEK_END);
      nbytes = ftell(f);
      fseek(f, 0, SEEK_SET);
      buffer = static_cast<char*>(
          pugi::get_memory_allocation_function()(nbytes));
      unsigned long nread = fread(buffer, 1, nbytes, f);
      fclose(f);
      if (nread == nbytes)
        {
        MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, 0, comm);
        MPI_Bcast(buffer, nbytes, MPI_CHAR, 0, comm);
        }
      else
        {
        SENSEI_ERROR("read error on \""  << filename << "\"" << endl
          << strerror(errno))
        nbytes = 0;
        MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, 0, comm);
        return -1;
        }
      }
    else
      {
      SENSEI_ERROR("failed to open \""  << filename << "\"" << endl
        << strerror(errno))
      MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, 0, comm);
      return -1;
      }
    }
  else
    {
    MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, 0, comm);
    if (!nbytes)
        return -1;
    buffer = static_cast<char*>(pugi::get_memory_allocation_function()(nbytes));
    MPI_Bcast(buffer, nbytes, MPI_CHAR, 0, comm);
    }
  pugi::xml_parse_result result = doc.load_buffer_inplace_own(buffer, nbytes);
  if (!result)
    {
    SENSEI_ERROR("XML [" << filename << "] parsed with errors, attr value: ["
      << doc.child("node").attribute("attr").value() << "]" << endl
      << "Error description: " << result.description() << endl
      << "Error offset: " << result.offset << endl)
    return -1;
    }
  return 0;
}

class ConfigurableAnalysis::InternalsType
{
public:
  // adds the histogram analysis
  int AddHistogram(MPI_Comm comm, pugi::xml_node node);

  // adds the VTK-m analysis, if VTK-m features are present
  int AddVTKmContour(MPI_Comm comm, pugi::xml_node node);

  // adds the ADIOS analysis, if ADIOS features are present
  int AddAdios(MPI_Comm comm, pugi::xml_node node);

  // adds the Catalyst analysis, if Catalyst features are present
  int AddCatalyst(MPI_Comm comm, pugi::xml_node node);

  // adds the libsim analysis, if libsim features are present
  int AddLibsim(MPI_Comm comm, pugi::xml_node node);

  // adds the autocorrelation analysis, if autocorrelation features are present
  int AddAutoCorrelation(MPI_Comm comm, pugi::xml_node node);

  // adds the post hoc I/O analysis
  int AddPosthocIO(MPI_Comm comm, pugi::xml_node node);

  // adds the VTK AMR writer
  int AddVTKAmrWriter(MPI_Comm comm, pugi::xml_node node);

public:
  AnalysisAdaptorVector Analyses;
#ifdef ENABLE_LIBSIM
  vtkSmartPointer<LibsimAnalysisAdaptor> LibsimAdaptor;
#endif
#ifdef ENABLE_CATALYST
  vtkSmartPointer<CatalystAnalysisAdaptor> CatalystAdaptor;
#endif
};

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddHistogram(MPI_Comm comm,
  pugi::xml_node node)
{
  if (requireAttribute(node, "mesh") || requireAttribute(node, "array"))
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

  vtkNew<Histogram> histogram;

  histogram->Initialize(comm, bins, mesh, association, array);

  this->Analyses.push_back(histogram.GetPointer());

  SENSEI_STATUS("Configured histogram with " << bins
    << " " << assocStr << "data array " << array
    << " on mesh " << mesh)

  return 0;
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddVTKmContour(MPI_Comm comm,
  pugi::xml_node node)
  {
#ifndef ENABLE_VTK_M
  (void)comm;
  (void)node;
  SENSEI_ERROR("VTK-m was requested but is disabled in this build")
  return -1;
#else

  if (requireAttribute(node, "mesh") || requireAttribute(node, "array"))
    {
    SENSEI_ERROR("Failed to initialize VTKmContourAnalysis");
    return -1;
    }

  double value = node.attribute("value").as_double(0.0);
  bool writeOutput = node.attribute("write_output").as_bool(false);

  vtkNew<VTKmContourAnalysis> contour;

  contour->Initialize(comm, mesh.value(),
    array.value(), value, writeOutput);

  this->Analyses.push_back(contour.GetPointer());

  SENSEI_STATUS("Configured VTKmContourAnalysis " << array.value())

  return 0;
#endif
  }

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddAdios(MPI_Comm comm,
  pugi::xml_node node)
{
  (void)comm;
#ifndef ENABLE_ADIOS
  (void)node;
  SENSEI_ERROR("ADIOS was requested but is disabled in this build")
  return -1;
#else

  vtkNew<ADIOSAnalysisAdaptor> adios;
  pugi::xml_attribute filename = node.attribute("filename");
  if (filename)
    adios->SetFileName(filename.value());

  pugi::xml_attribute method = node.attribute("method");
  if (method)
    adios->SetMethod(method.value());

  this->Analyses.push_back(adios.GetPointer());

  SENSEI_STATUS("Configured ADIOSAnalysisAdaptor \"" << filename.value()
    << "\" method " << method.value())

  return 0;
#endif
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddCatalyst(MPI_Comm comm,
  pugi::xml_node node)
{
#ifndef ENABLE_CATALYST
  (void)comm;
  (void)node;
  SENSEI_ERROR("Catalyst was requested but is disabled in this build")
  return -1;
#else
  (void)comm;
  if (!this->CatalystAdaptor)
    {
    this->CatalystAdaptor = vtkSmartPointer<CatalystAnalysisAdaptor>::New();
    this->Analyses.push_back(this->CatalystAdaptor);
    }
  if (strcmp(node.attribute("pipeline").value(), "slice") == 0)
    {
    vtkNew<CatalystSlice> slice;
    double tmp[3];
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
int ConfigurableAnalysis::InternalsType::AddLibsim(MPI_Comm comm,
  pugi::xml_node node)
{
#ifndef ENABLE_LIBSIM
  (void)comm;
  (void)node;
  SENSEI_ERROR("Libsim was requested but is disabled in this build")
  return -1;
#else

  // We keep around a single instance of the libsim adaptor and then tell it to
  // do different things.
  if (!this->LibsimAdaptor)
    {
    this->LibsimAdaptor = vtkSmartPointer<LibsimAnalysisAdaptor>::New();
    this->LibsimAdaptor->SetComm(comm);
    if(node.attribute("trace"))
      this->LibsimAdaptor->SetTraceFile(node.attribute("trace").value());
    if(node.attribute("options"))
      this->LibsimAdaptor->SetOptions(node.attribute("options").value());
    if(node.attribute("visitdir"))
      this->LibsimAdaptor->SetVisItDirectory(node.attribute("visitdir").value());
    if(node.attribute("mode"))
      this->LibsimAdaptor->SetMode(node.attribute("mode").value());
    this->LibsimAdaptor->Initialize();
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
  if(node.attribute("image-filename") != NULL)
  {
    imageProps.SetFilename(node.attribute("image-filename").value());
    filename = node.attribute("image-filename").value();
  }
  if(node.attribute("filename") != NULL)
  {
    imageProps.SetFilename(node.attribute("filename").value());
    filename = node.attribute("filename").value();
  }
  if(node.attribute("image-width") != NULL)
    imageProps.SetWidth(node.attribute("image-width").as_int());

  if(node.attribute("image-height") != NULL)
    imageProps.SetHeight(node.attribute("image-height").as_int());

  std::string plots, plotVars;
  double origin[3] = {0.,0.,0.};
  double normal[3] = {1.,0.,0.};
  bool slice = false, project = false;

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
    if(!this->LibsimAdaptor->AddExport(frequency, plots, plotVars,
      slice, project, origin, normal, filename))
      return -2;

    SENSEI_STATUS("configured LibsimAnalysisAdaptor export")
  }
  else
  {
    // Add the image that we want to make.
    if(!this->LibsimAdaptor->AddRender(frequency, plots, plotVars,
      slice, project, origin, normal, imageProps))
      return -2;

    SENSEI_STATUS("configured LibsimAnalysisAdaptor render")
  }

#endif
  return 0;
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddAutoCorrelation(MPI_Comm comm,
  pugi::xml_node node)
{
  if (requireAttribute(node, "mesh") || requireAttribute(node, "array"))
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


  vtkNew<Autocorrelation> adaptor;
  adaptor->Initialize(comm, window, meshName, assoc, arrayName, kMax);

  this->Analyses.push_back(adaptor.GetPointer());

  SENSEI_STATUS("Configured Autocorrelation " << assocStr
    << " data array " << arrayName << " on mesh " << meshName
    << " window " << window << " k-max " << kMax)

  return 0;
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddPosthocIO(MPI_Comm comm,
  pugi::xml_node node)
{
#ifndef ENABLE_VTK_XMLP
  (void)comm;
  (void)node;
  SENSEI_ERROR("VTK I/O was requested but is disabled in this build")
  return -1;
#else
  DataRequirements req;

  if (req.Initialize(node) ||
    (req.GetNumberOfRequiredMeshes() < 1))
    {
    SENSEI_ERROR("Failed to initialize VTKPosthocIO. "
      "At least one mesh is required")
    return -1;
    }

  std::string outputDir = node.attribute("output_dir").as_string("./");
  std::string fileName = node.attribute("file_name").as_string("data");
  std::string mode = node.attribute("mode").as_string("visit");

  vtkNew<VTKPosthocIO> adapter;

  if (adapter->SetCommunicator(comm) ||
    adapter->SetOutputDir(outputDir) ||
    adapter->SetMode(mode) ||
    adapter->SetDataRequirements(req))
    {
    SENSEI_ERROR("Failed to initialize the VTKPosthocIO analysis")
    return -1;
    }

  this->Analyses.push_back(adapter.GetPointer());

  SENSEI_STATUS("Configured VTKPosthocIO")

  return 0;
#endif
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddVTKAmrWriter(MPI_Comm comm,
  pugi::xml_node node)
{
#ifndef ENABLE_VTK_XMLP
  (void)comm;
  (void)node;
  SENSEI_ERROR("VTK AMR Writer was requested but is disabled in this build")
  return -1;
#else
  DataRequirements req;

  if (req.Initialize(node) ||
    (req.GetNumberOfRequiredMeshes() < 1))
    {
    SENSEI_ERROR("Failed to initialize VTKAmrWriter. "
      "At least one mesh is required")
    return -1;
    }

  std::string outputDir = node.attribute("output_dir").as_string("./");
  std::string fileName = node.attribute("file_name").as_string("data");
  std::string mode = node.attribute("mode").as_string("visit");

  vtkNew<VTKAmrWriter> adapter;

  if (adapter->SetCommunicator(comm) ||
    adapter->SetOutputDir(outputDir) ||
    adapter->SetMode(mode) ||
    adapter->SetDataRequirements(req) ||
    adapter->Initialize())
    {
    SENSEI_ERROR("Failed to initialize the VTKAmrWriter analysis")
    return -1;
    }

  this->Analyses.push_back(adapter.GetPointer());

  SENSEI_STATUS("Configured VTKAmrWriter")

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
bool ConfigurableAnalysis::Initialize(MPI_Comm comm, const std::string& filename)
{
  int rank = 0;
  MPI_Comm_rank(comm, &rank);
  pugi::xml_document doc;
  if (parse(comm, rank, filename, doc))
    {
    SENSEI_ERROR("failed to parse configuration")
    return false;
    }

  pugi::xml_node root = doc.child("sensei");
  for (pugi::xml_node node = root.child("analysis");
    node; node = node.next_sibling("analysis"))
    {
    if (!node.attribute("enabled").as_int(0))
      continue;

    std::string type = node.attribute("type").value();
    if (!(((type == "histogram") && !this->Internals->AddHistogram(comm, node))
      || ((type == "autocorrelation") && !this->Internals->AddAutoCorrelation(comm, node))
      || ((type == "adios") && !this->Internals->AddAdios(comm, node))
      || ((type == "catalyst") && !this->Internals->AddCatalyst(comm, node))
      || ((type == "libsim") && !this->Internals->AddLibsim(comm, node))
      || ((type == "PosthocIO") && !this->Internals->AddPosthocIO(comm, node))
      || ((type == "VTKAmrWriter") && !this->Internals->AddVTKAmrWriter(comm, node))
      || ((type == "vtkmcontour") && !this->Internals->AddVTKmContour(comm, node))))
      {
      if (rank == 0)
        SENSEI_ERROR("Failed to add '" << type << "' analysis")
      }
    }
  return true;
}

//----------------------------------------------------------------------------
bool ConfigurableAnalysis::Execute(DataAdaptor* data)
{
  AnalysisAdaptorVector::iterator iter = this->Internals->Analyses.begin();
  AnalysisAdaptorVector::iterator end = this->Internals->Analyses.end();
  for (; iter != end; ++iter)
    {
    // TODO -- should we be checking the return value here?
    // what happens when an analysis errors out? do we keep trying
    // to execute it?
    (*iter)->Execute(data);
    }
  return true;
}

//----------------------------------------------------------------------------
int ConfigurableAnalysis::Finalize()
{
  AnalysisAdaptorVector::iterator iter = this->Internals->Analyses.begin();
  AnalysisAdaptorVector::iterator end = this->Internals->Analyses.end();
  for (; iter != end; ++iter)
    {
    (*iter)->Finalize();
    }
  return true;
}
//----------------------------------------------------------------------------
void ConfigurableAnalysis::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
