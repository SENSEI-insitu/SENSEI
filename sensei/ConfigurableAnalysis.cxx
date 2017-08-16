#include "ConfigurableAnalysis.h"
#include "senseiConfig.h"
#include "Error.h"

#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkNew.h>
#include <vtkDataObject.h>

#include "senseiConfig.h"
#include "Autocorrelation.h"
#include "PosthocIO.h"
#include "Histogram.h"
#ifdef ENABLE_VTK_M
# include "VTKmContourAnalysis.h"
#ifdef ENABLE_CINEMA
# include "VTKmVolumeReductionAnalysis.h"
#endif
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
#ifdef ENABLE_CINEMA
#include "CatalystCinema.h"
#include "VTKmContourCompositeAnalysis.h"
#endif

#include <vector>
#include <pugixml.hpp>
#include <sstream>
#include <cstdio>
#include <errno.h>

namespace sensei
{

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

  // adds the VTK-m/Cinema analysis, if VTK-m features are present
  int AddVTKmContourCompositeAnalysis(MPI_Comm comm, pugi::xml_node node);

  // adds the VTK-m/Cinema analysis, if VTK-m features are present
  int AddVTKmVolumeReductionAnalysis(MPI_Comm comm, pugi::xml_node node);

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

private:
  int GetAssociation(pugi::xml_attribute asscNode);

public:
  std::vector<vtkSmartPointer<AnalysisAdaptor>> Analyses;
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
  pugi::xml_attribute array = node.attribute("array");
  if (!array)
    {
    SENSEI_ERROR("'histogram' missing required attribute 'array'");
    return -1;
    }

  SENSEI_STATUS("configured histogram " << array.value())

  int association = GetAssociation(node.attribute("association"));
  int bins = node.attribute("bins")? node.attribute("bins").as_int() : 10;

  vtkNew<Histogram> histogram;
  histogram->Initialize(comm, bins, association, array.value());
  this->Analyses.push_back(histogram.GetPointer());
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
  if (node.attribute("enabled") && !node.attribute("enabled").as_int())
    return -1;

  pugi::xml_attribute array = node.attribute("array");
  if (!array)
    {
    SENSEI_ERROR("'vtkmcontour' missing required attribute 'array'. Skipping.");
    return -1;
    }

  double value = node.attribute("value")? node.attribute("value").as_double() : 0.0;
  bool writeOutput = node.attribute("write_output")? node.attribute("write_output").as_bool() : 0.0;

  vtkNew<VTKmContourAnalysis> contour;
  contour->Initialize(comm, array.value(), value, writeOutput);
  this->Analyses.push_back(contour.GetPointer());
  return 0;
#endif
  }

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddVTKmVolumeReductionAnalysis(MPI_Comm comm,
  pugi::xml_node node)
  {
#ifndef ENABLE_CINEMA
  (void)comm;
  (void)node;
  SENSEI_ERROR("VTK-m was requested but is disabled in this build")
  return -1;
#else
  if (node.attribute("enabled") && !node.attribute("enabled").as_int())
    return -1;

  vtkNew<VTKmVolumeReductionAnalysis> volumeReduction;
  volumeReduction->Initialize(comm,
    node.attribute("working-directory").value(),
    node.attribute("reduction").as_int());
  this->Analyses.push_back(volumeReduction.GetPointer());
  return 0;
#endif
  }

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddVTKmContourCompositeAnalysis(MPI_Comm comm,
  pugi::xml_node node)
  {
#ifndef ENABLE_CINEMA
  (void)comm;
  (void)node;
  SENSEI_ERROR("VTK-m was requested but is disabled in this build")
  return -1;
#else
  if (node.attribute("enabled") && !node.attribute("enabled").as_int())
    return -1;

  int imageSize[2];
  imageSize[0] = node.attribute("image-width").as_int();
  imageSize[1] = node.attribute("image-height").as_int();

  vtkNew<VTKmContourCompositeAnalysis> contourComposite;
  contourComposite->Initialize(comm,
    node.attribute("working-directory").value(),
    imageSize,
    node.attribute("contours").value(),
    node.attribute("camera").value());
  this->Analyses.push_back(contourComposite.GetPointer());
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

  SENSEI_STATUS("configured ADIOS " << filename.value()
    << " " << method.value())

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
  SENSEI_STATUS("configured Catalyst")

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
    slice->ColorBy(
      this->GetAssociation(node.attribute("association")), node.attribute("array").value());
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
  if (strcmp(node.attribute("pipeline").value(), "cinema") == 0)
    {
    vtkNew<CatalystCinema> cinema;
    if (node.attribute("working-directory"))
      {
      cinema->SetImageParameters(
        node.attribute("working-directory").value(),
        node.attribute("image-width").as_int(800),
        node.attribute("image-height").as_int(800));
      }
    if (node.attribute("camera"))
      {
      cinema->SetCameraConfiguration(
        node.attribute("camera").value());
      }
    if (node.attribute("export-type"))
      {
      cinema->SetExportType(
        node.attribute("export-type").value());
      }
    if (node.attribute("contours"))
      {
      cinema->SetContours(
        node.attribute("contours").value());
      }
    this->CatalystAdaptor->AddPipeline(cinema.GetPointer());
    }
  if (strcmp(node.attribute("pipeline").value(), "pythonscript") == 0)
    {
#ifndef ENABLE_CATALYST_PYTHON
    SENSEI_ERROR("Catalyst Python was requested but is disabled in this build")
#else
    SENSEI_STATUS("configured Catalyst Python")
    if (node.attribute("filename"))
      {
      std::string fileName = node.attribute("filename").value();
      this->CatalystAdaptor->AddPythonScriptPipeline(fileName);
      }
#endif
    }
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
  SENSEI_STATUS("configured libsim")

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
    this->LibsimAdaptor->Initialize();
    this->Analyses.push_back(this->LibsimAdaptor);
    }

  LibsimImageProperties imageProps;
  if(node.attribute("image-filename") != NULL)
    imageProps.SetFilename(node.attribute("image-filename").value());

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

  // Add the image that we want to make.
  if(!this->LibsimAdaptor->AddPlots(plots, plotVars,
    slice, project, origin, normal, imageProps))
    return -2;
#endif
  return 0;
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddAutoCorrelation(MPI_Comm comm,
  pugi::xml_node node)
{
  vtkNew<Autocorrelation> adaptor;
  std::string arrayname = node.attribute("array").value();
  adaptor->Initialize(comm,
    node.attribute("window")? node.attribute("window").as_int() : 10,
    this->GetAssociation(node.attribute("association")),
    arrayname,
    node.attribute("k-max")? node.attribute("k-max").as_int() : 3);
  this->Analyses.push_back(adaptor.GetPointer());

  SENSEI_STATUS("configured autocorrelation " << arrayname)

  return 0;
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::AddPosthocIO(MPI_Comm comm,
  pugi::xml_node node)
{
  if (!node.attribute("array"))
    {
    SENSEI_ERROR("need at least one array");
    return -1;
    }
  std::string arrayName = node.attribute("array").value();

  SENSEI_STATUS("configured PosthocIO " << arrayName)

  std::vector<std::string> cellArrays;
  std::vector<std::string> pointArrays;
  pugi::xml_attribute assoc_att = node.attribute("association");
  if (assoc_att && (std::string(assoc_att.value()) == "cell"))
    cellArrays.push_back(arrayName);
  else
    pointArrays.push_back(arrayName);

  std::string outputDir = "./";
  if (node.attribute("output_dir"))
    outputDir = node.attribute("output_dir").value();

  std::string fileBase = "PosthocIO";
  if (node.attribute("file_base"))
    fileBase = node.attribute("file_base").value();

  std::string blockExt = "block";
  if (node.attribute("block_ext"))
    blockExt = node.attribute("block_ext").value();

  int mode = PosthocIO::mpiIO;
  if (node.attribute("mode"))
    {
    std::string val = node.attribute("mode").value();
    if (val == "vtkXmlP")
      mode = PosthocIO::vtkXmlP;
    else
    if (val == "mpiIO")
      mode = PosthocIO::mpiIO;
    else
      {
      SENSEI_ERROR(<< "invalid mode \"" << val << "\"");
      return -1;
      }
    }

  int period = 1;
  if (node.attribute("period"))
    period = node.attribute("period").as_int();

  PosthocIO *adapter = PosthocIO::New();

  adapter->Initialize(comm, outputDir, fileBase,
      blockExt, cellArrays, pointArrays, mode, period);

  this->Analyses.push_back(adapter);
  adapter->Delete();

  return 0;
}

// --------------------------------------------------------------------------
int ConfigurableAnalysis::InternalsType::GetAssociation(
  pugi::xml_attribute asscNode)
{
  std::string association = asscNode.value();
  if (!asscNode || (association == "point"))
    {
    return vtkDataObject::FIELD_ASSOCIATION_POINTS;
    }
  if (association == "cell")
    {
    return vtkDataObject::FIELD_ASSOCIATION_CELLS;
    }
  SENSEI_ERROR(<< "Invalid association type '"
    << association.c_str() << "'. Assuming 'point'");
  return vtkDataObject::FIELD_ASSOCIATION_POINTS;
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
    if (node.attribute("enabled") && !node.attribute("enabled").as_int())
      continue;

    std::string type = node.attribute("type").value();
    if (!(((type == "histogram") && !this->Internals->AddHistogram(comm, node))
      || ((type == "autocorrelation") && !this->Internals->AddAutoCorrelation(comm, node))
      || ((type == "adios") && !this->Internals->AddAdios(comm, node))
      || ((type == "catalyst") && !this->Internals->AddCatalyst(comm, node))
      || ((type == "libsim") && !this->Internals->AddLibsim(comm, node))
      || ((type == "PosthocIO") && !this->Internals->AddPosthocIO(comm, node))
      || ((type == "vtkmcontour") && !this->Internals->AddVTKmContour(comm, node))
      || ((type == "vtkmcontour-composite") && !this->Internals->AddVTKmContourCompositeAnalysis(comm, node))
      || ((type == "volume-reduction") && !this->Internals->AddVTKmVolumeReductionAnalysis(comm, node))))
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
  for (std::vector<vtkSmartPointer<AnalysisAdaptor> >::iterator iter
    = this->Internals->Analyses.begin();
    iter != this->Internals->Analyses.end(); ++iter)
    {
    iter->GetPointer()->Execute(data);
    }
  return true;
}

//----------------------------------------------------------------------------
void ConfigurableAnalysis::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}

}
