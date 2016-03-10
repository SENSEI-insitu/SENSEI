#include "ConfigurableAnalysis.h"

#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkNew.h>
#include <vtkDataObject.h>

#include "Autocorrelation.h"
#include "PosthocIO.h"
#include "Histogram.h"
#ifdef ENABLE_ADIOS
# include "adios/AnalysisAdaptor.h"
#endif
#ifdef ENABLE_CATALYST
# include "catalyst/AnalysisAdaptor.h"
# include "catalyst/Slice.h"
#endif

#include <vector>
#include <pugixml.hpp>
#include <sstream>

#define ConfigurableAnalysisError(_arg) \
  cerr << "ERROR: " << __FILE__ " : "  << __LINE__ << std::endl \
    << "" _arg << std::endl;

namespace sensei
{
// --------------------------------------------------------------------------
int parse(MPI_Comm comm, int rank,
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
        ConfigurableAnalysisError(
          << "read error on \""  << filename << "\"" << endl
          << strerror(errno))
        nbytes = 0;
        MPI_Bcast(&nbytes, 1, MPI_UNSIGNED_LONG, 0, comm);
        return -1;
        }
      }
    else
      {
      ConfigurableAnalysisError(
        << "failed to open \""  << filename << "\"" << endl
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
    ConfigurableAnalysisError(
      "XML [" << filename << "] parsed with errors, attr value: ["
      << doc.child("node").attribute("attr").value() << "]" << endl
      << "Error description: " << result.description() << endl
      << "Error offset: " << result.offset << endl)
    return -1;
    }
  return 0;
}

class ConfigurableAnalysis::vtkInternals
{
  int GetAssociation(pugi::xml_attribute asscNode)
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
    ConfigurableAnalysisError(<< "Invalid association type '" << association.c_str() << "'. Assuming 'point'");
    return vtkDataObject::FIELD_ASSOCIATION_POINTS;
    }

public:
  std::vector<vtkSmartPointer<AnalysisAdaptor> > Analyses;

  int AddHistogram(MPI_Comm comm, pugi::xml_node node)
    {
    if (node.attribute("enabled") && !node.attribute("enabled").as_int())
      return -1;

    pugi::xml_attribute array = node.attribute("array");
    if (!array)
      {
      ConfigurableAnalysisError(<< "'histogram' missing required attribute 'array'. Skipping.");
      return -1;
      }
    int association = GetAssociation(node.attribute("association"));
    int bins = node.attribute("bins")? node.attribute("bins").as_int() : 10;

    vtkNew<Histogram> histogram;
    histogram->Initialize(comm, bins, association, array.value());
    this->Analyses.push_back(histogram.GetPointer());
    return 0;
    }

#ifdef ENABLE_ADIOS
  int AddAdios(MPI_Comm comm, pugi::xml_node node)
    {
    if (node.attribute("enabled") && !node.attribute("enabled").as_int())
      return -1;

    vtkNew<adios::AnalysisAdaptor> adios;
    pugi::xml_attribute filename = node.attribute("filename");
    pugi::xml_attribute method = node.attribute("method");
    if (filename)
      {
      adios->SetFileName(filename.value());
      }
    if (method)
      {
      adios->SetMethod(method.value());
      }
    this->Analyses.push_back(adios.GetPointer());

    return 0;
    }
#endif

#ifdef ENABLE_CATALYST
  vtkSmartPointer<catalyst::AnalysisAdaptor> CatalystAnalysisAdaptor;

  int AddCatalyst(MPI_Comm comm, pugi::xml_node node)
    {
    if (node.attribute("enabled") && !node.attribute("enabled").as_int())
      return -1;

    if (!this->CatalystAnalysisAdaptor)
      {
      this->CatalystAnalysisAdaptor = vtkSmartPointer<catalyst::AnalysisAdaptor>::New();
      this->Analyses.push_back(this->CatalystAnalysisAdaptor);
      }
    if (strcmp(node.attribute("pipeline").value(), "slice") == 0)
      {
      vtkNew<catalyst::Slice> slice;
      // TODO: parse origin and normal.
      slice->SetSliceNormal(0, 0, 1);
      slice->SetSliceOrigin(0, 0, 0);
      slice->ColorBy(
        this->GetAssociation(node.attribute("association")), node.attribute("array").value());
      if (node.attribute("image-filename") && node.attribute("image-width") && node.attribute("image-height"))
        {
        slice->SetImageParameters(
          node.attribute("image-filename").value(),
          node.attribute("image-width").as_int(),
          node.attribute("image-height").as_int());
        }
      this->CatalystAnalysisAdaptor->AddPipeline(slice.GetPointer());
      }

    return 0;
    }
#endif

  int AddAutoCorrelation(MPI_Comm comm, pugi::xml_node node)
    {
    if (node.attribute("enabled") && !node.attribute("enabled").as_int())
      return -1;

    vtkNew<Autocorrelation> adaptor;
    std::string arrayname = node.attribute("array").value();
    adaptor->Initialize(comm,
      node.attribute("window")? node.attribute("window").as_int() : 10,
      this->GetAssociation(node.attribute("association")),
      arrayname,
      node.attribute("k-max")? node.attribute("k-max").as_int() : 3);
    this->Analyses.push_back(adaptor.GetPointer());

    return 0;
    }

  int AddPosthocIO(MPI_Comm comm, pugi::xml_node node)
    {
    if (!node.attribute("enabled") || !node.attribute("enabled").as_int())
      return -1;

    if (!node.attribute("array"))
      {
      ConfigurableAnalysisError(<< "need at least one array");
      return -1;
      }
    std::string arrayName = node.attribute("array").value();

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
        ConfigurableAnalysisError(<< "invalid mode \"" << val << "\"");
      }

    int period = 1;
    if (node.attribute("period"))
        period = node.attribute("period").as_int();


    PosthocIO *adapter = PosthocIO::New();

    adapter->Initialize(comm, outputDir, fileBase,
        blockExt, cellArrays, pointArrays, mode,
        period);

    this->Analyses.push_back(adapter);
    adapter->Delete();

    return 0;
    }
};

//----------------------------------------------------------------------------
vtkStandardNewMacro(ConfigurableAnalysis);

//----------------------------------------------------------------------------
ConfigurableAnalysis::ConfigurableAnalysis()
  : Internals(new ConfigurableAnalysis::vtkInternals())
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
    ConfigurableAnalysisError("failed to parse configuration")
    return false;
    }

  pugi::xml_node sensei = doc.child("sensei");
  for (pugi::xml_node analysis = sensei.child("analysis");
    analysis; analysis = analysis.next_sibling("analysis"))
    {
    std::string type = analysis.attribute("type").value();
    if ((type == "histogram") &&
      !this->Internals->AddHistogram(comm, analysis))
      continue;
#ifdef ENABLE_ADIOS
    if ((type == "adios") &&
      !this->Internals->AddAdios(comm, analysis))
      continue;
#endif
#ifdef ENABLE_CATALYST
    if ((type == "catalyst") &&
      !this->Internals->AddCatalyst(comm, analysis))
      continue;
#endif
    if ((type == "autocorrelation") &&
      !this->Internals->AddAutoCorrelation(comm, analysis))
      continue;

    if ((type == "PosthocIO") &&
      !this->Internals->AddPosthocIO(comm, analysis))
      continue;

    if (rank == 0)
      std::cerr << "Skipping '" << type.c_str() << "'." << std::endl;
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
