#include "ConfigurableAnalysis.h"

#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkNew.h>
#include <vtkDataObject.h>

#include "Autocorrelation.h"
#include "PosthocIO.h"
#ifdef ENABLE_HISTOGRAM
# include "Histogram.h"
#endif
#ifdef ENABLE_ADIOS
# include "adios/AnalysisAdaptor.h"
#endif
#ifdef ENABLE_CATALYST
# include "catalyst/AnalysisAdaptor.h"
# include "catalyst/Slice.h"
#endif

#include <vector>
#include <pugixml.hpp>

#define ConfigurableAnalysisError(_arg) \
  cerr << "ERROR: " << __FILE__ " : "  << __LINE__ << std::endl \
    << "" _arg << std::endl;

namespace sensei
{

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

#ifdef ENABLE_HISTOGRAM
  void AddHistogram(MPI_Comm comm, pugi::xml_node node)
    {
    pugi::xml_attribute array = node.attribute("array");
    if (array)
      {
      int association = GetAssociation(node.attribute("association"));
      int bins = node.attribute("bins")? node.attribute("bins").as_int() : 10;

      vtkNew<Histogram> histogram;
      histogram->Initialize(comm, bins, association, array.value());
      this->Analyses.push_back(histogram.GetPointer());
      }
    else
      {
      ConfigurableAnalysisError(<< "'histogram' missing required attribute 'array'. Skipping.");
      }
    }
#endif

#ifdef ENABLE_ADIOS
  void AddAdios(MPI_Comm comm, pugi::xml_node node)
    {
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
    }
#endif

#ifdef ENABLE_CATALYST
  vtkSmartPointer<catalyst::AnalysisAdaptor> CatalystAnalysisAdaptor;

  void AddCatalyst(MPI_Comm comm, pugi::xml_node node)
    {
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
      this->CatalystAnalysisAdaptor->AddPipeline(slice.GetPointer());
      }
    }
#endif

  void AddAutoCorrelation(MPI_Comm comm, pugi::xml_node node)
    {
    vtkNew<Autocorrelation> adaptor;
    adaptor->Initialize(comm,
      node.attribute("window")? node.attribute("window").as_int() : 10,
      this->GetAssociation(node.attribute("association")),
      node.attribute("array").value(),
      node.attribute("k-max")? node.attribute("k-max").as_int() : 3);
    this->Analyses.push_back(adaptor.GetPointer());
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
bool ConfigurableAnalysis::Initialize(MPI_Comm world, const std::string& filename)
{
  int rank;
  MPI_Comm_rank(world, &rank);

  pugi::xml_document doc;
  pugi::xml_parse_result result = doc.load_file(filename.c_str());
  if (!result)
    {
    ConfigurableAnalysisError(
      "XML [" << filename << "] parsed with errors, attr value: ["
      << doc.child("node").attribute("attr").value() << "]" << endl
      << "Error description: " << result.description() << endl
      << "Error offset: " << result.offset << endl)
    return false;
    }
  pugi::xml_node sensei = doc.child("sensei");
  for (pugi::xml_node analysis = sensei.child("analysis");
    analysis; analysis = analysis.next_sibling("analysis"))
    {
    std::string type = analysis.attribute("type").value();
#ifdef ENABLE_HISTOGRAM
    if (type == "histogram")
      {
      this->Internals->AddHistogram(world, analysis);
      continue;
      }
#endif
#ifdef ENABLE_ADIOS
    if (type == "adios")
      {
      this->Internals->AddAdios(world, analysis);
      continue;
      }
#endif
#ifdef ENABLE_CATALYST
    if (type == "catalyst")
      {
      this->Internals->AddCatalyst(world, analysis);
      continue;
      }
#endif
    if (type == "autocorrelation")
      {
      this->Internals->AddAutoCorrelation(world, analysis);
      continue;
      }

    if ((type == "PosthocIO") &&
      !this->Internals->AddPosthocIO(world, analysis))
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
