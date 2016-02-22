/*=========================================================================

  Program:   Visualization Toolkit
  Module:    vtkConfigurableAnalysisAdaptor.cxx

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/
#include "vtkConfigurableAnalysisAdaptor.h"

#include <vtkObjectFactory.h>
#include <vtkSmartPointer.h>
#include <vtkNew.h>
#include <vtkDataObject.h>

#ifdef ENABLE_HISTOGRAM
# include <HistogramAnalysisAdaptor.h>
#endif
#ifdef ENABLE_ADIOS
# include <vtkADIOSAnalysisAdaptor.h>
#endif
#ifdef ENABLE_CATALYST
# include <vtkCatalystSlicePipeline.h>
# include <vtkCatalystAnalysisAdaptor.h>
#endif
#include <AutocorrelationAnalysisAdaptor.h>

#include <vector>
#include <pugixml.hpp>

class vtkConfigurableAnalysisAdaptor::vtkInternals
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
    cout << "Invalid association type '" << association.c_str() << "'. Assuming 'point'" << endl;
    return vtkDataObject::FIELD_ASSOCIATION_POINTS;
    }

public:
  std::vector<vtkSmartPointer<vtkInsituAnalysisAdaptor> > Analyses;

#ifdef ENABLE_HISTOGRAM
  void AddHistogram(MPI_Comm comm, pugi::xml_node node)
    {
    pugi::xml_attribute array = node.attribute("array");
    if (array)
      {
      int association = GetAssociation(node.attribute("association"));
      int bins = node.attribute("bins")? node.attribute("bins").as_int() : 10;

      vtkNew<HistogramAnalysisAdaptor> histogram;
      histogram->Initialize(comm, bins, association, array.value());
      this->Analyses.push_back(histogram.GetPointer());
      }
    else
      {
      cerr << "'histogram' missing required attribute 'array'. Skipping." << endl;
      }
    }
#endif

#ifdef ENABLE_ADIOS
  void AddAdios(MPI_Comm comm, pugi::xml_node node)
    {
    vtkNew<vtkADIOSAnalysisAdaptor> adios;
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
  vtkSmartPointer<vtkCatalystAnalysisAdaptor> CatalystAnalysisAdaptor;

  void AddCatalyst(MPI_Comm comm, pugi::xml_node node)
    {
    if (!this->CatalystAnalysisAdaptor)
      {
      this->CatalystAnalysisAdaptor = vtkSmartPointer<vtkCatalystAnalysisAdaptor>::New();
      this->Analyses.push_back(this->CatalystAnalysisAdaptor);
      }
    if (strcmp(node.attribute("pipeline").value(), "slice") == 0)
      {
      vtkNew<vtkCatalystSlicePipeline> slice;
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
    vtkNew<AutocorrelationAnalysisAdaptor> adaptor;
    adaptor->Initialize(comm,
      node.attribute("window")? node.attribute("window").as_int() : 10,
      this->GetAssociation(node.attribute("association")),
      node.attribute("array").value());
    this->Analyses.push_back(adaptor.GetPointer());
    }
};

vtkStandardNewMacro(vtkConfigurableAnalysisAdaptor);
//----------------------------------------------------------------------------
vtkConfigurableAnalysisAdaptor::vtkConfigurableAnalysisAdaptor()
  : Internals(new vtkConfigurableAnalysisAdaptor::vtkInternals())
{
}

//----------------------------------------------------------------------------
vtkConfigurableAnalysisAdaptor::~vtkConfigurableAnalysisAdaptor()
{
  delete this->Internals;
}

//----------------------------------------------------------------------------
bool vtkConfigurableAnalysisAdaptor::Initialize(MPI_Comm world, const std::string& filename)
{
  pugi::xml_document doc;
  pugi::xml_parse_result result = doc.load_file(filename.c_str());
  if (!result)
    {
    cout << "XML [" << filename << "] parsed with errors, attr value: [" << doc.child("node").attribute("attr").value() << "]\n";
    cout << "Error description: " << result.description() << "\n";
    cout << "Error offset: " << result.offset << endl;
    return false;
    }
  pugi::xml_node sensei = doc.child("sensei");
  for (pugi::xml_node analysis = sensei.child("analysis"); analysis; analysis = analysis.next_sibling("analysis"))
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
    cerr << "Skipping '" << type.c_str() << "'." << endl;
    }
}

//----------------------------------------------------------------------------
bool vtkConfigurableAnalysisAdaptor::Execute(vtkInsituDataAdaptor* data)
{
  for (std::vector<vtkSmartPointer<vtkInsituAnalysisAdaptor> >::iterator iter
    = this->Internals->Analyses.begin();
    iter != this->Internals->Analyses.end(); ++iter)
    {
    iter->GetPointer()->Execute(data);
    }
  return true;
}

//----------------------------------------------------------------------------
void vtkConfigurableAnalysisAdaptor::PrintSelf(ostream& os, vtkIndent indent)
{
  this->Superclass::PrintSelf(os, indent);
}
