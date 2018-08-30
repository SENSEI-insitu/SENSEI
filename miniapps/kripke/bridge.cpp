#include <conduit.hpp>
#include <ConduitDataAdaptor.h>
#include "bridge.h"
#include <Error.h>
#include <mpi.h>
#include <string>

#include <ConfigurableAnalysis.h>
#include <vtkNew.h>
#include <vtkSmartPointer.h>
#include <vtkDataObject.h>
#include <vtkObjectBase.h>
#include <iostream>

namespace BridgeGuts
{
  static vtkSmartPointer<sensei::ConduitDataAdaptor> DataAdaptor;
  static vtkSmartPointer<sensei::ConfigurableAnalysis> AnalysisAdaptor;
  static MPI_Comm comm;
}

void initialize(MPI_Comm comm, 
                conduit::Node* node, 
                const std::string& config_file)
{
  //setup communication
  BridgeGuts::comm = comm;
    
  //data adaptor
  BridgeGuts::DataAdaptor = vtkSmartPointer<sensei::ConduitDataAdaptor>::New();
  BridgeGuts::DataAdaptor->Initialize(node); 

  //analysis adaptor
  BridgeGuts::AnalysisAdaptor = vtkSmartPointer<sensei::ConfigurableAnalysis>::New();
  BridgeGuts::AnalysisAdaptor->Initialize(config_file); 
}

void analyze(conduit::Node* node)
{
  BridgeGuts::DataAdaptor->SetNode(node);
  if(!BridgeGuts::AnalysisAdaptor->Execute(BridgeGuts::DataAdaptor))
  {
    SENSEI_ERROR("ERROR: Failed to execute analysis")
    abort();
  }
    
  BridgeGuts::DataAdaptor->ReleaseData();
}

void finalize()
{
  BridgeGuts::AnalysisAdaptor->Finalize();
  BridgeGuts::AnalysisAdaptor = NULL;
  BridgeGuts::DataAdaptor     = NULL;
}
