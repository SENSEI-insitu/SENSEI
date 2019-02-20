#include <ConduitDataAdaptor.h>
#include <ConfigurableAnalysis.h>
#include <Error.h>
#include <vtkSmartPointer.h>

#include "bridge.h"

namespace SenseiBridge
{
  static vtkSmartPointer<sensei::ConduitDataAdaptor> DataAdaptor;
  static vtkSmartPointer<sensei::ConfigurableAnalysis> AnalysisAdaptor;
}

void SenseiInitialize( const std::string& config_file )
{
  // Create data adaptor.
  SenseiBridge::DataAdaptor = vtkSmartPointer<sensei::ConduitDataAdaptor>::New();

  // Create analysis adaptor.
  SenseiBridge::AnalysisAdaptor = vtkSmartPointer<sensei::ConfigurableAnalysis>::New();
  if( SenseiBridge::AnalysisAdaptor->Initialize(config_file) < 0 )
  {
    SENSEI_ERROR( "ERROR: Failed to create analysis" );
    exit( -1 );
  }
}

void SenseiAnalyze( conduit::Node &node )
{
  SenseiBridge::DataAdaptor->SetNode( &node );
  if( !SenseiBridge::AnalysisAdaptor->Execute(SenseiBridge::DataAdaptor) )
  {
    SENSEI_ERROR("ERROR: Failed to execute analysis");
    exit( -1 );
  }
    
  SenseiBridge::DataAdaptor->ReleaseData();
}

void SenseiFinalize()
{
  SenseiBridge::AnalysisAdaptor->Finalize();

  SenseiBridge::AnalysisAdaptor = nullptr;
  SenseiBridge::DataAdaptor     = nullptr;
}

