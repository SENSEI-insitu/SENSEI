#ifndef InTransitAdaptorFactory_h
#define InTransitAdaptorFactory_h

#include "InTransitDataAdaptor.h"
#include "AnalysisAdaptor.h"

#include <vtkSmartPointer.h>


namespace sensei
{

using InTransitDataAdaptorPtr = vtkSmartPointer<sensei::InTransitDataAdaptor>;
using AnalysisAdaptorPtr = vtkSmartPointer<sensei::AnalysisAdaptor>;

// the InTransitAdaptorFactory creates a sensei::ConfigurableAnalysis adaptor
// and sensei::InTransitDataAdaptor based on a SENSEI XML config file. The
// factory will be used in in transit end-points.
//
// The factory uses the 'type' attribute to construct the appropriate
// InTransitDataAdaptor instance and initializes it with the data_adaptor
// element. See InTranistDataAdaptor::Initialize for details.
//
// The factory forwards analysis XML to the configurable analysis instance.
//
// The supported transport types are:
//
//   adios_2
//   data_elevators
//   libis
//
// Illustrative example of the XML:
//
// <sensei>
//   <data_adaptor transport="adios_2" partitioner="block" ... >
//     ...
//   </data_adaptor>
//
//   <analysis type="histogram" ... >
//     ...
//   </analysis>
//
//   ...
// <sensei>
namespace InTransitAdaptorFactory
{

int Initialize(const std::string &fileName, MPI_Comm comm,
  AnalysisAdaptor *&analysisAdaptor, InTransitDataAdaptor *&dataAdaptor);

int Initialize(const pugi::xml_node &root, MPI_Comm comm,
  AnalysisAdaptor *&analysisAdaptor, InTransitDataAdaptor *&dataAdaptor);

}

}

#endif
