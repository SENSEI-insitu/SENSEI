#ifndef InTransitAdaptorFactory_h
#define InTransitAdaptorFactory_h

#include "DataAdaptor.h"
#include "AnalysisAdaptor.h"

#include <vtkSmartPointer.h>


namespace sensei
{

using DataAdaptorPtr = vtkSmartPointer<sensei::DataAdaptor>;
using AnalysisAdaptorPtr = vtkSmartPointer<sensei::AnalysisAdaptor>;

// the InTransitAdaptorFactory creates pairs of sensei::DataAdaptors and
// sensei::AnalysisAdaptors based on a SENSEI XML config file. The factory
// will be used in in transit end-points where each analysis adaptor has
// associated with it a data adaptor.
//
// The factory constructs a sensei::ConfigurableAnalysisAdaptor instance
// for each analysis element in the XML config. The analysis adaptor instance
// is initialized as usual by the ConfigurableAnalaysis XML parser.
//
// The factory expects each analysis element in the XML file to nest a
// data_adaptor element that names the desired sensei::InTransitDataAdaptor
// class. The factory constructs an instance of the named data adaptor
// and initializes it with the data_adaptor element. See
// InTranistDataAdaptor::Initialize for details.
//
// A list of pairs of adaptors is returned, a zero return indicates success.
//
// <analysis type="blah" ... >
//
//   <data_adaptor transport="blah" ... >
//     ...
//   </data_adaptor>
//
//   ...
//
// </analysis>
namespace InTransitAdaptorFactory
{

int Initialize(const std::string &fileName,
  std::vector<std::pair<AnalysisAdaptorPtr, DataAdaptorPtr>> &adaptors);

int Initialize(const pugi::xml_node &root,
  std::vector<std::pair<AnalysisAdaptorPtr, DataAdaptorPtr>> &adaptors);

}

}

#endif
