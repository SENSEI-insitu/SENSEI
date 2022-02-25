#ifndef InTransitAdaptorFactory_h
#define InTransitAdaptorFactory_h

#include "InTransitDataAdaptor.h"
#include "AnalysisAdaptor.h"

#include <svtkSmartPointer.h>

/// @file

namespace sensei
{

using InTransitDataAdaptorPtr = svtkSmartPointer<sensei::InTransitDataAdaptor>;
using AnalysisAdaptorPtr = svtkSmartPointer<sensei::AnalysisAdaptor>;

/// Factory methods for creating in transit adaptors from XML.
namespace InTransitAdaptorFactory
{
/** Creates a sensei::ConfigurableAnalysis adaptor and
 * sensei::InTransitDataAdaptor based on a SENSEI XML config file. The factory
 * will be used in in transit end-points.
 *
 * The factory uses the 'type' attribute to construct the appropriate
 * InTransitDataAdaptor instance and initializes it with the data_adaptor
 * element. See InTranistDataAdaptor::Initialize for details.
 *
 * The factory forwards analysis XML to the configurable analysis instance.
 *
 * The supported transport types are:
 *
 *   adios1
 *   adios2
 *   hdf5
 *   libis
 *
 * Illustrative example of the XML:
 *
 * ```xml
 * <sensei>
 *   <data_adaptor transport="adios2" partitioner="block" ... >
 *     ...
 *   </data_adaptor>
 *
 *   <analysis type="histogram" ... >
 *     ...
 *   </analysis>
 *
 *   ...
 * <sensei>
 * ```
 */
int Initialize(MPI_Comm comm, const std::string &fileName, InTransitDataAdaptor *&dataAdaptor);

/** Creates a sensei::ConfigurableAnalysis adaptor and
 * sensei::InTransitDataAdaptor based on a SENSEI XML config file. The factory
 * will be used in in transit end-points.
 *
 * The factory uses the 'type' attribute to construct the appropriate
 * InTransitDataAdaptor instance and initializes it with the data_adaptor
 * element. See InTranistDataAdaptor::Initialize for details.
 *
 * The factory forwards analysis XML to the configurable analysis instance.
 *
 * The supported transport types are:
 *
 *   adios_1
 *   adios_2
 *   data_elevators
 *   libis
 *
 * Illustrative example of the XML:
 *
 * ```xml
 * <sensei>
 *   <data_adaptor transport="adios_2" partitioner="block" ... >
 *     ...
 *   </data_adaptor>
 *
 *   <analysis type="histogram" ... >
 *     ...
 *   </analysis>
 *
 *   ...
 * <sensei>
 * ```
 */
int Initialize(MPI_Comm comm, const pugi::xml_node &root, InTransitDataAdaptor *&dataAdaptor);
}

}

#endif
