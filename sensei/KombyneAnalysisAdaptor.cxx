#include <iostream>
#include <unistd.h>

#include "KombyneAnalysisAdaptor.h"

#include <kombyne_execution.h>


namespace sensei
{

//-----------------------------------------------------------------------------
senseiNewMacro(KombyneAnalysisAdaptor);

//-----------------------------------------------------------------------------
void KombyneAnalysisAdaptor::Initialize()
{
  kb_role newrole; // will be the same as the input role
  MPI_Comm comm, split;
  int rank, size;

  comm = this->GetCommunicator();
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::cout << "KombyneAnalysisAdaptor::Initialize()" << std::endl;
  std::cout << "    config: " << this->configFile << std::endl;
  std::cout << "    log: " << this->logFile << std::endl;
  std::cout << "    mode: " << this->mode << std::endl;
  std::cout << "    verbose: " << (this->verbose != 0) << std::endl;

  kb_initialize(comm,
      "KombyneAnalysisAdaptor",
      "Kombyne analysis for SENSEI",
      KB_ROLE_SIMULATION_AND_ANALYSIS, // in situ
      size, size,
      "session.txt",
      &split,
      &newrole);
  exit(0);
}

//-----------------------------------------------------------------------------
/// Invoke in situ processing using Kombyne
bool KombyneAnalysisAdaptor::Execute(DataAdaptor* data, DataAdaptor**)
{
  std::cout << "KombyneAnalysisAdaptor::Execute()" << std::endl;
  return true;
}

//-----------------------------------------------------------------------------
/// Shuts Kombyne down.
int KombyneAnalysisAdaptor::Finalize()
{
  std::cout << "KombyneAnalysisAdaptor::Finalize()" << std::endl;
  kb_finalize();
  return 0;
}

int KombyneAnalysisAdaptor::SetConfigFile(std::string cfgfile)
{
  std::cout << "KombyneAnalysisAdaptor::SetConfigFile(\""
    << cfgfile << "\")" << std::endl;
  this->configFile = cfgfile;
  return 0;
}

int KombyneAnalysisAdaptor::SetMode(std::string mode)
{
  std::cout << "KombyneAnalysisAdaptor::SetMode(\""
    << mode << "\")" << std::endl;
  this->mode = mode;
  return 0;
}

void KombyneAnalysisAdaptor::SetVerbose(int verbose)
{
  std::cout << "KombyneAnalysisAdaptor::SetVerbose("
    << verbose << ")" << std::endl;
  this->verbose = (verbose != 0);
}

//-----------------------------------------------------------------------------
KombyneAnalysisAdaptor::KombyneAnalysisAdaptor()
{
}

//-----------------------------------------------------------------------------
KombyneAnalysisAdaptor::~KombyneAnalysisAdaptor()
{
}

}
