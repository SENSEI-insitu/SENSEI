#include <iostream>
#include <unistd.h>

#include "KombyneAnalysisAdaptor.h"

#include <kombyne_data.h>
#include <kombyne_execution.h>


namespace sensei
{

//-----------------------------------------------------------------------------
senseiNewMacro(KombyneAnalysisAdaptor);

//-----------------------------------------------------------------------------
KombyneAnalysisAdaptor::KombyneAnalysisAdaptor() :
  pipelineFile("kombyne.yaml"),
  sessionName("kombyne-session"),
  role(KB_ROLE_SIMULATION_AND_ANALYSIS),
  verbose(false),
  initialized(false),
  hp(KB_HANDLE_NULL)
{
}

//-----------------------------------------------------------------------------
KombyneAnalysisAdaptor::~KombyneAnalysisAdaptor()
{
  if (this->hp != KB_HANDLE_NULL)
    kb_pipeline_collection_free(this->hp);
}

//-----------------------------------------------------------------------------
void KombyneAnalysisAdaptor::Initialize()
{
  MPI_Comm comm;
  int rank, size;

  comm = this->GetCommunicator();
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::cout << "[GAA]: KombyneAnalysisAdaptor::Initialize()" << std::endl;
  std::cout << "    pipeline: " << this->pipelineFile << std::endl;
  if (this->role == KB_ROLE_SIMULATION)
    std::cout << "    mode: in-transit" << std::endl;
  else
    std::cout << "    mode: in-situ" << std::endl;
  std::cout << "    verbose: " << (this->verbose != 0) << std::endl;

  kb_role newrole;  // will be the same as the input role
  MPI_Comm split;   // will be the same as the input communicator

  kb_initialize(comm,
      "KombyneAnalysisAdaptor",
      "Kombyne analysis adaptor for SENSEI",
      this->role,
      size, size,
      this->sessionName.c_str(),
      &split,
      &newrole);

  // Create a pipeline collection
  this->hp = kb_pipeline_collection_alloc();
  if (this->hp == KB_HANDLE_NULL)
  {
    SENSEI_ERROR("Failed to allocate Kombyne pipeline collection.");
    return;
  }

  kb_pipeline_collection_set_filename(this->hp, this->pipelineFile.c_str());
  if (kb_pipeline_collection_initialize(this->hp) != KB_RETURN_OKAY)
  {
    SENSEI_ERROR("Failed to initialize Kombyne pipeline collection.");
    kb_pipeline_collection_free(this->hp);
    this->hp = KB_HANDLE_NULL;
    return;
  }

  this->initialized = true;
}

//-----------------------------------------------------------------------------
/// Invoke in situ processing using Kombyne
bool KombyneAnalysisAdaptor::Execute(DataAdaptor* data, DataAdaptor**)
{
  std::cout << "[GAA]: KombyneAnalysisAdaptor::Execute()" << std::endl;

  kb_pipeline_data_handle hpd;
  kb_mesh_handle hmesh;
  kb_return ierr;

  if (this->hp == KB_HANDLE_NULL)
    return false; // we already printed an error during initialization

  // Assemble a pipeline_data
  hpd = kb_pipeline_data_alloc();
  if (hpd == KB_HANDLE_NULL)
  {
    SENSEI_ERROR("Failed to allocate Kombyne pipeline data.");
    return false;
  }

  // Expose simulation data to Kombyne here

  int promises = KB_PROMISE_STATIC_FIELDS | KB_PROMISE_STATIC_GRID;
  ierr = kb_pipeline_data_set_promises(hpd, promises);

  // Add a sample for the simulation iteration time.
  //gaatmp
  static float sit = 0.;
  kb_add_sample("Simulation Iteration", sit);
  sit += 1.0;

  // Execute the simulation side of things.
  kb_simulation_execute(this->hp, hpd, nullptr);
#if 0
  //gaatmp - do we do this explicitly if running in-situ?
  // Execute the analysis end of things.
  kb_analysis_execute(this->hp);
#endif

  // Free the pipeline data.
  kb_pipeline_data_free(hpd);

  return true;
}

//-----------------------------------------------------------------------------
/// Shuts Kombyne down.
int KombyneAnalysisAdaptor::Finalize()
{
  std::cout << "[GAA]: KombyneAnalysisAdaptor::Finalize()" << std::endl;
  // free stuff here?
  kb_finalize();
  this->initialized = false;
  return 0;
}

//-----------------------------------------------------------------------------
int KombyneAnalysisAdaptor::SetPipelineFile(std::string filename)
{
  this->pipelineFile = filename;
  return 0;
}

//-----------------------------------------------------------------------------
int KombyneAnalysisAdaptor::SetSessionName(std::string sessionname)
{
  this->sessionName = sessionname;
  return 0;
}

//-----------------------------------------------------------------------------
int KombyneAnalysisAdaptor::SetMode(std::string mode)
{
  if (mode == "in-transit")
    this->role = KB_ROLE_SIMULATION;
  else if (mode == "in-situ")
    this->role = KB_ROLE_SIMULATION_AND_ANALYSIS;
  else
  {
    SENSEI_ERROR("Unknown mode: \"" << mode << "\".  "
        "Expected in-situ or in-transit.");
    return -1;
  }
  return 0;
}

//-----------------------------------------------------------------------------
void KombyneAnalysisAdaptor::SetVerbose(int verbose)
{
  this->verbose = (verbose != 0);
}

}
