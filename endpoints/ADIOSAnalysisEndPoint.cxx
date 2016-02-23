/*!
 * This program is designed to be an endpoint component in a scientific
 * workflow. It can read a data-stream using ADIOS-FLEXPATH. When enabled, this end point
 * supports histogram and catalyst-slice analysis via the Sensei infrastructure.
 *
 * Usage:
 *  <exec> input-stream-name
 */
#include <opts/opts.h>
#include <mpi.h>
#include <iostream>
#include <sensei/adios/DataAdaptor.h>
#include <sensei/ConfigurableAnalysis.h>
#include <vtkNew.h>
#include <vtkDataSet.h>

using std::cout;
using std::cerr;
using std::endl;

int main(int argc, char** argv)
{
  int rank, size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Init (&argc, &argv);
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::string input;
  std::string readmethod("bp");
  std::string config_file;

  opts::Options ops(argc, argv);
  ops >> opts::Option('r', "readmethod", readmethod, "specify read method: bp, bp_aggregate, dataspaces, dimes, or flexpath ")
      >> opts::Option('f', "config", config_file, "Sensei analysis configuration xml (required)");

  if (ops >> opts::Present('h', "help", "show help") ||
    !(ops >> opts::PosOption(input)) ||
    config_file.empty())
    {
    if (rank == 0)
      {
      cout << "Usage: " << argv[0] << "[OPTIONS] input-stream-name\n\n" << ops;
      }
    MPI_Barrier(comm);
    return 1;
    }

  std::map<std::string, ADIOS_READ_METHOD> readmethods;
  readmethods["bp"] = ADIOS_READ_METHOD_BP;
  readmethods["bp_aggregate"] = ADIOS_READ_METHOD_BP_AGGREGATE;
  readmethods["dataspaces"] = ADIOS_READ_METHOD_DATASPACES;
  readmethods["dimes"] = ADIOS_READ_METHOD_DIMES;
  readmethods["flexpath"] = ADIOS_READ_METHOD_FLEXPATH;

  vtkNew<sensei::ConfigurableAnalysis> analysis;
  analysis->Initialize(comm, config_file);

  vtkNew<sensei::adios::DataAdaptor> dataAdaptor;
  dataAdaptor->Open(comm, readmethods[readmethod], input);
  do
    {
    // request reading of meta-data for this step.
    dataAdaptor->ReadStep();
    analysis->Execute(dataAdaptor.GetPointer());
    dataAdaptor->ReleaseData();
    }
  while (dataAdaptor->Advance());
  return 0;
}
