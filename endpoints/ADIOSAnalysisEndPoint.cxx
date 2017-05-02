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
#include <ADIOSDataAdaptor.h>
#include <ConfigurableAnalysis.h>
#include <Timer.h>
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

  bool log = ops >> opts::Present("log", "generate time and memory usage log");
  bool shortlog = ops >> opts::Present("shortlog", "generate a summary time and memory usage log");
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

  timer::SetLogging(log || shortlog);
  timer::SetTrackSummariesOverTime(shortlog);

  std::map<std::string, ADIOS_READ_METHOD> readmethods;
  readmethods["bp"] = ADIOS_READ_METHOD_BP;
  readmethods["bp_aggregate"] = ADIOS_READ_METHOD_BP_AGGREGATE;
  readmethods["dataspaces"] = ADIOS_READ_METHOD_DATASPACES;
  readmethods["dimes"] = ADIOS_READ_METHOD_DIMES;
  readmethods["flexpath"] = ADIOS_READ_METHOD_FLEXPATH;

  vtkSmartPointer<sensei::ConfigurableAnalysis> analysis =
    vtkSmartPointer<sensei::ConfigurableAnalysis>::New();
  analysis->Initialize(comm, config_file);

  cout << "Opening: '" << input.c_str() << "' using '" << readmethod.c_str() << "'" << endl;
  vtkNew<sensei::ADIOSDataAdaptor> dataAdaptor;
  dataAdaptor->Open(comm, readmethods[readmethod], input);
  cout << "Done opening  '" << input.c_str() << "'" << endl;

  int t_count = 0;
  double t = 0.0;
  do
    {
    timer::MarkStartTimeStep(t_count, t);

    timer::MarkStartEvent("adios::advance");
    // request reading of meta-data for this step.
    dataAdaptor->ReadStep();
    timer::MarkEndEvent("adios::advance");

    timer::MarkStartEvent("adios::analysis");
    analysis->Execute(dataAdaptor.GetPointer());
    timer::MarkEndEvent("adios::analysis");

    dataAdaptor->ReleaseData();

    timer::MarkEndTimeStep();
    }
  while (dataAdaptor->Advance());

  timer::MarkStartEvent("adios::finalize");
  analysis = NULL;
  timer::MarkEndEvent("adios::finalize");

  timer::PrintLog(std::cout, comm);
  return 0;
}
