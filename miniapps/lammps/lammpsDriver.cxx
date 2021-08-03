/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   www.cs.sandia.gov/~sjplimp/lammps.html
   Steve Plimpton, sjplimp@sandia.gov, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under 
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#include <mpi.h>

// LAMMPS include files
#include "library.h"         
#include "domain.h"         
#include "lammps.h"
#include "modify.h"
#include "fix.h"
#include "fix_external.h"

// SENSEI bridge 
#include "lammpsBridge.h"   

using namespace LAMMPS_NS;

struct Info {
  int me;
  LAMMPS *lmp;
} globalInfo;

void insituCallback(void *ptr, bigint ntimestep, int nlocal, 
                    int *id, double **x, double **f) 
{
  (void) x;
  (void) f;
 
  Info *info = (Info *) ptr;

  double xsublo = info->lmp->domain->sublo[0];
  double xsubhi = info->lmp->domain->subhi[0];
  double ysublo = info->lmp->domain->sublo[1];
  double ysubhi = info->lmp->domain->subhi[1];
  double zsublo = info->lmp->domain->sublo[2];
  double zsubhi = info->lmp->domain->subhi[2];

  const char *typ = "type";
  int *type = (int *)lammps_extract_atom(info->lmp,(char *)typ);

  const char *ng = "nghost";
  int *nghost = (int*)lammps_extract_global(info->lmp,(char *)ng);

  // The arrays we get through extract atom include the ghost atoms as well
  const char *xx = "x";
  double **all_pos = (double**)lammps_extract_atom(info->lmp,(char *)xx);

  lammpsBridge::SetData(ntimestep, nlocal, id, *nghost, type, all_pos, 
                        xsublo, xsubhi, ysublo, ysubhi, zsublo, zsubhi);

  if (0 == globalInfo.me)
    std::cout << "###### SENSEI instrumentation: bridge analyze() ######" << std::endl;    
  lammpsBridge::Analyze();

  if (0 == globalInfo.me)
    std::cout << "###### SENSEI instrumentation: after bridge analyze() ######" << std::endl;    
 


}

const static std::string USAGE =
"Usage: ./driver <lammps input> [options] [-lmp <lammps args>]\n"
"Options:\n"
"  -h                  Print this help text\n"
"  -sensei <file.xml>  Pass this to set the SENSEI xml config file\n"
"  -n <int>            Specify the number of steps to simulate for. Default is 10000\n"
"  -lmp <lammps args>  Pass the list of arguments <args> to LAMMPS as if they were\n"
"                      command line args to LAMMPS. This must be the last argument, all\n"
"                      following arguments will be passed to lammps.\n"
"  -log                Generate full time and memory usage log.\n"
"  -shortlog           Generate a summary time and memory usage log.";

int main(int argc, char **argv) 
{
  std::vector<std::string> args(argv, argv + argc);
  if (args.size() < 2) 
    {
    std::cout << USAGE << "\n";
    return 1;
    }
  
  std::string lammps_input = args[1];
	
  // Additional args the user may be passing to lammps
  std::vector<char*> lammps_args(1, argv[0]);
  bool log = false;
  bool shortlog = false;
  std::string sensei_xml;
  size_t sim_steps = 10000;
  for (size_t i = 2; i < args.size(); ++i) 
    {
    if (args[i] == "-sensei") 
      sensei_xml = args[++i];
    else if (args[i] == "-n" ) 
      sim_steps = std::stoull(args[++i]);
    else if (args[i] == "-h") 
      {
      std::cout << USAGE << "\n";
      return 0;
      } 
    else if (args[i] == "-log" ) 
      log = true;
    else if (args[i] == "-shortlog" ) 
      shortlog = true;
    else if (args[i] == "-lmp") 
      {
      ++i;
      for (; i < args.size(); ++i) 
	lammps_args.push_back(&args[i][0]);
      break;
      }
    }

  MPI_Init(&argc, &argv);
  MPI_Comm sim_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(sim_comm, &globalInfo.me);

  // Initialize SENSEI bridge 
  if (0 == globalInfo.me) 
    std::cout << "###### SENSEI instrumentation: initialize bridge ######" << std::endl;
  lammpsBridge::Initialize(sim_comm, sensei_xml );

  LAMMPS *lammps;
  lammps_open(lammps_args.size(), lammps_args.data(), sim_comm, (void**)&lammps);
  globalInfo.lmp = lammps;

  // run the input script thru LAMMPS one line at a time until end-of-file
  // driver proc 0 reads a line, Bcasts it to all procs
  // (could just send it to proc 0 of comm_lammps and let it Bcast)
  // all LAMMPS procs call lammps_command() on the line

  if (0 == globalInfo.me) 
    {
    std::cout << "Loading lammps input: '" << lammps_input << "'\n";
    std::ifstream input(lammps_input.c_str());
    for (std::string line; std::getline(input, line);) 
      {
      int len = line.size();
      
      // Skip empty lines
      if (len == 0) 
        continue;
      
      MPI_Bcast(&len, 1, MPI_INT, 0, sim_comm);
      MPI_Bcast(&line[0], len, MPI_CHAR, 0, sim_comm);
                lammps_command(lammps, &line[0]);
      }
      
    // Bcast out we're done with the file
    int len = 0;
    MPI_Bcast(&len, 1, MPI_INT, 0, sim_comm);
    } 
  else 
    {
    while (true) 
      {
      int len = 0;
      MPI_Bcast(&len, 1, MPI_INT, 0, sim_comm);
      if (len == 0) 
        break;
      else 
        {
	std::vector<char> line(len + 1, '\0');
	MPI_Bcast(line.data(), len, MPI_CHAR, 0, sim_comm);
		  lammps_command(lammps, line.data());
	}
      }
    }

  // Setup the fix external callback
  int ifix = lammps->modify->find_fix_by_style("external");
	
  // If there's no external fix, abort
  if (ifix == -1) 
    {
    if (0 == globalInfo.me) 
      std::cout << "You need to add a fix external line to the input file. Abort" << std::endl;
    return -1;
    }
  FixExternal *fix = (FixExternal*)lammps->modify->fix[ifix];
  fix->set_callback(insituCallback, &globalInfo);

  // run for a number of steps
  for (size_t i = 0; i < sim_steps; ++i) 
    {
    const char * string = "run 1 pre no post no";
    lammps_command(lammps, (char *)string);
    }

  // all LAMMPS timesteps computed
  lammps_close(lammps);

  // Finalize SENSEI bridge 
  if (0 == globalInfo.me) 
    std::cout << "###### SENSEI instrumentation: finalize bridge ######" << std::endl;
  lammpsBridge::Finalize();

  // close down MPI
  MPI_Finalize();

  return 0;
}
