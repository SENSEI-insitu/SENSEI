#pragma once
#include <mpi.h>
#include <string>

namespace lammpsBridge
{
  void Initialize(MPI_Comm world, const std::string& config_file);
  void SetData(long ntimestep, int nlocal, int *id, 
               int    nghost, int *type, double **x, 
               double xsublo, double xsubhi, 
               double ysublo, double ysubhi, 
               double zsublo, double zsubhi );
  void Analyze();
  void Finalize();
}
