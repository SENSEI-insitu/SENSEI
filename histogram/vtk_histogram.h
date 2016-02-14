#ifndef VTK_HISTOGRAM_H
#define VTK_HISTOGRAM_H

#include <mpi.h>

class vtkDataArray;

/// Parallel histogram implementation using array-layout independent
/// VTK's Generic Array infrastructure.
void vtk_histogram(MPI_Comm comm, vtkDataArray* array, int bins);

#endif
