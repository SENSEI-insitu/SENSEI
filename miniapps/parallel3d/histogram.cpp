#include <vector>
#include <mpi.h>
#include <cstdio>

#include "histogram.h"

void histogram(MPI_Comm comm, double* data, long sz, int bins)
{
    int rank;
    MPI_Comm_rank(comm, &rank);

    // find max and min
    double min = data[0];
    double max = data[0];
    for (long i = 1; i < sz; ++i)
    {
        if (data[i] > max) max = data[i];
        if (data[i] < min) min = data[i];
    }
    double g_min, g_max;

    // Find the global max/min
    MPI_Allreduce(&min, &g_min, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&max, &g_max, 1, MPI_DOUBLE, MPI_MAX, comm);

    //printf("[%d] min: %f, max: %f\n", rank, g_min, g_max);

    double width = (g_max - g_min)/bins;
    std::vector<unsigned>   hist(bins);
    for (long i = 0; i < sz; ++i)
    {
        int idx = int((data[i] - g_min)/width);
        if (idx == bins)        // we hit the max
            --idx;
        //printf("[%d]: %f -> %d\n", rank, data[i], idx);
        ++hist[idx];
    }

    // Global reduce histograms
    std::vector<unsigned> g_hist(bins);
    MPI_Reduce(&hist[0], &g_hist[0], bins, MPI_UNSIGNED, MPI_SUM, 0, comm);

    if (rank == 0)
    {
        printf("Histogram:\n");
        for (int i = 0; i < bins; ++i)
            printf("  %f-%f: %d\n", g_min + i*width, g_min + (i+1)*width, g_hist[i]);
    }
}
