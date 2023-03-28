#include <vector>
#include <chrono>
#include <ctime>
#include <memory>

#include <opts/opts.h>

#include <sdiy/master.hpp>
#include <sdiy/decomposition.hpp>
#include <sdiy/io/bov.hpp>
#include <sdiy/grid.hpp>
#include <sdiy/vertices.hpp>
#include <sdiy/point.hpp>

#include "Oscillator.h"
#include "Particles.h"
#include "Block.h"

#include "MemoryUtils.h"
#include "senseiConfig.h"
#include "MPIManager.h"
#ifdef ENABLE_SENSEI
#include "bridge.h"
#else
#include "analysis.h"
#endif

#include <Profiler.h>
#include <DataAdaptor.h>

using sensei::Profiler;
using sensei::TimeEvent;

using Grid   = sdiy::Grid<float,3>;
using Vertex = Grid::Vertex;
using SpPoint = sdiy::Point<float,3>;
using Bounds = sdiy::Point<float,6>;

using Link   = sdiy::RegularGridLink;
using Master = sdiy::Master;
using Proxy  = Master::ProxyWithLink;

using RandomSeedType = std::default_random_engine::result_type;

using Time = std::chrono::high_resolution_clock;
using ms   = std::chrono::milliseconds;

int main(int argc, char** argv)
{
    sensei::MPIManager mpiMan(argc, argv);
    auto start = Time::now();

    //sdiy::mpi::environment     env(argc, argv);
    sdiy::mpi::communicator comm;

    Profiler::SetCommunicator(comm);
    Profiler::Initialize();
    Profiler::StartEvent("oscillators::initialize");

    using namespace opts;

    Vertex                      shape     = { 64, 64, 64 };
    int                         nblocks   = comm.size();
    float                       t_end     = 10;
    float                       dt        = .01;
    float                       velocity_scale = 50.0f;
#ifndef ENABLE_SENSEI
    size_t                      window    = 10;
    size_t                      k_max     = 3;
#endif
    int                         threads   = 1;
    int                         ghostCells = 1;
    int                         numberOfParticles = 0;
    int                         seed = 0x240dc6a9;
    std::string                 config_file;
    std::string                 out_prefix = "";
    Bounds                      bounds{0.,-1.,0.,-1.,0.,-1.};

    Options ops(argc, argv);
    ops
        >> Option('b', "blocks", nblocks,   "number of blocks to use. must greater or equal to number of MPI ranks.")
        >> Option('s', "shape",  shape,     "global number of cells in the domain")
        >> Option('e', "bounds", bounds,    "global bounds of the domain")
        >> Option('t', "dt",     dt,        "time step")
#ifdef ENABLE_SENSEI
        >> Option('f', "config", config_file, "Sensei analysis configuration xml (required)")
#else
        >> Option('w', "window", window,    "analysis window")
        >> Option('k', "k-max",  k_max,     "number of strongest autocorrelations to report")
#endif
        >> Option('E', "t-end",  t_end,     "end time")
        >> Option('j', "jobs",   threads,   "number of threads to use")
        >> Option('o', "output", out_prefix, "prefix to save output")
        >> Option('g', "ghost-cells", ghostCells, "number of ghost cells")
        >> Option('p', "particles", numberOfParticles, "number of random particles to generate")
        >> Option('v', "v-scale", velocity_scale, "scale factor to convert function gradient to velocity")
        >> Option(     "seed", seed, "specify a random seed")
    ;
    bool sync = ops >> Present("sync", "synchronize after each time step");
    bool verbose = ops >> Present("verbose", "print debugging messages");

    std::string infn;
    if (  ops >> Present('h', "help", "show help") ||
        !(ops >> PosOption(infn))
#ifdef ENABLE_SENSEI
        || config_file.empty()
#endif
        )
    {
        if (comm.rank() == 0)
            std::cerr << "Usage: " << argv[0] << " [OPTIONS] OSCILLATORS.txt\n\n" << ops << std::endl;
        return 1;
    }

    if (nblocks < comm.size())
    {
        if (comm.rank() == 0)
        {
            std::cerr << "Error: too few blocks\n";
            std::cerr << "Usage: " << argv[0] << " [OPTIONS] OSCILLATORS.txt\n\n" << ops << std::endl;
        }
        return 1;
    }

    if (numberOfParticles < 0)
        numberOfParticles = nblocks*64;

    int particlesPerBlock = numberOfParticles / nblocks;

    if (verbose && (comm.rank() == 0))
    {
        std::cerr << comm.rank() << " numberOfParticles = " << numberOfParticles << std::endl
            << comm.rank() << " particlesPerBlock = " << particlesPerBlock << std::endl;
    }

    if (seed == -1)
    {
        if (comm.rank() == 0)
        {
            seed = static_cast<int>(std::time(nullptr));
            if (verbose)
                std::cerr << comm.rank() << " seed = " << seed << std::endl;
        }

        sdiy::mpi::broadcast(comm, seed, 0);
    }

    std::default_random_engine rng(static_cast<RandomSeedType>(seed));
    for (int i = 0; i < comm.rank(); ++i)
        rng(); // different seed for each rank


    // read the oscillators from disk
    OscillatorArray oscillators;
    oscillators.Initialize(comm, infn);

    if (verbose && (comm.rank() == 0))
    {
      std::cerr << "oscillators read from disk" << std::endl;
      oscillators.Print(std::cerr);
    }

    // initialize diy
    sdiy::Master master(comm, threads, -1,
                       &Block::create,
                       &Block::destroy);

    sdiy::ContiguousAssigner assigner(comm.size(), nblocks);

    sdiy::DiscreteBounds domain;
    domain.min[0] = domain.min[1] = domain.min[2] = 0;
    for (unsigned i = 0; i < 3; ++i)
      domain.max[i] = shape[i] - 1;

    SpPoint origin{0.,0.,0.};
    SpPoint spacing{1.,1.,1.};
    if (bounds[1] >= bounds[0])
    {
      // valid bounds specififed on the command line, calculate the
      // global origin and spacing.
      for (int i = 0; i < 3; ++i)
        origin[i] = bounds[2*i];

      for (int i = 0; i < 3; ++i)
        spacing[i] = (bounds[2*i+1] - bounds[2*i])/shape[i];
    }

    if (verbose && (comm.rank() == 0))
    {
      std::cerr << comm.rank() << " domain = " << domain.min
         << ", " << domain.max << std::endl
         << comm.rank() << "bounds = " << bounds << std::endl
         << comm.rank() << "origin = " << origin << std::endl
         << comm.rank() << "spacing = " << spacing << std::endl;
    }

    // record various parameters to initialize analysis
    std::vector<int> gids,
                     from_x, from_y, from_z,
                     to_x,   to_y,   to_z;

    sdiy::RegularDecomposer<sdiy::DiscreteBounds>::BoolVector share_face;
    sdiy::RegularDecomposer<sdiy::DiscreteBounds>::BoolVector wrap(3, true);
    sdiy::RegularDecomposer<sdiy::DiscreteBounds>::CoordinateVector ghosts = {ghostCells, ghostCells, ghostCells};

    // decompose the domain
    sdiy::decompose(3, comm.rank(), domain, assigner,
                   [&](int gid, const sdiy::DiscreteBounds &, const sdiy::DiscreteBounds &bounds,
                       const sdiy::DiscreteBounds &domain, const Link& link)
                   {
                      Block *b = new Block(gid, bounds, domain, origin,
                        spacing, ghostCells, velocity_scale);

                      // generate particles
                      int start = particlesPerBlock * gid;
                      int count = particlesPerBlock;
                      if ((start + count) > numberOfParticles)
                        count = std::max(0, numberOfParticles - start);

                      b->particles = GenerateRandomParticles<float>(rng,
                        b->domain, b->bounds, b->origin, b->spacing, b->nghost,
                        start, count);

                      master.add(gid, b, new Link(link));

                      gids.push_back(gid);

                      from_x.push_back(bounds.min[0]);
                      from_y.push_back(bounds.min[1]);
                      from_z.push_back(bounds.min[2]);

                      to_x.push_back(bounds.max[0]);
                      to_y.push_back(bounds.max[1]);
                      to_z.push_back(bounds.max[2]);

                      if (verbose)
                          std::cerr << comm.rank() << " Block " << *b << std::endl;
                   },
                   share_face, wrap, ghosts);

    Profiler::EndEvent("oscillators::initialize");

#ifdef ENABLE_SENSEI
    {
    TimeEvent<128>("oscillators::initialize_in_situ");
    bridge::initialize(nblocks, gids.size(), origin.data(), spacing.data(),
                       domain.max[0] + 1, domain.max[1] + 1, domain.max[2] + 1,
                       &gids[0],
                       &from_x[0], &from_y[0], &from_z[0],
                       &to_x[0],   &to_y[0],   &to_z[0],
                       &shape[0], ghostCells,
                       config_file);
    }
#endif

    int t_count = 0;
    float t = 0.;
    while (t < t_end)
    {
        if (verbose && (comm.rank() == 0))
            std::cerr << "started step = " << t_count << " t = " << t << std::endl;

        {
        TimeEvent<128>("oscillators::solve");

        master.foreach([&](Block* b, const Proxy&)
                              {
                                b->update_fields(t, oscillators);
                              });

        master.foreach([&](Block* b, const Proxy&)
                              {
                                b->update_particles(t, oscillators);
                              });

        master.foreach([&](Block* b, const Proxy& p)
                              {
                                b->move_particles(dt, p);
                              });

        master.exchange();

        master.foreach([=](Block* b, const Proxy& p)
                              {
                                b->handle_incoming_particles(p);
                              });
        }
#ifdef ENABLE_SENSEI
        {
        TimeEvent<128> event("oscillators::invoke_in_situ");
        // do the analysis using sensei
        // update data adaptor with new data
        bridge::set_oscillators(oscillators);
        master.foreach([=](Block* b, const Proxy&)
                              {
                              bridge::set_data(b->gid, b->grid.data());
                              bridge::set_particles(b->gid, b->particles);
                              });

        // invoke in situ processing
        sensei::DataAdaptor *daOut = nullptr;
        bridge::execute(t_count, t, &daOut);

        // If the analysis modified the oscillators, process the updates.
        if (daOut)
        {
          oscillators.Initialize(comm, daOut);
          daOut->ReleaseData();
          daOut->Delete();

          if (verbose && (comm.rank() == 0))
          {
            std::cerr << "oscillators returned from in situ analysis" << std::endl;
            oscillators.Print(std::cerr);
          }
        }
        }
#endif
        if (!out_prefix.empty())
        {
            TimeEvent<128> event("oscillators::write");

            // Save the corr buffer for debugging
            std::ostringstream outfn;
            outfn << out_prefix << "-" << t << ".bin";

            sdiy::mpi::io::file out(comm, outfn.str(), sdiy::mpi::io::file::wronly | sdiy::mpi::io::file::create);
            sdiy::io::BOV writer(out, shape);
            master.foreach([&writer](Block* b, const sdiy::Master::ProxyWithLink& cp)
                                           {
                                             auto link = static_cast<Link*>(cp.link());
                                             writer.write(link->bounds(), b->grid.data(), true);
                                           });
        }

        if (sync)
            comm.barrier();

        t += dt;
        ++t_count;
    }

#ifdef ENABLE_SENSEI
    {
    TimeEvent<128> event("oscillators::in_situ_finalize");
    bridge::finalize();
    }
#endif

    Profiler::Finalize();

    comm.barrier();
    if (comm.rank() == 0)
    {
        auto duration = std::chrono::duration_cast<ms>(Time::now() - start);
        std::cerr << "Total run time: " << duration.count() / 1000
            << "." << duration.count() % 1000 << " s" << std::endl;
    }

    return 0;
}
