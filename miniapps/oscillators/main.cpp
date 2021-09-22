#include <vector>
#include <chrono>
#include <ctime>

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
    sdiy::mpi::communicator    world;

    using namespace opts;

    Vertex                      shape     = { 64, 64, 64 };
    int                         nblocks   = world.size();
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
        >> Option(     "t-end",  t_end,     "end time")
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
        if (world.rank() == 0)
            std::cerr << "Usage: " << argv[0] << " [OPTIONS] OSCILLATORS.txt\n\n" << ops << std::endl;
        return 1;
    }

    if (nblocks < world.size())
    {
        if (world.rank() == 0)
        {
            std::cerr << "Error: too few blocks\n";
            std::cerr << "Usage: " << argv[0] << " [OPTIONS] OSCILLATORS.txt\n\n" << ops << std::endl;
        }
        return 1;
    }

    if (numberOfParticles < 0)
        numberOfParticles = nblocks*64;

    int particlesPerBlock = numberOfParticles / nblocks;

    if (verbose && (world.rank() == 0))
    {
        std::cerr << world.rank() << " numberOfParticles = " << numberOfParticles << std::endl
            << world.rank() << " particlesPerBlock = " << particlesPerBlock << std::endl;
    }

    if (seed == -1)
    {
        if (world.rank() == 0)
        {
            seed = static_cast<int>(std::time(nullptr));
            if (verbose)
                std::cerr << world.rank() << " seed = " << seed << std::endl;
        }

        sdiy::mpi::broadcast(world, seed, 0);
    }

    std::default_random_engine rng(static_cast<RandomSeedType>(seed));
    for (int i = 0; i < world.rank(); ++i)
        rng(); // different seed for each rank

    sensei::Profiler::StartEvent("oscillators::initialize");

    std::shared_ptr<Oscillator> oscillators;
    unsigned long nOscillators = 0;
    if (world.rank() == 0)
    {
        // read the oscillators
        std::vector<Oscillator> tmp = read_oscillators(infn);

        // distribute
        nOscillators = tmp.size();
        MPI_Bcast(&nOscillators, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        // allocate the buffer for the oscillator array
        unsigned long nBytes = nOscillators*sizeof(Oscillator);
#if defined(OSCILLATOR_CUDA)
        Oscillator *pTmpOsc = nullptr;
        cudaMallocManaged(&pTmpOsc, nBytes);
        oscillators = std::shared_ptr<Oscillator>(pTmpOsc, cudaFree);
#else
        oscillators = std::shared_ptr<Oscillator>((Oscillator*)malloc(nBytes), free);
#endif
        memcpy(oscillators.get(), tmp.data(), nBytes);

        // send the list of oscillators
        MPI_Bcast(oscillators.get(), nBytes, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else
    {
        // recieve the number of oscillators
        MPI_Bcast(&nOscillators, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

        // allocate the buffer for the oscillator array
        unsigned long nBytes = nOscillators*sizeof(Oscillator);

#if defined(OSCILLATOR_CUDA)
        Oscillator *pTmpOsc = nullptr;
        cudaMallocManaged(&pTmpOsc, nBytes);
        oscillators = std::shared_ptr<Oscillator>(pTmpOsc, cudaFree);
#else
        oscillators = std::shared_ptr<Oscillator>((Oscillator*)malloc(nBytes), free);
#endif

        // recieve the list of oscillators
        MPI_Bcast(oscillators.get(), nBytes, MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    // dump the list of oscillators
    if (verbose && (world.rank() == 0))
    {
        const Oscillator *pOsc = oscillators.get();
        for (unsigned long q = 0; q < nOscillators; ++q)
        {
            std::cerr << world.rank() << " center = " << pOsc[q].center_x << ", "
              << pOsc[q].center_y << ", " << pOsc[q].center_z
              << " radius = " << pOsc[q].radius << " omega0 = " << pOsc[q].omega0
              << " zeta = " << pOsc[q].zeta << std::endl;
        }
    }

    sdiy::Master master(world, threads, -1,
                       &Block::create,
                       &Block::destroy);

    sdiy::ContiguousAssigner assigner(world.size(), nblocks);

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

    if (verbose && (world.rank() == 0))
      {
      std::cerr << world.rank() << " domain = " << domain.min
         << ", " << domain.max << std::endl
         << world.rank() << "bounds = " << bounds << std::endl
         << world.rank() << "origin = " << origin << std::endl
         << world.rank() << "spacing = " << spacing << std::endl;
      }

    // record various parameters to initialize analysis
    std::vector<int> gids,
                     from_x, from_y, from_z,
                     to_x,   to_y,   to_z;

    sdiy::RegularDecomposer<sdiy::DiscreteBounds>::BoolVector share_face;
    sdiy::RegularDecomposer<sdiy::DiscreteBounds>::BoolVector wrap(3, true);
    sdiy::RegularDecomposer<sdiy::DiscreteBounds>::CoordinateVector ghosts = {ghostCells, ghostCells, ghostCells};

    // decompose the domain
    sdiy::decompose(3, world.rank(), domain, assigner,
                   [&](int gid, const sdiy::DiscreteBounds &, const sdiy::DiscreteBounds &bounds,
                       const sdiy::DiscreteBounds &domain, const Link& link)
                   {
                      Block *b = new Block(gid, bounds, domain, origin,
                        spacing, ghostCells, oscillators, nOscillators,
                        velocity_scale);

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
                          std::cerr << world.rank() << " Block " << *b << std::endl;
                   },
                   share_face, wrap, ghosts);

    sensei::Profiler::EndEvent("oscillators::initialize");

    sensei::Profiler::StartEvent("oscillators::analysis::initialize");
#ifdef ENABLE_SENSEI
    bridge::initialize(nblocks, gids.size(), origin.data(), spacing.data(),
                       domain.max[0] + 1, domain.max[1] + 1, domain.max[2] + 1,
                       &gids[0],
                       &from_x[0], &from_y[0], &from_z[0],
                       &to_x[0],   &to_y[0],   &to_z[0],
                       &shape[0], ghostCells,
                       config_file);
#else
    init_analysis(world, window, gids.size(),
                  domain.max[0] + 1, domain.max[1] + 1, domain.max[2] + 1,
                  &gids[0],
                  &from_x[0], &from_y[0], &from_z[0],
                  &to_x[0],   &to_y[0],   &to_z[0]);
#endif
    sensei::Profiler::EndEvent("oscillators::analysis::initialize");

    int t_count = 0;
    float t = 0.;
    while (t < t_end)
    {
        if (verbose && (world.rank() == 0))
            std::cerr << "started step = " << t_count << " t = " << t << std::endl;

        sensei::Profiler::StartEvent("oscillators::update_fields");
        master.foreach([=](Block* b, const Proxy&)
                              {
                                b->update_fields(t);
                              });
        sensei::Profiler::EndEvent("oscillators::update_fields");

        sensei::Profiler::StartEvent("oscillators::update_particles");
        master.foreach([=](Block* b, const Proxy&)
                              {
                                b->update_particles(t);
                              });
        sensei::Profiler::EndEvent("oscillators::update_particles");

        sensei::Profiler::StartEvent("oscillators::move_particles");
        master.foreach([=](Block* b, const Proxy& p)
                              {
                                b->move_particles(dt, p);
                              });
        sensei::Profiler::EndEvent("oscillators::move_particles");

        sensei::Profiler::StartEvent("oscillators::master.exchange");
        master.exchange();
        sensei::Profiler::EndEvent("oscillators::master.exchange");

        sensei::Profiler::StartEvent("oscillators::handle_incoming_particles");
        master.foreach([=](Block* b, const Proxy& p)
                              {
                                b->handle_incoming_particles(p);
                              });
        sensei::Profiler::EndEvent("oscillators::handle_incoming_particles");

        sensei::Profiler::StartEvent("oscillators::analysis");
#ifdef ENABLE_SENSEI
        // do the analysis using sensei
        // update data adaptor with new data
        master.foreach([=](Block* b, const Proxy&)
                              {
                              bridge::set_data(b->gid, b->grid.data());
                              bridge::set_particles(b->gid, b->particles);
                              });
        // push data to sensei
        bridge::execute(t_count, t);
#else
        // do the analysis without using sensei
        // call the analysis function for each block
        master.foreach([=](Block* b, const Proxy&)
                              {
                              analyze(b->gid, b->grid.data());
                              });
#endif
        sensei::Profiler::EndEvent("oscillators::analysis");

        if (!out_prefix.empty())
        {
            auto out_start = Time::now();

            // Save the corr buffer for debugging
            std::ostringstream outfn;
            outfn << out_prefix << "-" << t << ".bin";

            sdiy::mpi::io::file out(world, outfn.str(), sdiy::mpi::io::file::wronly | sdiy::mpi::io::file::create);
            sdiy::io::BOV writer(out, shape);
            master.foreach([&writer](Block* b, const sdiy::Master::ProxyWithLink& cp)
                                           {
                                             auto link = static_cast<Link*>(cp.link());
                                             writer.write(link->bounds(), b->grid.data(), true);
                                           });
            if (verbose && (world.rank() == 0))
            {
                auto out_duration = std::chrono::duration_cast<ms>(Time::now() - out_start);
                std::cerr << "Output time for " << outfn.str() << ":" << out_duration.count() / 1000
                    << "." << out_duration.count() % 1000 << " s" << std::endl;
            }
        }

        if (sync)
            world.barrier();

        t += dt;
        ++t_count;
    }

    sensei::Profiler::StartEvent("oscillators::finalize");
#ifdef ENABLE_SENSEI
    bridge::finalize();
#else
    analysis_final(k_max, nblocks);
#endif
    sensei::Profiler::EndEvent("oscillators::finalize");

    world.barrier();
    if (world.rank() == 0)
    {
        auto duration = std::chrono::duration_cast<ms>(Time::now() - start);
        std::cerr << "Total run time: " << duration.count() / 1000
            << "." << duration.count() % 1000 << " s" << std::endl;
    }

    return 0;
}
