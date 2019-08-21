#include <vector>
#include <chrono>
#include <ctime>

#include <format.h>
#include <opts/opts.h>

#include <diy/master.hpp>
#include <diy/decomposition.hpp>
#include <diy/io/bov.hpp>
#include <diy/grid.hpp>
#include <diy/vertices.hpp>
#include <diy/point.hpp>

#include "Oscillator.h"
#include "Particles.h"
#include "Block.h"

#include "senseiConfig.h"
#ifdef ENABLE_SENSEI
#include "bridge.h"
#else
#include "analysis.h"
#endif
#include <Timer.h>


using Grid   = diy::Grid<float,3>;
using Vertex = Grid::Vertex;
using SpPoint = diy::Point<float,3>;
using Bounds = diy::Point<float,6>;

using Link   = diy::RegularGridLink;
using Master = diy::Master;
using Proxy  = Master::ProxyWithLink;

using RandomSeedType = std::default_random_engine::result_type;


int main(int argc, char** argv)
{
    using Time = std::chrono::high_resolution_clock;
    using ms   = std::chrono::milliseconds;

    auto start = Time::now();

    diy::mpi::environment     env(argc, argv);
    diy::mpi::communicator    world;

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
    bool log = ops >> Present("log", "generate full time and memory usage log");
    bool shortlog = ops >> Present("shortlog", "generate a summary time and memory usage log");
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
            fmt::print("Usage: {} [OPTIONS] OSCILLATORS.txt\n\n{}\n", argv[0], ops);
        return 1;
    }

    if (nblocks < world.size())
    {
        if (world.rank() == 0)
        {
            fmt::print("Error: too few blocks\n");
            fmt::print("Usage: {} [OPTIONS] OSCILLATORS.txt\n\n{}\n", argv[0], ops);
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

        diy::mpi::broadcast(world, seed, 0);
    }

    std::default_random_engine rng(static_cast<RandomSeedType>(seed));
    for (int i = 0; i < world.rank(); ++i) rng(); // different seed for each rank


    if (log || shortlog)
        sensei::Timer::Enable(shortlog);

    sensei::Timer::Initialize();
    sensei::Timer::MarkStartEvent("oscillators::initialize");

    std::vector<Oscillator> oscillators;
    if (world.rank() == 0)
    {
        oscillators = read_oscillators(infn);
        diy::MemoryBuffer bb;
        diy::save(bb, oscillators);
        diy::mpi::broadcast(world, bb.buffer, 0);
    }
    else
    {
        diy::MemoryBuffer bb;
        diy::mpi::broadcast(world, bb.buffer, 0);
        diy::load(bb, oscillators);
    }

    if (verbose && (world.rank() == 0))
        for (auto& o : oscillators)
            std::cerr << world.rank() << " center = " << o.center
              << " radius = " << o.radius << " omega0 = " << o.omega0
              << " zeta = " << o.zeta << std::endl;

    diy::Master master(world, threads, -1,
                       &Block::create,
                       &Block::destroy);

    diy::ContiguousAssigner assigner(world.size(), nblocks);

    diy::DiscreteBounds domain;
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

    diy::RegularDecomposer<diy::DiscreteBounds>::BoolVector share_face;
    diy::RegularDecomposer<diy::DiscreteBounds>::BoolVector wrap(3, true);
    diy::RegularDecomposer<diy::DiscreteBounds>::CoordinateVector ghosts = {ghostCells, ghostCells, ghostCells};

    // decompose the domain
    diy::decompose(3, world.rank(), domain, assigner,
                   [&](int gid, const diy::DiscreteBounds &, const diy::DiscreteBounds &bounds,
                       const diy::DiscreteBounds &domain, const Link& link)
                   {
                      Block *b = new Block(gid, bounds, domain, origin,
                        spacing, ghostCells, oscillators, velocity_scale);

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

    sensei::Timer::MarkEndEvent("oscillators::initialize");

    sensei::Timer::MarkStartEvent("oscillators::analysis::initialize");
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
    sensei::Timer::MarkEndEvent("oscillators::analysis::initialize");

    int t_count = 0;
    float t = 0.;
    while (t < t_end)
    {
        sensei::Timer::MarkStartTimeStep(t_count, t);

        if (verbose && (world.rank() == 0))
            std::cerr << "started step = " << t_count << " t = " << t << std::endl;

        sensei::Timer::MarkStartEvent("oscillators::update_fields");
        master.foreach([=](Block* b, const Proxy&)
                              {
                                b->update_fields(t);
                              });
        sensei::Timer::MarkEndEvent("oscillators::update_fields");

        sensei::Timer::MarkStartEvent("oscillators::move_particles");
        master.foreach([=](Block* b, const Proxy& p)
                              {
                                b->move_particles(dt, p);
                              });
        sensei::Timer::MarkEndEvent("oscillators::move_particles");

        sensei::Timer::MarkStartEvent("oscillators::master.exchange");
        master.exchange();
        sensei::Timer::MarkEndEvent("oscillators::master.exchange");

        sensei::Timer::MarkStartEvent("oscillators::handle_incoming_particles");
        master.foreach([=](Block* b, const Proxy& p)
                              {
                                b->handle_incoming_particles(p);
                              });
        sensei::Timer::MarkEndEvent("oscillators::handle_incoming_particles");

        sensei::Timer::MarkStartEvent("oscillators::analysis");
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
        sensei::Timer::MarkEndEvent("oscillators::analysis");

        if (!out_prefix.empty())
        {
            auto out_start = Time::now();

            // Save the corr buffer for debugging
            std::string outfn = fmt::format("{}-{}.bin", out_prefix, t);
            diy::mpi::io::file out(world, outfn, diy::mpi::io::file::wronly | diy::mpi::io::file::create);
            diy::io::BOV writer(out, shape);
            master.foreach([&writer](Block* b, const diy::Master::ProxyWithLink& cp)
                                           {
                                             auto link = static_cast<Link*>(cp.link());
                                             writer.write(link->bounds(), b->grid.data(), true);
                                           });
            if (verbose && (world.rank() == 0))
            {
                auto out_duration = std::chrono::duration_cast<ms>(Time::now() - out_start);
                std::cerr << "Output time for " << outfn << ":" << out_duration.count() / 1000
                    << "." << out_duration.count() % 1000 << " s" << std::endl;
            }
        }

        if (sync)
            world.barrier();

        sensei::Timer::MarkEndTimeStep();

        t += dt;
        ++t_count;
    }

    sensei::Timer::MarkStartEvent("oscillators::finalize");
#ifdef ENABLE_SENSEI
    bridge::finalize();
#else
    analysis_final(k_max, nblocks);
#endif
    sensei::Timer::MarkEndEvent("oscillators::finalize");

    sensei::Timer::Finalize();

    world.barrier();
    if (world.rank() == 0)
    {
        auto duration = std::chrono::duration_cast<ms>(Time::now() - start);
        std::cerr << "Total run time: " << duration.count() / 1000
            << "." << duration.count() % 1000 << " s" << std::endl;
    }

    return 0;
}
