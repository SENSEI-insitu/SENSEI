#include <vector>
#include <chrono>
#include <ctime>

#include <format.h>
#include <opts/opts.h>

#include <diy/master.hpp>
#include <diy/decomposition.hpp>
#include <diy/io/bov.hpp>

#include <grid/grid.h>
#include <grid/vertices.h>

#include "oscillator.h"
#include "particles.h"

#include "senseiConfig.h"
#ifdef ENABLE_SENSEI
#include "bridge.h"
#include "MemoryProfiler.h"
#else
#include "analysis.h"
#endif

#include <timer/Timer.h>


using Grid   = grid::Grid<float,3>;
using Vertex = Grid::Vertex;

using Bounds = diy::DiscreteBounds;
using Link   = diy::RegularGridLink;
using Master = diy::Master;
using Proxy  = Master::ProxyWithLink;

using RandomSeedType = std::default_random_engine::result_type;


inline bool IsVertexInsideBounds(const Vertex& v, const Bounds& b)
{
    return (v[0] >= static_cast<float>(b.min[0])) && (v[0] <= static_cast<float>(b.max[0])) &&
           (v[1] >= static_cast<float>(b.min[1])) && (v[1] <= static_cast<float>(b.max[1])) &&
           (v[2] >= static_cast<float>(b.min[2])) && (v[2] <= static_cast<float>(b.max[2]));
}

struct Block
{
     Block(int gid_, const Bounds& bounds_, const Bounds& domain_, const Oscillators& oscillators_, float velocity_scale_):
                gid(gid_),
                velocity_scale(velocity_scale_),
                bounds(bounds_),
                domain(domain_),
                grid(Vertex(&bounds.max[0]) - Vertex(&bounds.min[0]) + Vertex::one()),
                oscillators(oscillators_)
    {
    }

    void advance(float t)
    {
        grid::for_each(grid.shape(), [&](const Vertex& v)
        {
            auto& gv = grid(v);
                  gv = 0;
            auto v_global = v + Vertex(&bounds.min[0]);

            for (auto& o : oscillators)
                gv += o.evaluate(v_global, t);
        });

        for (auto& particle : particles)
        {
            particle.velocity = { 0, 0, 0 };
            for (auto& o : oscillators)
            {
                particle.velocity += o.evaluateGradient(particle.position, t);
            }
            // scale the gradient to get "units" right for velocity
            particle.velocity *= velocity_scale;
        }
    }

    void move_particles(float dt, const Proxy& cp)
    {
        auto link = static_cast<Link*>(cp.link());

        auto particle = particles.begin();
        while (particle != particles.end())
        {
            particle->position += particle->velocity * dt;
            // warp position if needed
            for (int i = 0; i < 3; ++i)
            {
                if (particle->position[i] > static_cast<float>(domain.max[i]))
                {
                    particle->position[i] += static_cast<float>(domain.min[i] - domain.max[i]);
                }
                else if (particle->position[i] < static_cast<float>(domain.min[i]))
                {
                    particle->position[i] += static_cast<float>(domain.max[i] - domain.min[i]);
                }
            }

            if (!IsVertexInsideBounds(particle->position, bounds))
            {
                bool enqueued = false;
                for (int i = 0; i < link->size(); ++i)
                {
                    if (IsVertexInsideBounds(particle->position, link->bounds(i)))
                    {
                        cp.enqueue(link->target(i), *particle);
                        enqueued = true;
                        break;
                    }
                }
                if (!enqueued)
                {
                    std::cerr << "Error: could not find appropriate neighbor for particle: id: " << particle->id
                        << ", position: (" << particle->position[0] << ", " << particle->position[1] << ", " << particle->position[2]
                        << "), velocity: (" << particle->velocity[0] << ", " << particle->velocity[1] << ", " << particle->velocity[2]
                        << std::endl;
                }
                particle = particles.erase(particle);
            }
            else
            {
                ++particle;
            }
        }
    }

    void handle_incoming_particles(const Proxy& cp)
    {
        auto link = static_cast<Link*>(cp.link());
        for (int i = 0; i < link->size(); ++i)
        {
            auto nbr = link->target(i).gid;
            while(cp.incoming(nbr))
            {
                Particle particle;
                cp.dequeue(nbr, particle);
                particles.push_back(particle);
            }
        }
    }

    void    analyze_block()
      {
#ifdef ENABLE_SENSEI
      bridge::set_data(gid, grid.data());
      bridge::set_particles(gid, particles);
#else
      analyze(gid, grid.data());
#endif
      }

    static void* create()                   { return new Block; }
    static void  destroy(void* b)           { delete static_cast<Block*>(b); }

    int                             gid;
    float                           velocity_scale;
    Bounds                          bounds;
    Bounds                          domain;
    Grid                            grid;
    Particles                       particles;
    Oscillators                     oscillators;

    private:
        Block() // for create; to let Master manage the blocks
          : gid(-1), velocity_scale(1.0f)
        {}
};


int main(int argc, char** argv)
{
    diy::mpi::environment     env(argc, argv);
    diy::mpi::communicator    world;

    using namespace opts;

    Vertex                      shape     = { 64, 64, 64 };
    int                         nblocks   = world.size();
    float                       t_end     = 10;
    float                       dt        = .01;
    float                       velocity_scale = 50.0f;
    size_t                      window    = 10;
    size_t                      k_max     = 3;
    int                         threads   = 1;
    int                         ghostLevels = 0;
    int                         numberOfParticles = 64;
    int                         seed = -1;
    std::string                 config_file;
    std::string                 out_prefix = "";
    Options ops(argc, argv);
    ops
        >> Option('b', "blocks", nblocks,   "number of blocks to use. must greater or equal to number of MPI ranks.")
        >> Option('s', "shape",  shape,     "domain shape")
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
        >> Option('g', "ghost levels", ghostLevels, "number of ghost levels")
        >> Option('p', "particles", numberOfParticles, "number of random particles to generate")
        >> Option('v', "velocity scale", velocity_scale, "scale factor to convert function gradient to velocity")
        >> Option(     "seed", seed, "specify a random seed")
    ;
    bool sync = ops >> Present("sync", "synchronize after each time step");
    bool log = ops >> Present("log", "generate full time and memory usage log");
    bool shortlog = ops >> Present("shortlog", "generate a summary time and memory usage log");

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

    int particlesPerBlock = (numberOfParticles + nblocks - 1) / nblocks;
    if (seed == -1)
    {
        if (world.rank() == 0)
            seed = static_cast<int>(std::time(nullptr));

        diy::mpi::broadcast(world, seed, 0);
    }

    std::default_random_engine rng(static_cast<RandomSeedType>(seed));
    for (int i = 0; i < world.rank(); ++i) rng(); // different seed for each rank

    if (log || shortlog)
        timer::Enable(shortlog);

    timer::Initialize();
    timer::MarkStartEvent("oscillators::initialize");

    Oscillators oscillators;
    if (world.rank() == 0)
    {
        oscillators = read_oscillators(infn);
        diy::MemoryBuffer bb;
        diy::save(bb, oscillators);
        diy::mpi::broadcast(world, bb.buffer, 0);
    } else
    {
        diy::MemoryBuffer bb;
        diy::mpi::broadcast(world, bb.buffer, 0);
        diy::load(bb, oscillators);
    }
    //for (auto& o : oscillators)
    //    fmt::print("center = {}, radius = {}, omega0 = {}, zeta = {}\n", o.center, o.radius, o.omega0, o.zeta);

    diy::Master               master(world, threads, -1,
                                     &Block::create,
                                     &Block::destroy);
    diy::ContiguousAssigner   assigner(world.size(), nblocks);

    Bounds domain;
    domain.min[0] = domain.min[1] = domain.min[2] = 0;
    for (unsigned i = 0; i < 3; ++i)
      domain.max[i] = shape[i] - 1;

    // record various parameters to initialize analysis
    std::vector<int> gids,
                     from_x, from_y, from_z,
                     to_x,   to_y,   to_z;


    diy::RegularDecomposer<Bounds>::BoolVector share_face;
    diy::RegularDecomposer<Bounds>::BoolVector wrap(true);
    diy::RegularDecomposer<Bounds>::CoordinateVector ghosts = {ghostLevels, ghostLevels, ghostLevels};

    // decompose the domain
    diy::decompose(3, world.rank(), domain, assigner,
                   [&](int gid, const Bounds&, const Bounds& bounds, const Bounds& domain, const Link& link)
                   {
                      auto b = new Block(gid, bounds, domain, oscillators, velocity_scale);

                      // generate particles
                      int start = particlesPerBlock * gid;
                      int count = particlesPerBlock;
                      if (start + count > numberOfParticles)
                      {
                        count = std::max(0, numberOfParticles - start);
                      }
                      float bds[6];
                      for (int i = 0; i < 3; ++i)
                      {
                        bds[2*i] = static_cast<float>(bounds.min[i]);
                        bds[2*i + 1] = static_cast<float>(bounds.max[i]);
                      }
                      b->particles = GenerateRandomParticles(rng, bds, start, count);

                      master.add(gid, b, new Link(link));

                      // std::cout << world.rank() << " Block " << gid << ": " << Vertex(bounds.min) << " - " << Vertex(bounds.max) << std::endl;

                      gids.push_back(gid);
                      from_x.push_back(bounds.min[0]);
                      from_y.push_back(bounds.min[1]);
                      from_z.push_back(bounds.min[2]);
                      to_x.push_back(bounds.max[0]);
                      to_y.push_back(bounds.max[1]);
                      to_z.push_back(bounds.max[2]);
                   },
                   share_face, wrap, ghosts);

    timer::MarkEndEvent("oscillators::initialize");

    timer::MarkStartEvent("oscillators::analysis::initialize");
#ifdef ENABLE_SENSEI
    bridge::initialize(world, window, nblocks, gids.size(),
                       domain.max[0] + 1, domain.max[1] + 1, domain.max[2] + 1,
                       &gids[0],
                       &from_x[0], &from_y[0], &from_z[0],
                       &to_x[0],   &to_y[0],   &to_z[0],
                       &shape[0], ghostLevels,
                       config_file);
#else
    init_analysis(world, window, gids.size(),
                  domain.max[0] + 1, domain.max[1] + 1, domain.max[2] + 1,
                  &gids[0],
                  &from_x[0], &from_y[0], &from_z[0],
                  &to_x[0],   &to_y[0],   &to_z[0]);
#endif
    timer::MarkEndEvent("oscillators::analysis::initialize");

    if (world.rank() == 0)
        fmt::print("Started\n");

    using Time = std::chrono::high_resolution_clock;
    using ms   = std::chrono::milliseconds;
    auto start = Time::now();

    int t_count = 0;
    float t = 0.;
    while (t < t_end)
    {
        timer::MarkStartTimeStep(t_count, t);
        timer::MarkStartEvent("oscillators::advance");
        master.foreach([=](Block* b, const Proxy&)
                              {
                                b->advance(t);
                              });
        timer::MarkEndEvent("oscillators::advance");

        timer::MarkStartEvent("oscillators::move_particles");
        master.foreach([=](Block* b, const Proxy& p)
                              {
                                b->move_particles(dt, p);
                              });
        timer::MarkEndEvent("oscillators::move_particles");

        timer::MarkStartEvent("oscillators::master.exchange");
        master.exchange();
        timer::MarkEndEvent("oscillators::master.exchange");

        timer::MarkStartEvent("oscillators::handle_incoming_particles");
        master.foreach([=](Block* b, const Proxy& p)
                              {
                                b->handle_incoming_particles(p);
                              });
        timer::MarkEndEvent("oscillators::handle_incoming_particles");

        timer::MarkStartEvent("oscillators::analysis");
        master.foreach([=](Block* b, const Proxy&)
                              {
                                b->analyze_block();
                              });
#ifdef ENABLE_SENSEI
        bridge::analyze(t);
#else
        analysis_round();   // let analysis do any kind of global operations it would like
#endif
        timer::MarkEndEvent("oscillators::analysis");

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
            if (world.rank() == 0)
            {
                auto out_duration = std::chrono::duration_cast<ms>(Time::now() - out_start);
                fmt::print("Output time for {}: {}.{} s\n", outfn, out_duration.count() / 1000, out_duration.count() % 1000);
            }
        }

        if (sync)
            world.barrier();

        timer::MarkEndTimeStep();

        t += dt;
        t_count++;
    }


    timer::MarkStartEvent("oscillators::finalize");
#ifdef ENABLE_SENSEI
    bridge::finalize(k_max, nblocks);
#else
    analysis_final(k_max, nblocks);
#endif
    timer::MarkEndEvent("oscillators::finalize");

    timer::Finalize();

    world.barrier();
    if (world.rank() == 0)
      {
      auto duration = std::chrono::duration_cast<ms>(Time::now() - start);
      fmt::print("Total run time: {}.{} s\n", duration.count() / 1000, duration.count() % 1000);
      }
}
