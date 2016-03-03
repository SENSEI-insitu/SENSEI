#include <vector>
#include <chrono>

#include <format.h>
#include <opts/opts.h>

#include <diy/master.hpp>
#include <diy/decomposition.hpp>

#include <grid/grid.h>
#include <grid/vertices.h>

#include "oscillator.h"

#ifdef ENABLE_SENSEI
#include "bridge.h"
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



struct Block
{
        Block(int gid_, const Bounds& bounds_, const Vertex& domain_shape_, const Oscillators& oscillators_):
                gid(gid_),
                bounds(bounds_),
                domain_shape(domain_shape_),
                from(bounds.min),
                grid(Vertex(bounds.max) - Vertex(bounds.min) + Vertex::one()),
                oscillators(oscillators_)
    {
    }

    void    advance(float t)
    {
        grid::for_each(grid.shape(), [&](const Vertex& v)
        {
            auto& gv = grid(v);
                  gv = 0;
            auto v_global = v + from;

            for (auto& o : oscillators)
                gv += o.evaluate(v_global, t);
        });
    }

    void    analyze_block()
      {
#ifdef ENABLE_SENSEI
      bridge::set_data(gid, grid.data());
#else
      analyze(gid, grid.data());
#endif
      }

    static void* create()                   { return new Block; }
    static void  destroy(void* b)           { delete static_cast<Block*>(b); }

    int                             gid;
    Bounds                          bounds;
    Vertex                          domain_shape;
    Vertex                          from;
    Grid                            grid;
    Oscillators                     oscillators;

    private:
        Block() {}      // for create; to let Master manage the blocks
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
    size_t                      window    = 10;
    size_t                      k_max     = 3;
    int                         threads   = 1;
    std::string                 config_file;
    Options ops(argc, argv);
    ops
        >> Option('b', "blocks", nblocks,   "number of blocks to use")
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
    ;
    bool sync = ops >> Present("sync", "synchronize after each time step");

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


    // decompose the domain
    diy::decompose(3, world.rank(), domain, assigner,
                   [&](int gid, const Bounds& core, const Bounds& bounds, const Bounds& domain, const Link& link)
                   {
                      auto b = new Block(gid, bounds, { domain.max[0] + 1, domain.max[1] + 1, domain.max[2] + 1 }, oscillators);
                      master.add(gid, b, new Link(link));

                      std::cout << world.rank() << " Block " << gid << ": " << Vertex(bounds.min) << " - " << Vertex(bounds.max) << std::endl;

                      gids.push_back(gid);
                      from_x.push_back(bounds.min[0]);
                      from_y.push_back(bounds.min[1]);
                      from_z.push_back(bounds.min[2]);
                      to_x.push_back(bounds.max[0]);
                      to_y.push_back(bounds.max[1]);
                      to_z.push_back(bounds.max[2]);
                   });

    timer::MarkEndEvent("oscillators::initialize");

    timer::MarkStartEvent("oscillators::analysis::initialize");
#ifdef ENABLE_SENSEI
    bridge::initialize(world, window, nblocks, gids.size(),
                       domain.max[0] + 1, domain.max[1] + 1, domain.max[2] + 1,
                       &gids[0],
                       &from_x[0], &from_y[0], &from_z[0],
                       &to_x[0],   &to_y[0],   &to_z[0],
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
        master.foreach<Block>([=](Block* b, const Proxy&, void*)
                              {
                                b->advance(t);
                              });
        timer::MarkEndEvent("oscillators::advance");

        timer::MarkStartEvent("oscillators::analysis");
        master.foreach<Block>([=](Block* b, const Proxy&, void*)
                              {
                                b->analyze_block();
                              });
#ifdef ENABLE_SENSEI
        bridge::analyze(t);
#else
        analysis_round();   // let analysis do any kind of global operations it would like
#endif
        timer::MarkEndEvent("oscillators::analysis");
        if (sync)
            world.barrier();
        timer::MarkEndTimeStep();

        t += dt;
        t_count++;
    }
    world.barrier();

    timer::MarkStartEvent("oscillators::finalize");
#ifdef ENABLE_SENSEI
    bridge::finalize(k_max, nblocks);
#else
    analysis_final(k_max, nblocks);
#endif
    timer::MarkEndEvent("oscillators::finalize");

    auto duration = std::chrono::duration_cast<ms>(Time::now() - start);
    if (world.rank() == 0)
      {
      fmt::print("Total run time: {}.{} s\n", duration.count() / 1000, duration.count() % 1000);
      timer::PrintLog(std::cout);
      }
}
