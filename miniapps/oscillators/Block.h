#ifndef BlockData_h
#define BlockData_h

#include <sdiy/master.hpp>
#include <sdiy/decomposition.hpp>
#include <sdiy/io/bov.hpp>
#include <sdiy/grid.hpp>
#include <sdiy/vertices.hpp>

#include "Oscillator.h"
#include "Particles.h"

#include <vector>
#include <ostream>

struct Block
{
    using Vertex = sdiy::Grid<float,3>::Vertex;

     Block(int gid_, const sdiy::DiscreteBounds& bounds_, const sdiy::DiscreteBounds& domain_,
         const sdiy::Point<float,3> &origin_, const sdiy::Point<float,3> &spacing_,
         int nghost_, const std::vector<Oscillator>& oscillators_, float velocity_scale_) :
                gid(gid_), velocity_scale(velocity_scale_), bounds(bounds_),
                domain(domain_), origin(origin_), spacing(spacing_), nghost(nghost_),
                grid(Vertex(&bounds.max[0]) - Vertex(&bounds.min[0]) + Vertex::one()),
                oscillators(oscillators_)
    {}

    // update scalar and vector fields
    void update_fields(float t);

    // update pareticle positions
    void move_particles(float dt, const sdiy::Master::ProxyWithLink& cp);

    // handle particle migration
    void handle_incoming_particles(const sdiy::Master::ProxyWithLink& cp);

    // sdiy memory management hooks
    static void *create(){
      static std::vector<Oscillator> empty;
      return new Block(empty);
    }
    static void destroy(void* b){ delete static_cast<Block*>(b); }

    int                             gid;    // block id
    float                           velocity_scale;
    sdiy::DiscreteBounds            bounds; // cell centered index space dimensions of this block
    sdiy::DiscreteBounds            domain; // cell centered index space dimensions of the computational domain
    sdiy::Point<float,3>            origin;  // lower left most corner in the computational domain
    sdiy::Point<float,3>            spacing; // mesh spacing
    int                             nghost; // number of ghost zones
    sdiy::Grid<float,3>             grid;   // container for the gridded data arrays
    std::vector<Particle>           particles;
    const std::vector<Oscillator>&  oscillators;

 private:
    // for create; to let Master manage the blocks
    Block(const std::vector<Oscillator>& oscillators_) : gid(-1), velocity_scale(1.0f), nghost(0),
      oscillators(oscillators_)
    {
        origin[0] = origin[1] = origin[2] = 0.0f;
        spacing[0] = spacing[1] = spacing[2] = 1.0f;
    }
};

// send to stream in human readbale format
std::ostream &operator<<(std::ostream &os, const Block &b);

#endif
