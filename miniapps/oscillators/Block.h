#ifndef BlockData_h
#define BlockData_h

#include <sdiy/master.hpp>
#include <sdiy/decomposition.hpp>
#include <sdiy/io/bov.hpp>
#include <sdiy/vertices.hpp>
#include <sdiy/point.hpp>

#include "Oscillator.h"
#include "Particles.h"
#include "Grid.h"

#include <vector>
#include <ostream>

struct Block
{
    using Vertex = oscillator::Grid<float,3>::Vertex;

     Block(int gid_, const sdiy::DiscreteBounds& bounds_, const sdiy::DiscreteBounds& domain_,
         const sdiy::Point<float,3> &origin_, const sdiy::Point<float,3> &spacing_,
         int nghost_, const std::shared_ptr<const Oscillator> &oscillators_, unsigned long nOscillators_,
         float velocity_scale_) :
                gid(gid_), velocity_scale(velocity_scale_), bounds(bounds_),
                domain(domain_), origin(origin_), spacing(spacing_), nghost(nghost_),
                grid(Vertex(&bounds.max[0]) - Vertex(&bounds.min[0]) + Vertex::one()),
                oscillators(oscillators_), nOscillators(nOscillators_)
    {}

    // update mesh based scalar and vector fields
    void update_fields(float t);

    // update particle based scalar and vector fields
    void update_particles(float t);

    // update pareticle positions
    void move_particles(float dt, const sdiy::Master::ProxyWithLink& cp);

    // handle particle migration
    void handle_incoming_particles(const sdiy::Master::ProxyWithLink& cp);

    // sdiy memory management hooks
    static void *create(){ return new Block; }
    static void destroy(void* b){ delete static_cast<Block*>(b); }

    int                               gid;    // block id
    float                             velocity_scale;
    sdiy::DiscreteBounds              bounds; // cell centered index space dimensions of this block
    sdiy::DiscreteBounds              domain; // cell centered index space dimensions of the computational domain
    sdiy::Point<float,3>              origin;  // lower left most corner in the computational domain
    sdiy::Point<float,3>              spacing; // mesh spacing
    int                               nghost; // number of ghost zones
    oscillator::Grid<float,3>         grid;   // container for the gridded data arrays
    std::vector<Particle>             particles;

    std::shared_ptr<const Oscillator> oscillators; // a pointer to the list of oscillators
    unsigned long                     nOscillators; // number of oscillators

 private:
    // for create; to let Master manage the blocks
    Block() : gid(-1), velocity_scale(1.0f), nghost(0)
    {
        origin[0] = origin[1] = origin[2] = 0.0f;
        spacing[0] = spacing[1] = spacing[2] = 1.0f;
    }
};

// send to stream in human readbale format
std::ostream &operator<<(std::ostream &os, const Block &b);

#endif
