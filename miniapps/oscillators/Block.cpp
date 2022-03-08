#include "Block.h"
#include "BlockInternals.h"

// --------------------------------------------------------------------------
void Block::update_fields(float t, const OscillatorArray &oscillators)
{
    // update the scalar oscillator field
    const Vertex &shape = grid.shape();
    int ni = shape[0];
    int nj = shape[1];
    int nk = shape[2];
    int i0 = bounds.min[0];
    int j0 = bounds.min[1];
    int k0 = bounds.min[2];
    float dx = spacing[0];
    float dy = spacing[1];
    float dz = spacing[2];
    float x0 = origin[0] + dx;
    float y0 = origin[1] + dy;
    float z0 = origin[2] + dz;
    float *pdata = grid.data();

    // TODO -- run time GPU/CPU load balancing.
#if defined(OSCILLATOR_CUDA)
    int deviceId = 0;
#else
    int deviceId = -1;
#endif
    BlockInternals::UpdateFields(deviceId, t, oscillators.Data(),
        oscillators.Size(), ni,nj,nk, i0,j0,k0, x0,y0,z0, dx,dy,dz,
        pdata);
}

// --------------------------------------------------------------------------
void Block::update_particles(float t, const OscillatorArray &oscillators)
{
    // update the velocity field on the particle mesh
    const Oscillator *pOsc = oscillators.Data();
    for (auto& particle : particles)
    {
        particle.velocity = { 0, 0, 0 };
        for (unsigned long q = 0; q < oscillators.Size(); ++q)
        {
            particle.velocity += pOsc[q].evaluateGradient(particle.position, t);
        }
        // scale the gradient to get "units" right for velocity
        particle.velocity *= velocity_scale;
    }
}

// --------------------------------------------------------------------------
void Block::move_particles(float dt, const sdiy::Master::ProxyWithLink& cp)
{
    auto link = static_cast<sdiy::RegularGridLink*>(cp.link());

    auto particle = particles.begin();
    while (particle != particles.end())
    {
        // update particle position
        particle->position += particle->velocity * dt;

        // warp position if needed
        // applies periodic bci
        sdiy::Bounds<float> wsdom = world_space_bounds(domain, origin, spacing);
        for (int i = 0; i < 3; ++i)
        {
            if ((particle->position[i] > wsdom.max[i]) ||
              (particle->position[i] < wsdom.min[i]))
            {
                float dm = wsdom.min[i];
                float dx = wsdom.max[i] - dm;
                if (fabs(dx) < 1.0e-6f)
                {
                  particle->position[i] = dm;
                }
                else
                {
                  float dp = particle->position[i] - dm;
                  float dpdx = dp / dx;
                  particle->position[i] = (dpdx - floor(dpdx))*dx + dm;
                }
            }
        }

        // check if the particle has left this block
        // block bounds have ghost zones
        if (!contains(domain, bounds, origin, spacing, nghost, particle->position))
        {
            bool enqueued = false;

            // search neighbor blocks for one that now conatins this particle
            for (int i = 0; i < link->size(); ++i)
            {
                // link bounds do not have ghost zones
                if (contains(link->bounds(i), origin, spacing, particle->position))
                {
                    /*std::cerr << "moving " << *particle << " from " << gid
                      << " to " << link->target(i).gid << std::endl;*/

                    cp.enqueue(link->target(i), *particle);

                    enqueued = true;
                    break;
                }
            }

            if (!enqueued)
            {
                std::cerr << "Error: could not find appropriate neighbor for particle: "
                   << *particle << std::endl;

                abort();
            }

            particle = particles.erase(particle);
        }
        else
        {
            ++particle;
        }
    }
}

// --------------------------------------------------------------------------
void Block::handle_incoming_particles(const sdiy::Master::ProxyWithLink& cp)
{
    auto link = static_cast<sdiy::RegularGridLink*>(cp.link());
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

std::ostream &operator<<(std::ostream &os, const Block &b)
{
    os << b.gid << ": " << b.bounds.min << " - " << b.bounds.max << std::endl;

    auto it = b.particles.begin();
    auto end = b.particles.end();
    for (; it != end; ++it)
        os << "    " << *it << std::endl;

    return os;
}

