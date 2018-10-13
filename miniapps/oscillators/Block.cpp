#include "Block.h"

// --------------------------------------------------------------------------
void Block::update_fields(float t)
{
    // update the scalar oscilltor field
    diy::for_each(grid.shape(), [&](const Vertex& v)
    {
        auto& gv = grid(v);
              gv = 0;
        auto v_global = v + Vertex(&bounds.min[0]);

        for (auto& o : oscillators)
            gv += o.evaluate(v_global, t);
    });

    // update the velocity field on the particle mesh
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

// --------------------------------------------------------------------------
void Block::move_particles(float dt, const diy::Master::ProxyWithLink& cp)
{
    auto link = static_cast<diy::RegularGridLink*>(cp.link());

    auto particle = particles.begin();
    while (particle != particles.end())
    {
        // update particle position
        particle->position += particle->velocity * dt;

        // warp position if needed
        // applies periodic bci
        diy::Bounds<float> wsdom = world_space_bounds(domain, origin, spacing);
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
void Block::handle_incoming_particles(const diy::Master::ProxyWithLink& cp)
{
    auto link = static_cast<diy::RegularGridLink*>(cp.link());
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

