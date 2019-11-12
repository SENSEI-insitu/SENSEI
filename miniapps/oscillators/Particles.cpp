#include "Particles.h"

// --------------------------------------------------------------------------
sdiy::DiscreteBounds remove_ghosts(const sdiy::DiscreteBounds &domain,
    const sdiy::DiscreteBounds &gbounds, int nghosts)
{
    sdiy::DiscreteBounds bds = gbounds;

    // remove ghost zones
    if (nghosts > 0)
    {
        for (int i = 0; i < 3; ++i)
        {
            if (domain.min[i] != bds.min[i])
                bds.min[i] += nghosts;

            if (domain.max[i] != bds.max[i])
                bds.max[i] -= nghosts;
        }
    }

    return bds;
}

// --------------------------------------------------------------------------
// --------------------------------------------------------------------------
std::ostream &operator<<(std::ostream &os, const Particle &particle)
{
    os << "(id " << particle.id
        << ", position: (" <<particle.position[0] << ", " << particle.position[1] << ", " << particle.position[2]
        << "), velocity: (" << particle.velocity[0] << ", " << particle.velocity[1] << ", " << particle.velocity[2]
        << "))";
    return os;
}
