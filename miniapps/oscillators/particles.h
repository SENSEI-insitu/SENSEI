#ifndef particles_h
#define particles_h

#include <grid/point.h>

#include <random>
#include <vector>

struct Particle
{
    using Vertex = grid::Point<float, 3>;

    int id;
    Vertex position;
    Vertex velocity;
};

using Particles = std::vector<Particle>;

inline Particles GenerateRandomParticles(std::default_random_engine& rng, float bounds[6], int startId, int count)
{
  std::uniform_real_distribution<float> rgx(bounds[0], bounds[1]);
  std::uniform_real_distribution<float> rgy(bounds[2], bounds[3]);
  std::uniform_real_distribution<float> rgz(bounds[4], bounds[5]);

  Particles particles;
  for (int i = 0; i < count; ++i)
  {
    Particle p;
    p.id = startId + i;
    p.position = { rgx(rng), rgy(rng), rgz(rng) };
    p.velocity = { 0, 0, 0 };
    particles.push_back(p);
  }

  return particles;
}

#endif // particles_h
