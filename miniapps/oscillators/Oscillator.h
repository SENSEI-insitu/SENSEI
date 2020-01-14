#ifndef Oscillator_h
#define Oscillator_h

#include <string>
#include <cmath>
#include <sdiy/point.hpp>
#include <sdiy/mpi.hpp>

struct Oscillator
{
    using Vertex = sdiy::Point<float,3>;

    static constexpr float pi = 3.14159265358979323846;

    float evaluate(const Vertex &v, float t) const
    {
        t *= 2*pi;

        float dist2 = (center - v).norm();
        float dist_damp = exp(-dist2/(2*radius*radius));

        if (type == damped)
        {
            float phi   = acos(zeta);
            float val   = 1. - exp(-zeta*omega0*t) * (sin(sqrt(1-zeta*zeta)*omega0*t + phi) / sin(phi));
            return val * dist_damp;
        }
        else if (type == decaying)
        {
            t += 1. / omega0;
            float val = sin(t / omega0) / (omega0 * t);
            return val * dist_damp;
        }
        else if (type == periodic)
        {
            t += 1. / omega0;
            float val = sin(t / omega0);
            return val * dist_damp;
        }

        return 0.0f; // impossible
    }

    Vertex evaluateGradient(const Vertex& x, float t) const
    {
        // let f(x, t) = this->evaluate(x,t) = o(t) * g(x)
        // where o = oscillator, g = gaussian
        // therefore, df/dx = o * dg/dx
        // given g = e^u(t), so dg/dx = e^u * du/dx
        // therefore, df/dx = o * e^u * du/dx = f * du/dx
        return evaluate(x, t) * ((center - x)/(radius * radius));
    }

    Vertex  center;
    float   radius;

    float   omega0;
    float   zeta;

    enum Type { damped, decaying, periodic };
    Type type;

    bool operator==(const Oscillator& other) const
    {
      return this->center == other.center &&
              this->radius == other.radius &&
              this->omega0 == other.omega0 &&
              this->zeta == other.zeta &&
              this->type == other.type;
    }
};

std::vector<Oscillator> read_oscillators(std::string fn);

namespace sensei
{
  class DataAdaptor;
};

/// Generate a new `sensei::DataAdaptor` that provides a "mesh" that represents
/// the oscillators provided.
sensei::DataAdaptor* new_adaptor(sdiy::mpi::communicator& world, const std::vector<Oscillator>& oscillators);

/// Generate a vector of Oscillators given the `sensei::DataAdaptor`. This
/// expects an "oscillators" mesh with appropriate arrays.
std::vector<Oscillator> read_oscillators(sensei::DataAdaptor* data);

#endif
