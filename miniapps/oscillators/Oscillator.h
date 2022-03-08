#ifndef Oscillator_h
#define Oscillator_h

#include <memory>
#include <string>
#include <cmath>
#include <sdiy/point.hpp>
#include <sdiy/mpi.hpp>

namespace sensei { class DataAdaptor; }

/// a perdiodic, decaying, or damped oscillator
struct Oscillator
{
    using Vertex = sdiy::Point<float,3>;

    static constexpr float pi = 3.14159265358979323846;

#if defined(OSCILLATOR_CUDA)
    __host__ __device__
#endif
    float evaluate(float vx, float vy, float vz, float t) const
    {
        t *= 2*pi;

        float dist_x = center_x - vx;
        float dist_y = center_y - vy;
        float dist_z = center_z - vz;
        float dist2 = sqrt( dist_x*dist_x + dist_y*dist_y + dist_z*dist_z );

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
        Vertex center({center_x, center_y, center_z});
        return evaluate(x[0], x[1], x[1], t) * ((center - x)/(radius * radius));
    }

    float   center_x;
    float   center_y;
    float   center_z;

    float   radius;

    float   omega0;
    float   zeta;

    enum Type { damped, decaying, periodic };
    Type type;
};

/// holds an array of oscillators and its size
class OscillatorArray
{
public:
    /// initialize the array from a file
    void Initialize(const sdiy::mpi::communicator &comm,
      const std::string &fn);

    /// initialize the array from a data adaptor
    void Initialize(const sdiy::mpi::communicator &comm,
      sensei::DataAdaptor *da);

    /// releases the array
    void Clear();

    /// allocate n Oscillators
    void Allocate(unsigned long n);

    /// access the array
    Oscillator *Data() { return mData.get(); }
    const Oscillator *Data() const { return mData.get(); }

    /// access the ith element
    Oscillator &operator[](unsigned long i) { return mData.get()[i]; }
    const Oscillator &operator[](unsigned long i) const { return mData.get()[i]; }

    /// the size of the array
    unsigned long Size() const { return mSize; }

    /// Print the oscillators
    void Print(std::ostream &os) const;

private:
    unsigned long mSize;
    std::shared_ptr<Oscillator> mData;
};


#endif
