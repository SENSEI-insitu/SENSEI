#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

#include <grid/point.h>

struct Oscillator
{
    using Vertex = grid::Point<int,3>;

    static constexpr float pi = 3.14159265358979323846;

    inline
    float   evaluate(Vertex v, float t) const
    {
        t *= 2*pi;

        float dist2 = (center - v).norm();
        float dist_damp = exp(-dist2/(2*radius*radius));

        if (type == damped)
        {
            float phi   = acos(zeta);
            float val   = 1. - exp(-zeta*omega0*t) * (sin(sqrt(1-zeta*zeta)*omega0*t + phi) / sin(phi));
            return val * dist_damp;
        } else if (type == decaying)
        {
            t += 1. / omega0;
            float val = sin(t / omega0) / (omega0 * t);
            return val * dist_damp;
        } else if (type == periodic)
        {
            t += 1. / omega0;
            float val = sin(t / omega0);
            return val * dist_damp;
        } else
            return 0;       // impossible
    }

    Vertex  center;
    float   radius;

    float   omega0;
    float   zeta;

    enum { damped, decaying, periodic } type;
};
using Oscillators = std::vector<Oscillator>;

// trim() from http://stackoverflow.com/a/217605/44738
static inline std::string &ltrim(std::string &s) { s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace)))); return s; }
static inline std::string &rtrim(std::string &s) { s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end()); return s; }
static inline std::string &trim(std::string &s)  { return ltrim(rtrim(s)); }

Oscillators read_oscillators(std::string fn)
{
    Oscillators res;

    std::ifstream in(fn);
    if (!in)
        throw std::runtime_error("Unable to open " + fn);
    std::string line;
    while(std::getline(in, line))
    {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);

        std::string stype;
        iss >> stype;

        auto type = Oscillator::periodic;
        if (stype == "damped")
            type = Oscillator::damped;
        else if (stype == "decaying")
            type = Oscillator::decaying;

        int x,y,z;
        iss >> x >> y >> z;

        float r, omega0, zeta;
        iss >> r >> omega0;

        if (type == Oscillator::damped)
            iss >> zeta;
        res.emplace_back(Oscillator { {x,y,z}, r, omega0, zeta, type });
    }
    return res;
}
