#! /usr/bin/env python

import sys
import argparse
import random
from math import sqrt
pi = 3.1415

parser = argparse.ArgumentParser(description='generate an oscillator input deck.')

parser.add_argument('-n', '--num', type=int, default=1,
    help='number of oscillators to generate')

parser.add_argument('-b', '--bounds', nargs=6, type=float, default=[0.,1.,0.,1.,0.,1.0],
    help='set the bounds of the simulation domain')

parser.add_argument('-w', '--omega', nargs=2, type=float, default=[pi/4., 4.*pi],
    help='oscillator angular frequency min/max')

parser.add_argument('-r', '--radius', nargs=2, type=float, default=[0.01, 0.45],
    help='Gaussian splat radii min/max')

parser.add_argument('-z', '--zeta', nargs=2, type=float, default=[0.0, 1.0],
    help='oscillator zeta min/max for damped type')

parser.add_argument('-f', '--frac-damped', type=float, default=0.75,
    help='fraction of damped type')

args = parser.parse_args()
print("# " + str(args))


nosc = args.num

r0 = args.radius[0]
r1 = args.radius[1]

w0 = args.omega[0]
w1 = args.omega[1]

n0 = args.zeta[0]
n1 = args.zeta[1]

dfrac = args.frac_damped

x0 = args.bounds[0]
x1 = args.bounds[1]

y0 = args.bounds[2]
y1 = args.bounds[3]

z0 = args.bounds[4]
z1 = args.bounds[5]

def rng(x0, x1):
    return x0 + (x1 - x0)*random.random()

print('#type    center    r    omega0    zeta\n')


max_dist = 0.
min_dist = x1-x0

pts = []

size = [x1 - x0, y1 - y0, z1 - z0]
gridCenter = [x0 + size[0]/2, y0 + size[1]/2, z0 + size[2]/2]
i = 0
while i < nosc:
    gaussian = min(1.0, max(0.05, random.gauss(0.6, 0.2)))
    # distribution of particles follows guassian from center
    center = [
        gridCenter[0] + random.uniform(-size[0]/2, size[0]/2) * gaussian,
        gridCenter[1] + random.uniform(-size[1]/2, size[1]/2) * gaussian,
        gridCenter[2] + random.uniform(-size[2]/2, size[2]/2) * gaussian ]
    x = center[0]
    y = center[1]
    z = center[2]
    w = rng(w0, w1)
    # larger particles near the center for added structure to random particles
    r = rng(r0, r1 * (1.0 - gaussian))
    n = rng(n0, n1)
    t = 'damped' if random.random() < dfrac else 'periodic'
    print('%s    %f %f %f    %f    %f    %f\n'%(t,x,y,z,r,w,n))
    pts += [x,y,z]
    i += 1

i = 0
while i < nosc:
    xa = pts[3*i]
    ya = pts[3*i + 1]
    za = pts[3*i + 2]
    j = 0
    while j < nosc:
        if i != j:
            xb = pts[3*j]
            yb = pts[3*j + 1]
            zb = pts[3*j + 2]
            dx = xb - xa
            dy = yb - ya
            dz = zb - za
            max_dist = max(max_dist, sqrt(dx**2+dy**2+dz**2))
            min_dist = min(min_dist, sqrt(dx**2+dy**2+dz**2))
        j += 1
    i += 1

print('# max_dist = %f\n'%(max_dist))
print('# min_dist = %f\n'%(min_dist))

