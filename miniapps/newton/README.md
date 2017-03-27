# Newton
The Newton mini-app is a simple yet complete example of using Sensei Python
bindings. It solves Newton's law of gravitation using a symplectic integrator.
The included initial condition, initializes an area roughly the size of, the
solar system with a number of of planets of varying masses at randomly chosen
locations. A sun sized body is placed at the center of the domain.

## Command line options
```bash
usage: newton.py [-h] [--analysis ANALYSIS] [--analysis_opts ANALYSIS_OPTS]
                 [--n_bodies N_BODIES] [--n_its N_ITS] [--dt DT]

optional arguments:
  -h, --help            show this help message and exit
  --analysis ANALYSIS   Type of analysis to run. posthoc,catalyst,configurable
  --analysis_opts ANALYSIS_OPTS
                        A CSV list of name=value pairs specific to each
                        analysis type. cataylst: script=a catalyst Python
                        script. posthoc: mode=pvd|visit,file=file name,dir=
                        output dir freq=number of steps between I/O.
                        configurable: config=xml file
  --n_bodies N_BODIES   Number of bodies per process
  --n_its N_ITS         Number of iterations to run
  --dt DT               Time step in seconds
```
One uses **--analysis** to select an analysis, and **--analysis_opts** to pass
analysis specific options in the form of a CSV list to the selcted analysis.

There are three **analysis** options, catalyst which exercises the CatalystAdaptor,
**posthoc** which exercises the VTKPosthocIO adaptor, and **configurable** which
exercises the ConfigurableAnalysis adaptor. Each of these require some analysis
specific options:

| Analysis | Option | Description |
|----------|--------|-------------|
| catalyst | script | path to a Catalyst Python script |
| libsim   |        |             |
| configurable | config | path to a SENSEI ConfigurableAnalysis XML config |
| posthoc | file | base output file prefix, step, block and extension are appended to this |
|         | dir | output directory |
|         | mode | 0 ParaView compatible  output, 1 VisIt compatible output |
|         | freq | number of steps in between I/O |


## Catalyst
### Building
One needs to point Python to the ParaView install while CMake figures things
out. This is because ParaView packages up some common Python modules, such as
mpi4py.

```bash
# point to ParaView's Python packages
export LD_LIBRARY_PATH=/work/SENSEI/PV/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=/work/SENSEI/PV/lib/site-packages/:/work/SENSEI/PV/lib/:$PYTHONPATH

# configure the build
cmake -DENABLE_PYTHON=ON -DENABLE_CATALYST=ON -DENABLE_CATALYST_PYTHON=ON \
  -DParaView_DIR=/work/SENSEI/PV ../sensei/
```

### Running
One needs to setup Python consistently with the build, and make sure that the
Python interpreter can see include ParaView and SENSEI Python modules.

```bash
# point to ParaView and SENSEI Python modules
export LD_LIBRARY_PATH=/work/SENSEI/PV/lib/:/work/SENSEI/sensei-build/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/work/SENSEI/PV/lib/site-packages/:/work/SENSEI/PV/lib/:/work/SENSEI/sensei-build/lib:$PYTHONPATH

# run the simulation
mpiexec -np 4 python ../sensei/miniapps/newton/newton.py --analysis=catalyst \
  --analysis_options=script=../sensei/miniapps/newton/newton_catalyst.py
```

## Libsim
### Building
One needs to point SENSEI to VTK and Python built by VisIt.
```bash
# point to our VisIt install
VISIT_INSTALL=/work/SENSEI/visit-install/2.13.0/linux-x86_64/
VISIT_DEPS=/work/SENSEI/visit-deps/
VISIT_PYTHON_HOME=${VISIT_DEPS}/visit/python/2.7.11/x86_64/
VISIT_VTK_HOME=${VISIT_DEPS}/visit/vtk/6.1.0/x86_64

# use VisIt's Python. VisIt's VTK should be built in this environment
# as well to ensure that system installs of Python are not accidentally
# linked into VTK
export PYTHONHOME=${VISIT_PYTHON_HOME}
export PYTHONPATH=${VISIT_PYTHON_HOME}/lib/python2.7/site-packages/
export PKG_CONFIG_PATH=${VISIT_PYTHON_HOME}/lib/pkgconfig/
export PATH=${VISIT_PYTHON_HOME}/bin/:$PATH

# use VisIt's VTK
export LD_LIBRARY_PATH=${VISIT_VTK_HOME}/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=${VISIT_VTK_HOME}/lib/python2.7/site-packages/:$PYTHONPATH

# configure the build
cmake -DENABLE_SENSEI=ON -DENABLE_LIBSIM=ON -DENABLE_PYTHON=ON \
  -DVTK_DIR=${VISIT_VTK_HOME}/lib/cmake/vtk-6.1/ -DVISIT_DIR=${VISIT_INSTALL}
  ../sensei
```

### Running
In addition to VisIt's Python and VTK; and SENSEI's Python module, the enviornment
must be set up for VisIt. This is easiest done in a launch script that sets up the
environment and invokes the given command.
```bash
#!/bin/bash

if [[ $# < 1 ]]
then
echo "Usage libsim_launch.sh [arg1 ... argn]"
echo
echo "pass the command to launch in arg1 to argn"
echo
exit -1
fi


VISIT_DEPS=/work/SENSEI/visit-deps/
VISIT_PYTHON_HOME=${VISIT_DEPS}/visit/python/2.7.11/x86_64/
VISIT_VTK_HOME=${VISIT_DEPS}/visit/vtk/6.1.0/x86_64
SENSEI_HOME=/work/SENSEI/sensei-build-libsim


# let VisIt set the environment variables libsim needs
source <(../visit-install/bin/visit -env engine | sed 's/^/export /')

# use VisIt's Python (note VisIt's VTK should be built in this environment)
export PYTHONHOME=${VISIT_PYTHON_HOME}
export PYTHONPATH=${VISIT_PYTHON_HOME}/lib/python2.7/site-packages/
export PKG_CONFIG_PATH=${VISIT_PYTHON_HOME}/lib/pkgconfig/
export PATH=${VISIT_PYTHON_HOME}/bin/:$PATH

# use VisIt's VTK
export LD_LIBRARY_PATH=${VISIT_VTK_HOME}/lib/:$LD_LIBRARY_PATH
export PYTHONPATH=${VISIT_VTK_HOME}/lib/python2.7/site-packages/:$PYTHONPATH

# use SENSEI's Python module
export LD_LIBRARY_PATH=${SENSEI_HOME}/lib:$LD_LIBRARY_PATH
export PYTHONPATH=${SENSEI_HOME}/lib:$PYTHONPATH

# launch the app using the command line
$*
```

To launch the simulation invoke the launch script with the desired command.
```bash
./libsim_launch.sh  mpiexec -np 4 python \
  ../sensei/miniapps/newton/newton.py --analysis=libsim
```

