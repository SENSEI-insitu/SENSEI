.. _reaction_rate_demo:

Chemical reaction on a 2D substrate
===================================
This example illustrates how to select different back-ends at run time via XML,
and how to switch in between in situ mode where the analysis runs in the same
address space as the simulation and in transit mode where the analysis runs in
a separate application called an end-point potentially on a different number of
MPI ranks.

This example makes use of the oscillator mini-app configured as a proxy for
simulation of a chemical reaction on a 2D substrate. The example uses
different back-ends to make a pseudo coloring of the reaction rate with an
iso-contour of 1. The Python analysis computes the area of the substrate where
the reaction rate is greater or equal to 1 and plots it over time.

In situ demos
-------------
In this part of the demo XML files are used to switch back-end data consumer.
The back-end data consumers are running in the same process as the simulation.
This enables the use of zero-copy data transfer between the simulation and data
consumer.

.. _ascent_insitu_demo:

Ascent in situ demo
^^^^^^^^^^^^^^^^^^^
.. _ascent_insitu_image:
.. figure:: http://hpcvis.com/vis/sensei/rtd-docs/images/random_2d_64_ascent_00020.png
   :width: 80 %
   :align: center

   A pseudocolor plot rendered by Ascent of the rectaion rate field
   with an iso-contour plotted at a reaction rate of 1.0.

In the demo data from the reaction rate proxy simulation is processed using
Ascent. Ascent is selected at run time via the following SENSEI XML:

.. _ascent_insitu_xml:
.. code-block:: xml

   <sensei>
     <analysis type="ascent" actions="configs/random_2d_64_ascent.json" enabled="1" >
       <mesh name="mesh">
           <cell_arrays> data </cell_arrays>
       </mesh>
     </analysis>
   </sensei>

   XML to select the Ascent back-end and configure it using a Ascent JSON
   configuration

The analysis element selects Ascent, the actions attribute points to the
Ascent specific configuration. In this case a JSON configuration. The following
shell script runs the demo on the VM.

.. _ascent_insitu_script:
.. code-block:: bash

   #!/bin/bash

   n=4
   b=64
   dt=0.25
   bld=`echo -e '\e[1m'`
   red=`echo -e '\e[31m'`
   grn=`echo -e '\e[32m'`
   blu=`echo -e '\e[36m'`
   wht=`echo -e '\e[0m'`

   echo "+ module load sensei/3.1.0-ascent-shared"
   module load sensei/3.1.0-ascent-shared

   set -x

   export OMP_NUM_THREADS=1

   cat ./configs/random_2d_${b}_ascent.xml | sed "s/.*/$blu&$wht/"

   mpiexec -n ${n} \
       oscillator -b ${n} -t ${dt} -s ${b},${b},1 -g 1 -p 0 \
       -f ./configs/random_2d_${b}_ascent.xml \
       ./configs/random_2d_${b}.osc 2>&1 | sed "s/.*/$red&$wht/"

During the run Ascent is configured to render a pseudocolor plot of the
reaction rate field. The plot includes an iso-contour where the reaction rate
is 1.


.. _catalyst_insitu_demo:

ParaView Catalyst in situ demo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. _catalyst_insitu_image:
.. figure:: http://hpcvis.com/vis/sensei/rtd-docs/images/random_2d_64_catalyst_00020.png
   :width: 80 %
   :align: center

   A pseudocolor plot rendered by ParaView Catalyst of the rectaion rate field
   with an iso-contour plotted at a reaction rate of 1.0.

In the demo data from the reaction rate proxy simulation is processed using
ParaView Catalyst. Catalyst is selected at run time via the following SENSEI
XML:

.. _catalyst_insitu_xml:
.. code-block:: xml

   <sensei>
     <analysis type="catalyst" pipeline="pythonscript"
       filename="configs/random_2d_64_catalyst.py" enabled="1" />
   </sensei>

The analysis element selects ParaView Catalyst, the filename attribute points
to the Catalyst specific configuration. In this case a Python script that was
generated using the ParaView GUI. The following shell script runs the demo on
the VM.

.. _catalyst_insitu_script:
.. code-block:: bash

   #!/bin/bash

   n=4
   b=64
   dt=0.25
   bld=`echo -e '\e[1m'`
   red=`echo -e '\e[31m'`
   grn=`echo -e '\e[32m'`
   blu=`echo -e '\e[36m'`
   wht=`echo -e '\e[0m'`

   echo "+ module load sensei/3.0.0-catalyst-shared"
   module load sensei/3.0.0-catalyst-shared

   set -x

   cat ./configs/random_2d_${b}_catalyst.xml | sed "s/.*/$blu&$wht/"

   mpiexec -n ${n} \
       oscillator -b ${n} -t ${dt} -s ${b},${b},1 -g 1 -p 0 \
       -f ./configs/random_2d_${b}_catalyst.xml \
       ./configs/random_2d_${b}.osc 2>&1 | sed "s/.*/$red&$wht/"

During the run ParaView Catalyst is configured to render a pseudocolor plot of
the reaction rate field. The plot includes an iso-contour where the reaction
rate is 1.


.. _libsim_insitu_demo:

VisIt Libsim in situ demo
^^^^^^^^^^^^^^^^^^^^^^^^^
.. _libsim_insitu_image:
.. figure:: http://hpcvis.com/vis/sensei/rtd-docs/images/random_2d_64_libsim_00020.png
   :width: 80 %
   :align: center

   A pseudocolor plot rendered by VisIt Libsim of the rectaion rate field
   with an iso-contour plotted at a reaction rate of 1.0.

In the demo data from the reaction rate proxy simulation is processed using
VisIt Libsim. Libsim is selected at run time via the following SENSEI
XML:

.. _libsim_insitu_xml:
.. code-block:: xml

   <sensei>
     <analysis type="libsim" mode="batch" frequency="1"
               session="configs/random_2d_64_libsim.session"
               image-filename="random_2d_64_libsim_%ts"
               image-width="800" image-height="800" image-format="png"
               options="-debug 0" enabled="1" />
   </sensei>

The analysis element selects VisIt Libsim, the filename attribute points to
the Libsim specific configuration. In this case a session file that was
generated using the VisIt GUI. The following shell script runs the demo on the
VM.

.. _libsim_insitu_script:
.. code-block:: bash

   #!/bin/bash

   n=4
   b=64
   dt=0.25
   bld=`echo -e '\e[1m'`
   red=`echo -e '\e[31m'`
   grn=`echo -e '\e[32m'`
   blu=`echo -e '\e[36m'`
   wht=`echo -e '\e[0m'`

   echo "+ module load sensei/3.0.0-libsim-shared"
   module load sensei/3.0.0-libsim-shared

   set -x

   cat ./configs/random_2d_${b}_libsim.xml | sed "s/.*/$blu&$wht/"

   mpiexec -n ${n} \
       oscillator -b ${n} -t ${dt} -s ${b},${b},1 -g 1 -p 0 \
       -f ./configs/random_2d_${b}_libsim.xml \
       ./configs/random_2d_${b}.osc 2>&1 | sed "s/.*/$red&$wht/"

   Shell script that runs the Libsim in situ demo on the VM.

During the run VisIt Libsim is configured to render a pseudocolor plot of
the reaction rate field. The plot includes an iso-contour where the reaction
rate is 1.

.. _python_insitu_demo:

Python in situ demo
^^^^^^^^^^^^^^^^^^^
.. _python_insitu_image:
.. figure:: http://hpcvis.com/vis/sensei/rtd-docs/images/random_2d_64_python.png
   :width: 80 %
   :align: center

   A plot of the time history of the area of the 2D substrate where the
   reaction rate is greater or equal to 1.0.

In the demo data from the reaction rate proxy simulation is processed using
Python. Python is selected at run time via the following SENSEI XML:

.. _python_insitu_xml:
.. code-block:: xml

   <sensei>
     <analysis type="python" script_file="configs/volume_above_sm.py" enabled="1">
       <initialize_source>
   threshold=1.0
   mesh='mesh'
   array='data'
   cen=1
   out_file='random_2d_64_python.png'
        </initialize_source>
     </analysis>
   </sensei>


The analysis element selects Python, the script_file attribute points to the
user provided Python script and initialize_source contains run time
configuration. The following shell script runs the demo on the VM.


.. _python_insitu_script:
.. code-block:: bash

   #!/bin/bash

   n=4
   b=64
   dt=0.01
   bld=`echo -e '\e[1m'`
   red=`echo -e '\e[31m'`
   grn=`echo -e '\e[32m'`
   blu=`echo -e '\e[36m'`
   wht=`echo -e '\e[0m'`

   export MPLBACKEND=Agg

   echo "+ module load sensei/3.0.0-vtk-shared"
   module load sensei/3.0.0-vtk-shared

   set -x

   cat ./configs/random_2d_${b}_python.xml | sed "s/.*/$blu&$wht/"

   mpiexec -n ${n} oscillator -b ${n} -t ${dt} -s ${b},${b},1 -p 0 \
       -f ./configs/random_2d_${b}_python.xml \
       ./configs/random_2d_${b}.osc  2>&1 | sed "s/.*/$red&$wht/"

During the run this user provided Python script computes the area of the 2D
substrate where the reaction rate is greater or equal to 1. The value is stored
and at the end of the run a plot of the time history is made.


In transit demos
----------------

ParaView Catalyst
^^^^^^^^^^^^^^^^^

VisIt Libsim
^^^^^^^^^^^^

Ascent
^^^^^^

Python
^^^^^^

