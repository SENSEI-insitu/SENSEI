.. _example_example:

Example Template
==========================

Come up with a decent title and underline it with your favorite non-alphanumeric character (# is a good choice).

Synopsis
########

Write a quick synopsis of your example. Detail briefly how it uses SENSEI and what code you are connecting to. `Hyperlinks <https://xkcd.com/2632/>`_ to the science code/repository are appropriated here.

Setting Up
##########

Here, we provide instructions on how to setup the build environment. Briefly describe which options you want for this example in SENSEI, but don't worry about going into detail about building dependencies that SENSEI needs as that is documented elsewhere (I hope...). Detail whatever unique build options and weird steps are required to build the science code with SENSEI enabled, but do not go into depth on how to build that science code in general, simply link to their build instructions (which are hopefully well documented).

If there are multiple ways of setting up the science code, or there are multiple ways to build SENSEI which provide different functionality, detail those variations here as well.

Running the Example
###################

Next, we detail the procedures to run the example. You will want to include your SENSEI config xml(s) here like this:

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

Subsections would be worthwhile if there are multiple ways to run the example.

You can write terminal commands here like this:

.. code-block:: shell

mpirun -np 1 /path/to/oscillator -g 1 -t 0.01 -f oscillator_catalyst_steering.xml simple.osc

If you have simple supplementary scripts (like Catalyst Python scripts, or bash scripts), you may want to write them in here as well, though sometimes these get quite long, and a hyperlink to the source code will suffice.

For interactive examples (i.e. steering), provide instructions for interacting with the example.

Results
#######

Finally, briefly showcase what successfully running the example will yield. Provide pictures of output, or sample data output etc.
