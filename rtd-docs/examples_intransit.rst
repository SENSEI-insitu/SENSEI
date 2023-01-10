.. _example_intransit:

In Transit MxN communication with LAMMPS, SENSEI, and Paraview/Catalyst
=======================================================================

Synopsis
########

In this example we instrument the molecular dynamics simulation code LAMMPS with SENSEI and demonstrate in transit capabilities. Our example showcases M to N ranks redistribution and the Catalyst analysis adaptor to generate a Cinema database.

Setting Up
##########

This example uses Docker containers. There are two containers: (1) Producer, which uses the LAMMPS molecular dynamics simulator instrumented with SENSEI; and (2) Consumer, which uses the SENSEI endpoint and the Paraview/Catalyst analysis adaptor. 

A Zenodo artifact is available at https://zenodo.org/record/6336286 , containing Dockerfile recipes to build the Producer and Consumer. Please refer to file "containers.zip"

You should be able to build your own images with the Dockerfiles provided. If you would like to use prebuilt docker containers you can get them from the Docker hub:

.. highlight:: shell

::

   docker pull srizzi/woiv22producer_runtime
   docker pull srizzi/woiv22consumer_runtime

If your site uses Singularity you can build Singularity images pulling from the Docker hub

Running the Example
###################

From the same Zenodo artifact, you can download "in_transit_demo_files.zip" , containing SENSEI xml configuration files and scripts to run the demo.

The LAMMPS producer is configured to run a simple simulation configured in file in_lj

With the parameters in in_lj the simulation evolves about 16 million hydrogen atoms. If you would like to change the size of the simulation, edit the multipliers for xx, yy, and zz in file in_lj. The multipliers are 16 in the file provided, which results in 67108864 atoms simulated.

The producer is launched with producer.sh . Notice that settings in this script are specific for ThetaGPU, but you should not find any major difficulties if you wish to adapt it for your system. The script also relies on a local build of mpich.

Notice that SENSEI uses the file adios_write_sst.xml to configure its backend. You will likely need to change the NetworkInterface in this xml file with an appropiate value for your own system.

The consumer side contains SENSEI with ParaView/Catalyst. For simplicity, in this demo we use the PosthocIO backend, which saves the received data in VTK format.

The consumer is launched with script consumer.sh and there are two xml configuration files required for SENSEI. The first defines the network transport and its called adios_transport_sst.xml . Once again, you may need to change the NetworkInterface parameter in this file according to your system.
The second xml file, vtk_io.xml in this case, activates the PosthocIO analysis adaptor in SENSEI and specifies a directory to save the data.

These are intended to run on different machines and different ranks on producer (M) and consumer (N). The scripts provided will launch 16 MPI ranks on the producer and 4 MPI ranks on the consumer.

Results
#######

The simulation is configure to run five timesteps. The SENSEI endpoint should receive data for each timestep and save it as VTK files. 
