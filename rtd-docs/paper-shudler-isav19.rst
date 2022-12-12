.. _ShudlerIsav19:

***************************************************************************
Spack meets singularity: creating movable in-situ analysis stacks with ease
***************************************************************************

Sergei Shudler, Nicola Ferrier, Joseph Insley, Michael E Papka, Silvio Rizzi

============
Full Text
============

Link to the full text `PDF <https://dl.acm.org/doi/abs/10.1145/3364228.3364682>`_.


========
Abstract
========

In situ data analysis and visualization is a promising technique to handle the enormous amount of data an extreme-scale application produces. One challenge users often face in adopting in situ techniques is setting the right environment on a target machine. Platforms such as SENSEI require complex software stacks that consist of various analysis packages and visualization applications. The user has to make sure all these prerequisites exist on the target machine, which often involves compiling and setting them up from scratch. In this paper, we leverage the containers technology (eg, light-weight virtualization images) and provide users with Singularity containers that encapsulate ready-to-use, movable in situ software stacks. Moreover, we make use of Spack to ease the process of creating these containers. Finally, we evaluate this solution by running in situ analysis from within a container on an HPC system.
