
.. _mateevitsiIsav23:

*********************************************************************************
Scaling Computational Fluid Dynamics: In Situ Visualization of NekRS using SENSEI
*********************************************************************************

 
V. A. Mateevitsi, M. Bode, N. Ferrier, P. Fischer, J. H. Göbbert, J. A. Insley, Y. H. Lan, M. Min, M. E. Papka, S. Patel, S. Rizzi, J. Windgassen

============
Full Text
============

Link to the full text `PDF <https://dl.acm.org/doi/abs/10.1145/3624062.3624159>`_ .

============
Abstract
============

In the realm of Computational Fluid Dynamics (CFD), the demand for memory and 
computation resources is extreme, necessitating the use of leadership-scale 
computing platforms for practical domain sizes. This intensive requirement renders 
traditional checkpointing methods ineffective due to the significant slowdown in 
simulations while saving state data to disk. As we progress towards exascale and 
GPU-driven High-Performance Computing (HPC) and confront larger problem sizes, the 
choice becomes increasingly stark: to compromise data fidelity or to reduce resolution.
To navigate this challenge, this study advocates for the use of in situ analysis and 
visualization techniques. These allow more frequent data "snapshots" to be taken 
directly from memory, thus avoiding the need for disruptive checkpointing. We detail 
our approach of instrumenting NekRS, a GPU-focused thermal-fluid simulation code employing 
the spectral element method (SEM), and describe varied in situ and in transit strategies 
for data rendering. Additionally, we provide concrete scientific use-cases and report on 
runs performed on Polaris, Argonne Leadership Computing Facility’s (ALCF) 44 Petaflop 
supercomputer and Jülich Wizard for European Leadership Science (JUWELS) Booster, Jülich 
Supercomputing Centre’s (JSC) 71 Petaflop High Performance Computing (HPC) system, 
offering practical insight into the implications of our methodology.
