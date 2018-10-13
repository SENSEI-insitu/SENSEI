title: "About SENSEI"
markdown:
  gfm: true
  breaks: false
---

This project takes aim at a set of research challenges for enabling scientific knowledge discovery within the context of in situ processing at extreme-scale concurrency. This work is motivated by a widening gap between FLOPs and I/O capacity which will make full-resolution, I/O-intensive post hoc analysis prohibitively expensive, if not impossible.

We focus on new algorithms for analysis, and visualization – topological, geometric, statistical analysis, flow field analysis, pattern detection and matching – suitable for use in an in situ context aimed specifically at enabling scientific knowledge discovery in several exemplar application areas of importance to DOE.

Complementary to the in situ algorithmic work, we focus on several leading in situ infrastructures, and tackle research questions germane to enabling new algorithms to run at scale across a diversity of existing in situ implementations.

Our intent is to move the field of in situ processing in a direction where it may ultimately be possible to write an algorithm once, then have it execute in one of several different in situ software implementations. The combination of algorithmic and infrastructure work is grounded in direct interactions with specific application code teams, all of which are engaged in their own R&D aimed at evolving to the exascale.

## Impact

This approach blends algorithmic R&D with focused solutions for important science problems, and pushes the limits of existing in situ infrastructure as applied to DOE science problems on extreme-concurrency platforms. This work will likely have immediate impact on several science areas through our collaborations, who are in desperate need of new analysis methods resulting for data of increasing size and complexity, as well as longer-term impact due to the emphasis on wider dissemination and distribution of these new in situ analysis algorithms as part of several different in situ frameworks. The result is an increased lifespan of software investments in key infrastructure.

## Science-Facing Projects

A primary driver for our in situ analysis algorithm R&D stems from science needs:

+ Simulations are increasingly multi-scale both in time and space;
+ The data they compute are increasingly complex;
+ It is increasingly impractical to do full-resolution I/O;
+ Data is not saved nor analyzed resulting in lost science.

To identify needs and evaluate solutions, we are using several science-facing projects, and their attendant science drivers, to shape a new generation of in situ analysis algorithm development.

## _In Situ_ Infrastructure

The in situ infrastructure thrust focuses on four in situ frameworks, namely Catalyst, Libsim, and ADIOS. While these infrastructures provide the means for performing various types of processing, analysis, and visualization operations from a live-running simulation, they differ in their approach to interfacing with the simulation, with how they are configured, and how they are extended by user-supplied code.

An infrastructure-based goal of the project is:

+ To achieve write-once, run-anywhere analysis.

A step to achieve this vision is by identifying the building blocks to design and refactor analysis codes on diverse in situ infrastructures as well as on a wide-variety of systems. By achieving this vision, we can enable analysis algorithms to fully exploit the underlying concurrency and heterogeneity of the supercomputing systems.

## Funding

This work is supported by the
Director, Office of Science,
Office of Advanced Scientific Computing Research,
of the U.S. Department of Energy under Contract No. DE-AC02-05CH11231,
through the grant
“Scalable Analysis Methods and In Situ Infrastructure for Extreme Scale Knowledge Discovery,”
program manager Dr. Laura Biven.

<!-- extra line breaks to prevent footer from obscuring text -->
<br><br><br>
