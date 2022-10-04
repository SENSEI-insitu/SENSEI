
.. _loringEgpgv20:

************************************************************************************************************
Improving Performance of M-to-N Processing and Data Redistribution in In Transit Analysis and Visualization
************************************************************************************************************

B. Loring 1, M. Wolf, J. Kress 2, S. Shudler, J. Gu , S. Rizzi, J. Logan, N. Ferrier, and E. W. Bethel

============
Full Text
============

Link to the full text `PDF <https://diglib.eg.org/handle/10.2312/pgv20201073>`_ .

============
Abstract
============

In an in transit setting, a parallel data producer, such as a numerical
simulation, runs on one set of ranks M, while a data consumer, such as a
parallel visualization application, runs on a different set of ranks N. One of
the central challenges in this in transit setting is to determine the mapping
of data from the set of M producer ranks to the set of N consumer ranks. This
is a challenging problem for several reasons, such as the producer and consumer
codes potentially having different scaling characteristics and different data
models. The resulting mapping from M to N ranks can have a significant impact
on aggregate application performance. In this work, we present an approach for
performing this M-to-N mapping in a way that has broad applicability across a
diversity of data producer and consumer applications. We evaluate its design
and performance with a study that runs at high concurrency on a modern HPC
platform. By leveraging design characteristics, which facilitate an
“intelligent” mapping from M-to-N, we observe significant performance gains are
possible in terms of several different metrics, including time-to-solution and
amount of data moved.
