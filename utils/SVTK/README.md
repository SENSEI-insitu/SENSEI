![SVTK - The Visualization Toolkit][svtk-banner]

Introduction
============

SVTK is an open-source software system for image processing, 3D
graphics, volume rendering and visualization. SVTK includes many
advanced algorithms (e.g., surface reconstruction, implicit modeling,
decimation) and rendering techniques (e.g., hardware-accelerated
volume rendering, LOD control).

SVTK is used by academicians for teaching and research; by government
research institutions such as Los Alamos National Lab in the US or
CINECA in Italy; and by many commercial firms who use SVTK to build or
extend products.

The origin of SVTK is with the textbook "The Visualization Toolkit, an
Object-Oriented Approach to 3D Graphics" originally published by
Prentice Hall and now published by Kitware, Inc. (Third Edition ISBN
1-930934-07-6). SVTK has grown (since its initial release in 1994) to a
world-wide user base in the commercial, academic, and research
communities.

Learning Resources
==================

* General information is available at the [SVTK Homepage][svtk-homepage].

* Community discussion takes place on the [SVTK Discourse][svtk-discourse] forum.

* Commercial [support][kitware-support] and [training][kitware-training]
  are available from [Kitware][kitware].

* Doxygen-generated nightly reference documentation is
  available [online][svtk-doxygen].

Reporting Bugs
==============

If you have found a bug:

1. If you have a patch, please read the [CONTRIBUTING.md][svtk-contributing] document.

2. Otherwise, please join the [SVTK Discourse][svtk-discourse] forum and ask
   about the expected and observed behaviors to determine if it is
   really a bug.

3. Finally, if the issue is not resolved by the above steps, open
   an entry in the [SVTK Issue Tracker][svtk-issues].

Requirements
============

In general SVTK tries to be as portable as possible; the specific configurations below are known to work and tested.

SVTK supports the following C++11 compilers:
1. Microsoft Visual Studio 2015 or newer
2. gcc 4.8.3 or newer
3. Clang 3.3 or newer
4. Apple Clang 5.0 (from Xcode 5.0) or newer
5. Intel 14.0 or newer

SVTK supports the following operating systems:
1. Windows Vista or newer
2. Mac OS X 10.7 or newer
3. Linux (ex: Ubuntu 12.04 or newer, Debian 4 or newer)

Contributing
============

See [CONTRIBUTING.md][svtk-contributing] for instructions to contribute.

License
=======

SVTK is distributed under the OSI-approved BSD 3-clause License.
See [Copyright.txt][svtk-copyright] for details.


[kitware]: https://www.kitware.com/
[kitware-support]: https://www.kitware.com/what-we-offer/#support
[kitware-training]: https://www.kitware.com/what-we-offer/#training
[svtk-banner]: svtkBanner.gif
[svtk-contributing]: CONTRIBUTING.md#contributing-to-svtk
[svtk-copyright]: Copyright.txt
[svtk-discourse]: https://discourse.svtk.org/
[svtk-doxygen]: https://www.svtk.org/doc/nightly/html
[svtk-homepage]: https://www.svtk.org/
[svtk-issues]: https://gitlab.kitware.com/svtk/svtk/issues
