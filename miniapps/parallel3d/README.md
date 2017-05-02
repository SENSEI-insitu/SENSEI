# Parallel3D
Code for miniapp from year I. The mini-app generates data on a uniform grid.
Infrastructures and custom analyses are configured via the XML file
[3dgrid.xml](configs/3dgrid.xml).

Usage (ENABLE_SENSEI=ON):
```bash
./bin/3D_Grid -g 4x4x4 -l 2x2x2 -f config.xml
    -g global dimensions
    -l local (per-process) dimensions
    -f SENSEI xml configuration file for analysis
```

Usage (ENABLE_SENSEI=OFF):
```bash
./bin/3D_Grid -g 4x4x4 -l 2x2x2 -b 10
    -g global dimensions
    -l local (per-process) dimensions
    -b histogram bins
```
