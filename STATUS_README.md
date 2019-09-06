Nicole's Work on Pencil Partition


Using Oscillator in_transit script: You need to change the PencilPartition.cxx code to take into account the negative extents. There is a off-by-one error when checking if the boxes intersect. The line of code required is commented out. 

Also, I never reconciled the ghost data that is included when more blocks are added. 
The oscillator data naturally overlaps so the container data size ends up being bigger than the actual size of the data (global extent)
This will break the fft because the shapes will not match.

Using testPartitionWrite.py in_transit script: GetArray is breaking everything as of 9/6


