
export CC=`which gcc`
export CXX=`which g++`

cmake -S src/vtk -B build/vtk \
 -DVTK_Group_Imaging=OFF \
 -DVTK_Group_MPI=OFF \
 -DVTK_Group_Qt=OFF \
 -DVTK_Group_Rendering=OFF \
 -DVTK_Group_StandAlone=ON \
 -DVTK_Group_Tk=OFF \
 -DVTK_Group_Views=OFF \
 -DVTK_Group_Web=OFF \
 -DVTK_RENDERING_BACKEND=None \
 -DCMAKE_BUILD_TYPE=Debug \
 -DBUILD_TESTING=OFF \
 -DCMAKE_INSTALL_PREFIX=`pwd`/install/vtk
