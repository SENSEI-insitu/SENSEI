#!/bin/bash

set -e

# clean out last run
rm -rf sVTK-9.0.1

# download
if [[ ! -e VTK-9.0.1.tar.gz ]]
then
    wget https://www.vtk.org/files/release/9.0/VTK-9.0.1.tar.gz
fi

# unpack the sources
tar xzfv VTK-9.0.1.tar.gz

mv -v VTK-9.0.1 sVTK-9.0.1
cd sVTK-9.0.1

# mangle directories
for i in `seq 1 10`
do
  for d in `find ./ -mindepth $i -maxdepth $i -type d -name '*vtk*'`
  do
      dn=`dirname $d`;
      fn=`basename $d`;
      fno=`echo $fn | sed 's/vtk/svtk/g'`
      mv -v ${dn}/${fn} ${dn}/${fno}
  done

  for d in `find ./ -mindepth $i -maxdepth $i -type d -name '*VTK*'`
  do
      dn=`dirname $d`;
      fn=`basename $d`;
      fno=`echo $fn | sed 's/VTK/SVTK/g'`
      mv -v ${dn}/${fn} ${dn}/s${fno}
  done
done

# mangle files
for f in `find ./ -type f -name '*vtk*'`;
do
    dn=`dirname $f`;
    fn=`basename $f`;
    fno=`echo $fn | sed 's/vtk/svtk/g'`

    mv -v $f ${dn}/${fno}
done
for f in `find ./ -type f -name '*VTK*'`;
do
    dn=`dirname $f`;
    fn=`basename $f`;
    fno=`echo $fn | sed 's/VTK/SVTK/g'`

    mv -v $f ${dn}/${fno}
done

# mangle classes
for f in `grep vtk ./ -rIl`
do
    echo $f
    sed -i 's/vtk/svtk/g' $f
done

for f in `grep VTK ./ -rIl`
do
    echo $f
    sed -i 's/VTK/SVTK/g' $f
done


