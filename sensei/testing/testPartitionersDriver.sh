#!/usr/bin/env bash

if [[ $# < 11 ]]
then
  echo "testPartitioners.sh [python] [mpiexec] [npflag] [M procs] [N procs] [src dir] [transport] [blocks x] [blocks y] [n its] [sync]"
  exit 1
fi

python=$1
mpiexec=`basename $2`
npflag=$3
nproc_m=$4
nproc_n=$5
src_dir=$6
transport=$7
nblocks_x=$8
nblocks_y=$9
n_its=${10}
sync_mode=${11}

# TODO -- write side should exercise more than one partitioner.
# currenlty hardcoded to use the block partitioenr.
parts_m=( block )

# different transports have different capabilities. for each capability
# to test, include an XML following the pattern: read_<transport>_<partitioner>.xml
# parts_n=`find ${src_dir} -name read_${transport}*.xml -exec basename \{\} \; | sed s/read_${transport}_//g | cut -d. -f1 | sort`
#
if [[ "$OSTYPE" == "darwin"* ]]; then
# mac:
   echo "find ${src_dir}/read_${transport}*.xml -exec basename \{\} \; | sed s/read_${transport}_//g | cut -d. -f1 | sort"
   parts_n=`find ${src_dir}/read_${transport}*.xml -exec basename \{\} \; | sed s/read_${transport}_//g | cut -d. -f1 | sort`
else
# other os:
   echo "find ${src_dir} -name read_${transport}*.xml -exec basename \{\} \; | sed s/read_${transport}_//g | cut -d. -f1 | sort"
   parts_n=`find ${src_dir} -name read_${transport}*.xml -exec basename \{\} \; | sed s/read_${transport}_//g | cut -d. -f1 | sort`
fi
for part_m in ${parts_m[*]}
do
  for part_n in ${parts_n[*]}
  do
    echo -n "Testing ${transport} M=${nproc_m} N=${nproc_n} part_M=${part_m} part_N=${part_n} ... "
    cmd="${src_dir}/testPartitioners.sh ${python} ${mpiexec} ${npflag} ${nproc_m} ${nblocks_x} ${nblocks_y} ${nproc_n} \"${src_dir}\" write_${transport}.xml catalyst_render_partition.xml read_${transport}_${part_n}.xml ${n_its} ${sync_mode}"
    if [[ -z "${VERBOSE}" ]]
    then
      test_output=$(eval ${cmd} 2>&1)
    else
      eval ${cmd}
    fi
    test_stat=$?
    if (( $test_stat != 0 ))
    then
      echo "ERROR"
      echo
      echo ${test_output}
      echo
      exit -1
    fi
    echo "OK"
    for f in `ls {sender,receiver}_decomp_*.png`
    do
      mv $f test_${transport}_M${nproc_m}_N${nproc_n}_${part_m}_${part_n}_$f
    done
  done
done
