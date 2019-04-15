#!/usr/bin/env bash

if [[ $# < 10 ]]
then
  echo "testPartitioners.sh [mpiexec] [npflag] [M procs] [N procs] [src dir] [transport] [blocks x] [blocks y] [n its] [sync]"
  exit 1
fi

mpiexec=`basename $1`
npflag=$2
nproc_m=$3
nproc_n=$4
src_dir=$5
transport=$6
nblocks_x=$7
nblocks_y=$8
n_its=$9
sync_mode=${10}

parts_m=( block )

parts_n=( block planar_1 planar_2 \
  planar_3 mapped_ring mapped_subset )

for part_m in ${parts_m[*]}
do
  for part_n in ${parts_n[*]}
  do
    echo -n "Testing ${transport} M=${nproc_m} N=${nproc_n} part_M=${part_m} part_N=${part_n} ... "
    test_output=$(${src_dir}/testPartitioners.sh ${mpiexec} ${npflag} \
      ${nproc_m} ${nblocks_x} ${nblocks_y} ${nproc_n} "${src_dir}" \
      write_${transport}.xml catalyst_render_partition.xml \
      read_${transport}_${part_n}.xml ${n_its} ${sync_mode} 2>&1)
    test_stat=$?
    if [[ -n "${VERBOSE}" ]]
    then
      echo
      echo ${test_output}
    fi
    if (( $test_stat != 0 ))
    then
      echo "ERROR"
      exit -1
    fi
    echo "OK"
    for f in `ls {sender,receiver}_decomp_*.png`
    do
      mv $f test_${transport}_M${nproc_m}_N${nproc_n}_${part_m}_${part_n}_$f
    done
  done
done
