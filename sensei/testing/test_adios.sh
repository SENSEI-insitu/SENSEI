#!/usr/bin/env bash

if [[ $# < 6 ]]
then
  echo "test_adios.sh [nproc] [src dir] [file] [write method] [read method] [nits]"
  exit 1
fi

nproc=$1
srcdir=$2
file=$3
write_method=$4
read_method=$5
nits=$6
delay=1s

trap 'echo $BASH_COMMAND' DEBUG

rm -f ${file}

mpiexec -np ${nproc} python ${srcdir}/test_adios_write.py ${file} ${write_method} ${nits} &
write_pid=$!

if [[ "${read_method}" == "BP" ]]
then
  echo "waiting for writer(${write_pid}) to complete"
  wait ${write_pid}
elif [[ "${read_method}" == "FLEXPATH" ]]
then
  echo "waiting for writer to start ${delay}"
  while [[ True ]]
  do
    if [[ -e "${file}_writer_info.txt" ]]
    then
      break
    else
      sleep ${delay}
    fi
  done
fi

mpiexec -np ${nproc} python ${srcdir}/test_adios_read.py ${file} ${read_method}
exit 0
