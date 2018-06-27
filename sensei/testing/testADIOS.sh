#!/usr/bin/env bash

if [[ $# < 8 ]]
then
  echo "test_adios.sh [mpiexec] [npflag] [nproc] [src dir] [file] [write method] [read method] [nits]"
  exit 1
fi

mpiexec=$1
npflag=$2
nproc=$3
srcdir=$4
file=$5
writeMethod=$6
readMethod=$7
nits=$8
delay=1
maxDelay=30

trap 'echo $BASH_COMMAND' DEBUG

rm -f ${file}

echo "testing ${writeMethod} -> ${readMethod}"

${mpiexec} ${npflag} ${nproc} python ${srcdir}/testADIOSWrite.py ${file} ${writeMethod} ${nits} &
writePid=$!

if [[ "${readMethod}" == "BP" ]]
then
  # with BP wait for the write side to completely finish before starting the read side
  echo "waiting for writer(${writePid}) to complete"
  wait ${writePid}
elif [[ "${readMethod}" == "FLEXPATH" ]]
then
  # with FLEXPATH wait for the writer to start before starting the read side
  # look for the files that it spits out with connection info as eviddence that
  # it is ready
  echo "waiting for writer to start ${delay}"
  while [[ True ]]
  do
    if [[ -e "${file}_writer_info.txt" ]]
    then
      break
    elif [[ ${maxDelay} -le 0 ]]
    then
      echo "ERROR: max delay exceded"
      exit -1
    else
      sleep ${delay}s
      let maxDelay=${maxDelay}-${delay}
    fi
  done
fi

${mpiexec} ${npflag} ${nproc} python ${srcdir}/testADIOSRead.py ${file} ${readMethod}
exit 0
