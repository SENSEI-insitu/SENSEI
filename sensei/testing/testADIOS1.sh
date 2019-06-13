#!/usr/bin/env bash

if [[ $# < 8 ]]
then
  echo "test_adios.sh [mpiexec] [npflag] [nproc] [py exec] [src dir] [file] [write method] [read method] [nits]"
  exit 1
fi

mpiexec=`basename $1`
npflag=$2
nproc=$3
nproc_write=$nproc
let nproc_read=$nproc-1
nproc_read=$(( nproc_read < 1 ? 1 : nproc_read ))
pyexec=$4
srcdir=$5
file=$6
writeMethod=$7
readMethod=$8
nits=$9
delay=1
maxDelay=30

trap 'eval echo $BASH_COMMAND' DEBUG

rm -f ${file}

echo "testing ${writeMethod} -> ${readMethod}"
echo "M=${nproc_write} x N=${nproc_read}"

${mpiexec} ${npflag} ${nproc_write} ${pyexec} ${srcdir}/testADIOS1Write.py ${file} ${writeMethod} ${nits} &
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

${mpiexec} ${npflag} ${nproc_read} ${pyexec} ${srcdir}/testADIOS1Read.py ${file} ${readMethod}
exit 0
