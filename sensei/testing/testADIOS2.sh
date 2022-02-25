#!/usr/bin/env bash

if [[ $# -lt 9 ]]
then
  echo "Num Args Detected... $#"
  echo "test_adios.sh [mpiexec] [npflag] [nproc] [py exec] [src dir] [file] [write method] [read method] [nits] -- <optional MPI args>"
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

shift 9
if [ "$1" == "--" ]; then
  shift
fi

stepsPerFile=0
if [[ "${writeMethod}" == "BP4" ]]
then
  stepsPerFile=2
fi

trap 'eval echo $BASH_COMMAND' DEBUG

rm -rf ${file}

echo "testing ${writeMethod} -> ${readMethod}"
echo "M=${nproc_write} x N=${nproc_read}"

${mpiexec} ${@} ${npflag} ${nproc_write} ${pyexec} ${srcdir}/testADIOS2Write.py ${writeMethod} ${file} ${stepsPerFile} ${nits} &
writePid=$!

if [[ "${readMethod}" == "BP4" ]]
then
  # with BP wait for the write side to completely finish before starting the read side
  echo "waiting for writer(${writePid}) to complete"
  wait ${writePid}
fi

${mpiexec} ${@} ${npflag} ${nproc_read} ${pyexec} ${srcdir}/testADIOS2Read.py ${readMethod} ${file}
exit 0
