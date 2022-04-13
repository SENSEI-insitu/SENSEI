#!/usr/bin/env bash

if [[ $# -lt 13 ]]
then
  echo "testPartitioners.sh [mpiexec] [npflag] [writer nproc] [blocks x] [blocks y] [reader nproc] [src dir] [writer analysis xml] [reader analysis xml] [reader transport xml] [nits] [sync] -- <optional MPI args>"
  exit 1
fi

python=$1
mpiexec=`basename $2`
npflag=$3
nproc_write=$4
nblock_x=$5
nblock_y=$6
nproc_read=$7
srcdir=$8
writer_analysis_xml=$9
reader_analysis_xml=${10}
reader_transport_xml=${11}
nits=${12}
sync_mode=${13}
delay=1
maxDelay=30
shift 13
if [ "$1" == "--" ]; then
  shift
fi

nblocks=`echo ${nblock_x}*${nblock_y} | bc`

trap 'eval echo $BASH_COMMAND' DEBUG

# ADIOS1
# safe guard against left over files from a previous crashed run
# if these are present the test may deadlock or crash
file=test.bp
rm -rf $file ${file}_writer_info.txt ${file}.sst

# ADIOS2 file series mode
rm -rf test_*.bp

export PROFILER_ENABLE=2 PROFILER_LOG_FILE=WriterTimes.csv MEMPROF_LOG_FILE=WriterMemProf.csv

${mpiexec} ${@} ${npflag} ${nproc_write} ${python} -m mpi4py \
  ${srcdir}/testPartitionersWrite.py \
  "${srcdir}/${writer_analysis_xml}" ${nits} ${nblock_x} ${nblock_y} 16 16    \
  -6.2832 6.2832 -6.2832 6.2832 0 6.2832 &
writePid=$!

# TODO -- this should be integrated into the adaptors
if [[ "${sync_mode}" == "0" ]]
then
  # wait for the write side to completely finish before starting the read side
  echo "waiting for writer(${writePid}) to complete"
  wait ${writePid}
elif [[ "${sync_mode}" == "1" ]]
then
  # wait for the writer to start before starting the read side look for the
  # files that it spits out with connection info as evidence that it is ready
  echo "waiting for writer to start ${delay}"
  while [[ True ]]
  do
    if [[ -e "${file}_writer_info.txt"  || -e "${file}.sst" ]]
    then
      break
    elif [[ ${maxDelay} -le 0 ]]
    then
      echo "ERROR: max delay exceded"
      kill -15 ${writePid}
      exit -1
    else
      sleep ${delay}s
      let maxDelay=${maxDelay}-${delay}
    fi
  done
fi

export PROFILER_ENABLE=2 TIMER_LOG_FILE=ReaderTimes.csv MEMPROF_LOG_FILE=ReaderMemProf.csv

${mpiexec} ${@} ${npflag} ${nproc_read} ${python} -m mpi4py \
  ${srcdir}/testPartitionersRead.py \
  "${srcdir}/${reader_analysis_xml}" "${srcdir}/${reader_transport_xml}"

test_stat=$?

exit $test_stat
