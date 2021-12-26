#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -o /data1/projects/MicroFunc/Jurjen/programs/packages/fmriprep
#$ -j Y
#$ -q long.q
#$ -V
#$ -pe smp 10
#Template provided by Daniel Levitas of Indiana University
#Edits by Andrew Jahn, University of Michigan, 07.22.2020

subj=${1}
nthreads=15
mem=20 #gb
container=singularity #docker or singularity

if [[ `whoami` != "fsluser" ]]; then
  base_dir=/data1/projects/MicroFunc/Jurjen
else
  base_dir=/mnt/hgfs/shared/spinoza/
fi

#Begin:
setup_file=${base_dir}/programs/linescanning/shell/spinoza_setup
source ${setup_file}

if [[ ! -f ${setup_file} ]]; then
  echo "no setup file present. please specify the path to spinoza_setup.sh"
  exit 1
fi

#Convert virtual memory from gb to mb
mem=`echo "${mem//[!0-9]/}"` #remove gb at end
mem_mb=`echo $(((mem*1000)-5000))` #reduce some memory for buffer space during pre-processing

#export TEMPLATEFLOW_HOME=$HOME/.cache/templateflow

fprep_container=${base_dir}/programs/packages/fmriprep/containers_bids-fmriprep--20.2.0.simg

if [[ ! -f ${fprep_container} ]]; then
  echo "Could not find singularity image for fmriprep.."
  exit 1
fi

#Run fmriprep
if [ ${container} == singularity ]; then
  if [[ ${2} == "anat" ]]; then
    unset PYTHONPATH; singularity run -B ${PATH_HOME}:${PATH_HOME} ${fprep_container} \
      ${DIR_DATA_HOME} ${DIR_DATA_DERIV} \
      participant \
      --participant-label ${subj} \
      --skip-bids-validation \
      --md-only-boilerplate \
      --fs-license-file ${DIR_SCRIPTS}/bin/utils/license.txt \
      --output-spaces fsnative \
      --anat-only \
      --nthreads ${nthreads} \
      --stop-on-first-crash \
      -w ${base_dir}/programs/packages/fmriprep

  else

    unset PYTHONPATH; singularity run -B ${PATH_HOME}:${PATH_HOME} ${fprep_container} \
      ${DIR_DATA_HOME} ${DIR_DATA_DERIV} \
      participant \
      --participant-label ${subj} \
      --skip-bids-validation \
      --md-only-boilerplate \
      --fs-license-file ${DIR_SCRIPTS}/bin/utils/license.txt \
      --output-spaces fsnative \
      --nthreads ${nthreads} \
      --stop-on-first-crash \
      -w ${base_dir}/programs/packages/fmriprep

  fi
fi
