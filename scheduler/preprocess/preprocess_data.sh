#!/bin/bash

ncpus='24'
mem='40G'
walltime='30:00:00'
name='preprocess_data'
script_location='/home/ftian/storage/ECG_XAI/src/scripts/preprocess/preprocess_data.py'

function submit_job {
    cmd_script="source deactivate; source activate ecg_xai; python ${script_location}; source deactivate"

    job_path="/home/ftian/storage/ECG_XAI/logs/scheduler/preprocess/${name}/"
    mkdir -p ${job_path}
    job_err="${job_path}${name}_error.txt"
    job_out="${job_path}${name}_out.txt"

    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd_script" -walltime $walltime -mem $mem -name $name -joberr $job_err -jobout $job_out -ncpus $ncpus
}

submit_job
