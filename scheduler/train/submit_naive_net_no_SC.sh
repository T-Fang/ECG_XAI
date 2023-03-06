#!/bin/bash

ngpus='1'
ncpus='4'
mem='30G'
walltime='25:00:00'
name='naive_net_no_SC'
script_location='/home/ftian/storage/pMFM_speedup/src/training/training_script/basic_models/naive_net_no_SC.py'

function submit_job {
    cmd_script="source deactivate; source activate pMFM_speedup-torch1.8; python ${script_location}; source deactivate"

    job_path="/home/ftian/storage/pMFM_speedup/submit_to_scheduler/jobs/training/basic_models/${name}/"
    mkdir -p ${job_path}
    job_err="${job_path}${name}_error.txt"
    job_out="${job_path}${name}_out.txt"

    $CBIG_CODE_DIR/setup/CBIG_pbsubmit -cmd "$cmd_script" -walltime $walltime -mem $mem -name $name -joberr $job_err -jobout $job_out -ngpus $ngpus -ncpus $ncpus
}

submit_job
