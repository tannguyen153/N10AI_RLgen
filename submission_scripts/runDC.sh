#!/bin/bash
#SBATCH -J deepcam-cgpu
#SBATCH -C gpu
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 1
#SBATCH --cpus-per-task 10
#SBATCH --time 04:00:00

# Setup software environment
module load cgpu
module load pytorch/1.8.0-gpu

# Job configuration
rankspernode=1
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))
run_tag="deepcam_cgpu_${SLURM_JOB_ID}"
data_dir_prefix="/global/cscratch1/sd/sfarrell/deepcam/data/dry-run"
#data_dir_prefix="/global/cscratch1/sd/tkurth/data/cam5_data/All-Hist"
output_dir=$SCRATCH/deepcam/results/$run_tag

# Create files
mkdir -p ${output_dir}
touch ${output_dir}/train.out
git clone https://github.com/mlperf/logging.git mlperf-logging
pip install --user -e mlperf-logging


kernel_set=("AllKernels")
metrics_set=("sm__cycles_elapsed.avg.per_second,sm__cycles_elapsed.avg,dram__bytes_read.sum,dram__bytes_write.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__inst_executed_pipe_tensor.sum,sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum")

# Run training

for kernel in ${kernel_set[@]}; do
   for metric in ${metrics_set[@]}; do
	kernel_trunc=${kernel:0:99}
        metric_trunc=${metric:0:99}
        outputstr=DC.ker_${kernel_trunc}.metrics_${metric_trunc}	
	profilestring="nv-nsight-cu-cli -s 3 --metrics ${metric} --kernel-regex-base demangled -o ${output_dir}/${outputstr}"
	srun -u -N ${SLURM_NNODES} -n ${totalranks} -c $(( 10 / ${rankspernode} )) --cpu_bind=cores \
     ${profilestring} python ../train_hdf5_ddp.py \
     --wireup_method "nccl-slurm" \
     --run_tag ${run_tag} \
     --data_dir_prefix ${data_dir_prefix} \
     --output_dir ${output_dir} \
     --max_inter_threads 2 \
     --model_prefix "classifier" \
     --optimizer "Adam" \
     --start_lr 1e-3 \
     --lr_schedule type="multistep",milestones="8192 16384",decay_rate="0.1" \
     --lr_warmup_steps 0 \
     --lr_warmup_factor 1. \
     --weight_decay 1e-2 \
     --validation_frequency 200 \
     --training_visualization_frequency 0 \
     --validation_visualization_frequency 0 \
     --logging_frequency 100 \
     --save_frequency 400 \
     --max_epochs 1 \
     --amp_opt_level O1 \
     --local_batch_size 4 |& tee -a ${output_dir}/train.out
     #--max_epochs 200 \
     #--max_validation_steps 50 \
     #--enable_wandb \
     #--wandb_certdir $HOME \
   done
done


#remove the temporary logging tool
rm -r mlperf-logging

