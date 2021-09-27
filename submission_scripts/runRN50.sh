#!/bin/bash
#SBATCH -C gpu
#SBATCH -J ResNet50_amp
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH -A m1759
#SBATCH -t 4:00:00

module load cgpu
module unload pytorch
module load pytorch/1.8.0-gpu
echo "Training with pure mixed precision operations"

run_tag="ResNet50_cgpu_${SLURM_JOB_ID}"
output_dir=$SCRATCH/ResNet50/results/$run_tag
mkdir -p ${output_dir}

#to avoid non-determinism behavior, we collect all metrics and all kernels in a single run
kernel_set=("AllKernels")
metrics_set=("sm__cycles_elapsed.avg.per_second,sm__cycles_elapsed.avg,dram__bytes_read.sum,dram__bytes_write.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__inst_executed_pipe_tensor.sum,sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum")


for kernel in ${kernel_set[@]}; do
   for metric in ${metrics_set[@]}; do
        kernel_trunc=${kernel:0:99}
        metric_trunc=${metric:0:99}
        outputstr=RN50.ker_${kernel_trunc}.${metric_trunc}
        profilestring="nv-nsight-cu-cli --metrics ${metric} -o ${output_dir}/${outputstr}"
        srun -u -n 1 -c 1 --cpu_bind=cores ${profilestring} python main_amp.py -a resnet50 --epochs 1 --b 128 --workers 1 --opt-level O1 ./
   done
done

