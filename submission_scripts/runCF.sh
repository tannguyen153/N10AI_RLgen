#!/bin/bash
#SBATCH -C gpu -c 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 1
#SBATCH --exclusive
#SBATCH -t 4:00:00
#SBATCH -J train-cgpu

module load tensorflow/2.4.1-gpu

set -x

run_tag="CosmoFlow_cgpu_${SLURM_JOB_ID}"
output_dir=$SCRATCH/CosmoFlow/results/$run_tag
mkdir -p ${output_dir}


#to avoid non-determinism behavior, we collect all metrics and all kernels in a single run
kernel_set=("AllKernels")
metrics_set=("sm__cycles_elapsed.avg.per_second,sm__cycles_elapsed.avg,dram__bytes_read.sum,dram__bytes_write.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,sm__inst_executed_pipe_tensor.sum,sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,sm__sass_thread_inst_executed_op_hmul_pred_on.sum")


for kernel in ${kernel_set[@]}; do
   for metric in ${metrics_set[@]}; do
	kernel_trunc=${kernel:0:99}
	metric_trunc=${metric:0:99}
        outputstr=CF.ker_${kernel_trunc}.metrics_${metric_trunc}
	#skip initial kernels that are not part of training iterations
        profilestring="nv-nsight-cu-cli -s 65 --metrics ${metric} --kernel-regex-base demangled -o ${output_dir}/${outputstr}"
	#Note: batch size can be configured in configs/cosmo.yaml  
	srun -u -n 1 -c 1 ${profilestring}  python train.py -d --rank-gpu --amp $@
   done
done


