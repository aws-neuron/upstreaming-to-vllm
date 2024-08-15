#!/bin/bash

set -x #echo on

# Use environment variables with defaults

if [ -z "$K8S_MASTER_ADDR" ]; then
	MASTER_ADDR=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
else
	MASTER_ADDR=$K8S_MASTER_ADDR
fi
NEURON_RT_ROOT_COMM_ID=$MASTER_ADDR:8990
NEURON_RANK_ID=${K8S_NEURON_RANK_ID:-$SLURM_NODEID}
NEURON_LOCAL_TP=${K8S_NEURON_LOCAL_TP:-32}
CPU_COMM_ID=$MASTER_ADDR:8989
NODE_ADDR=${K8S_NODE_ADDR:-$(hostname)}

export NEURON_RT_ROOT_COMM_ID
export NEURON_RANK_ID
export NEURON_LOCAL_TP
export CPU_COMM_ID
export MASTER_ADDR

echo $NEURON_RT_ROOT_COMM_ID
echo $NEURON_RANK_ID
echo $NEURON_LOCAL_TP
echo $CPU_COMM_ID
echo $NODE_ADDR

# Install RT (commented out for safety)
echo "running script"
if [ -n "$SLURM_NODEID" ]; then
	sudo dpkg -i ./neuron_dependencies/*.deb
fi

echo "runtime setup done"
echo "$(apt list --installed | grep neuron)"

sudo modprobe -r neuron; sudo modprobe neuron
export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa

pip list | grep neuron
sudo apt list --installed | grep neuron

# Calculate GLOBAL_TP
WORKER_COUNT=${K8S_WORKER_COUNT:-2}
GLOBAL_TP=$((WORKER_COUNT * NEURON_LOCAL_TP))

# Export environment variables
export WORLD_SIZE=${WORKER_COUNT:-2}
export NEURONX_DUMP_TO="./Meta_Llama_3_1_70b_compiler_work_dir/"
export NEURON_MULTI_NODE=True

python neuron_multi_node_runner.py --model="./Meta_Llama_3_1_70b/" --max-num-seqs=2 --max-model-len=1024 --block-size=1024 --tensor-parallel-size=$GLOBAL_TP --port=8080
