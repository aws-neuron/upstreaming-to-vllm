from vllm.entrypoints.neuron_multi_node import api_server
import os
import torch
import argparse


print(os.getenv(k) for k in [''])
def main():
    rank_id = int(os.getenv("NEURON_RANK_ID", "0"))
    if rank_id == 0:
        master()
    else:
        main_worker()



def master():
    rank_id = int(os.getenv("NEURON_RANK_ID", "0"))
    print(f"**** init master node with rank_id: {rank_id}")
    args, _ = api_server.initialize_worker()
    api_server.run_master(args)
    # call asyn llm engine


def main_worker():
    rank_id = int(os.getenv("NEURON_RANK_ID", "0"))
    print(f"**** init worker node with rank_id: {rank_id}")
    args, engine = api_server.initialize_worker()
    worker = engine.engine.model_executor.driver_worker
    while True:
        worker.execute_model()


if "__main__" == __name__:
    main()