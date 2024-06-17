import socket
import os
import torch
import torch.distributed as dist
import fairscale.nn.model_parallel.initialize as fs_init
from types import SimpleNamespace


def init_distributed_mode(args=SimpleNamespace()):
    def find_free_port(start_port: int, end_port: int):
        """
        Find a free port within the specified range.
        """
        for port in range(start_port, end_port):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.bind(("", port))  # Try to bind to the port
                s.close()  # Close the socket if successful
                return port
            except OSError as e:
                # print(f"Port {port} is in use, trying next port.")
                continue
        raise RuntimeError(f"No free ports found in range {start_port}-{end_port}")
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ and "LOCAL_RANK" in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.rank = int(os.environ["RANK"])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.local_rank = args.gpu
        args.dist_url = 'env://'
    else:
        os.environ['MASTER_ADDR'] = "127.0.0.1"
        os.environ['MASTER_PORT'] = str(find_free_port(9000, 10000))
        os.environ['RANK'] = '0'
        os.environ['LOCAL_RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        args.rank = 0
        args.gpu = args.local_rank = 0
        args.world_size = 1
        args.dist_url = 'env://'

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()


def init_env():
    # define the model
    init_distributed_mode()
    fs_init.initialize_model_parallel(dist.get_world_size())
