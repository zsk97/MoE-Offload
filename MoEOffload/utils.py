import os
import socket
from types import SimpleNamespace
from contextlib import contextmanager
import torch
""" utility functions that help you process nested dicts, tuples, lists and namedtuples """

from MoEOffload.custom_layers import SwitchMoeWrapperV1


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

def nested_compare(t, u):
    """
    Return whether nested structure of t1 and t2 matches.
    """
    if isinstance(t, (list, tuple)):
        if not isinstance(u, type(t)):
            return False
        if len(t) != len(u):
            return False
        for a, b in zip(t, u):
            if not nested_compare(a, b):
                return False
        return True

    if isinstance(t, dict):
        if not isinstance(u, dict):
            return False
        if set(t.keys()) != set(u.keys()):
            return False
        for k in t:
            if not nested_compare(t[k], u[k]):
                return False
        return True

    else:
        return True


def nested_flatten(t):
    """
    Turn nested list/tuple/dict into a flat iterator.
    """
    if isinstance(t, (list, tuple)):
        for x in t:
            yield from nested_flatten(x)
    elif isinstance(t, dict):
        for k, v in sorted(t.items()):
            yield from nested_flatten(v)
    else:
        yield t


def nested_pack(flat, structure):
    """
    Restore nested structure from flattened state
    :param flat: result of nested_flatten
    :param structure: used as example when recovering structure
    :returns: nested structure like :structure: filled with elements of :flat:
    """
    return _nested_pack(iter(flat), structure)


def _nested_pack(flat_iter, structure):
    if is_namedtuple(structure):
        return type(structure)(*[_nested_pack(flat_iter, x) for x in structure])
    elif isinstance(structure, (list, tuple)):
        return type(structure)(_nested_pack(flat_iter, x) for x in structure)
    elif isinstance(structure, dict):
        return {k: _nested_pack(flat_iter, v) for k, v in sorted(structure.items())}
    else:
        return next(flat_iter)


def is_namedtuple(x):
    """Checks if x is a namedtuple instance. Taken from https://stackoverflow.com/a/2166841 ."""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, "_fields", None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


def nested_map(fn, *t):
    # Check arguments.
    if not t:
        raise ValueError("Expected 2+ arguments, got 1")
    for i in range(1, len(t)):
        if not nested_compare(t[0], t[i]):
            msg = "Nested structure of %r and %r differs"
            raise ValueError(msg % (t[0], t[i]))

    flat = map(nested_flatten, t)
    return nested_pack(map(fn, *flat), t[0])

@contextmanager
def with_default_dtype(dtype):
    _dtype_original = torch.get_default_dtype()

    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(_dtype_original)

def forward_pre_hook(module, input):
    if isinstance(module, SwitchMoeWrapperV1):
        torch.cuda.nvtx.range_push(f"Layer ID {module.layer_id} {module.__class__.__name__}")
    else:
        torch.cuda.nvtx.range_push(f"Layer ID {module.__class__.__name__}")

def forward_post_hook(module, input, output):
    torch.cuda.nvtx.range_pop()