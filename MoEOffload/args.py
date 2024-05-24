import argparse

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="MoEOffload Arguments")

    # Add arguments
    parser.add_argument('--model_path', type=str, help='Model state path')
    parser.add_argument('--batch_size', type=int, help="Batch size for inference", default=8)
    parser.add_argument('--offload_size', type=int, help="Number of offload experts in each layer", default=8)
    parser.add_argument('--max_new_tokens', type=int, help="Maximum number of new generation tokens", default=8)
    parser.add_argument('--is_baseline', action='store_true', help='Whether run baseline offload')
    parser.add_argument('--is_profile', action='store_true', help='Whether profile the run')
    parser.add_argument('--ipdb', action='store_true', help='Whether to use ipdb for debugging')

    # Parse the arguments
    args = parser.parse_args()

    return args