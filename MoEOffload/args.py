import argparse

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description="MoEOffload Arguments")

    # Add arguments
    parser.add_argument('--data_name', type=str, help='Dataset name', default='xsum')
    parser.add_argument('--model_path', type=str, help='Model state path')
    parser.add_argument('--batch_size', type=int, help="Batch size for inference", default=8)
    parser.add_argument('--offload_size', type=int, help="Number of offload experts in each layer", default=8)
    parser.add_argument('--schedule_size', type=int, help="The total batch size for scheduling", default=128)
    parser.add_argument('--seed', type=int, help="Random seed for shuffling dataset", default=1234)
    parser.add_argument('--max_new_tokens', type=int, help="Maximum number of new generation tokens", default=8)
    parser.add_argument('--top_n', type=int, help='Select top n output as predict pattern', default=0)
    parser.add_argument('--num_batches', type=int, help='Run num batches data', default=8)
    parser.add_argument('--is_baseline', action='store_true', help='Whether run baseline offload')
    parser.add_argument('--is_profile', action='store_true', help='Whether profile the run')
    parser.add_argument('--ipdb', action='store_true', help='Whether to use ipdb for debugging')
    parser.add_argument('--is_predict', action='store_true', help='Whether run predictor')
    parser.add_argument('--in_order', action='store_true', help='Whether to schedule batch in order')

    # Parse the arguments
    args = parser.parse_args()

    return args