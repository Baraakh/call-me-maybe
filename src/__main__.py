"""The main entry point of call_me_maybe project"""

import argparse
import sys
from argparse import Namespace

from .input_handling import InputFileError, get_funcs_def, get_prompts_entry


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description="call_me_maybe: A function-calling tool that turns "
        "natural-language prompts into structured function calls"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the input JSON file (function calling tests) "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to the output JSON file (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    funcs_def_path: str = args.functions_definition
    input_path: str = args.input
    # output_path: str = args.output

    try:
        funcs_def = get_funcs_def(funcs_def_path)
        prompts = get_prompts_entry(input_path)

    except InputFileError as e:
        sys.exit(f"{e}")


if __name__ == "__main__":
    main()
