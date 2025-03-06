import argparse

from load_modules import load_modules
from run_modules import FormFillingIterator


def parse_terminal_arguments():

    import yaml
    with open("arguments.yaml", "r") as f:
        argument_template = yaml.safe_load(f)

    parser = argparse.ArgumentParser()

    for argtype in argument_template:
        for argname in argument_template[argtype]:
            arg_info = argument_template[argtype][argname]
            if type(arg_info["default"]) == bool:
                arg_info["action"] = argparse.BooleanOptionalAction
            else:
                arg_info["type"] = type(arg_info["default"])
            if not "help" in arg_info:
                arg_info["help"] = ""
            arg_info["help"] = "( in " + argtype + ") : " + arg_info["help"]
            parser.add_argument("--"+argname, **arg_info)
    args = parser.parse_args()
    return args

if __name__ == "__main__":


    args = parse_terminal_arguments()
    prepared_kwargs = load_modules(args)
    FormFillingIterator(args, **prepared_kwargs)()
