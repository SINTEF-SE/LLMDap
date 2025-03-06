import argparse
import yaml
from types import SimpleNamespace

from load_modules import load_modules
from run_modules import FormFillingIterator




def add_defaults(parameters):
    """ given parameters for a run, add default values for all fields that are not included (from arguments.yaml) """
    with open("arguments.yaml", "r") as f:
        argument_template = yaml.safe_load(f)

    for argtype in argument_template:
        for argname in argument_template[argtype]:
            if not argname in parameters:
                parameters[argname] = argument_template[argtype][argname]["default"]
    return parameters

def call_inference(
        schema,
        parsed_paper_text = None,
        raw_xml_paper_text = None,
        paper_path = None,
        paper_url = None,
        **kwargs):
    """
    Fill out the schema for one or several papers.

    Inputs:

    schema: pydantic form that the pipeline should filll out

    include ONE of the following four arguments:
        parsed_paper_text: paper text ready for reading
        raw_xml_paper_text: paper text in xml format (typically starting with "<?xml version="1.0" encoding="UTF-8"?>...)"
        paper_path: local path to paper file in XML format
        paper_url: url to paper file in XML format
    This argument should be either a string, a list of strings, or a dict of strings

    kwargs: Any argument from the arguments.yaml file (e.g. llm, retrievl method and parameters like chunk length)


    output:
    dictionary with the filled form and used contexts for each paper.

    """

    # prepare arguments
    parameters = add_defaults(kwargs)
    args=SimpleNamespace(**parameters)
    args.load=False
    args.save=False
    args.schema = schema

    # load stuff
    prepared_kwargs = load_modules(args)
    ff_iterator = FormFillingIterator(args, **prepared_kwargs)

    # make the argument into dictionary
    paper_argument = parsed_paper_text or raw_xml_paper_text or paper_path or paper_url
    if type(paper_argument) is str:
        paper_argument = {0: paper_argument}
    elif type(paper_argument) is list:
        paper_argument = {i:p for i,p in enumerate(paper_argument)}
    elif type(paper_argument) is dict:
        pass
    else:
        raise ValueError
    
    outputs = {}
    for key in paper_argument:
        
        # parse/load/fetch argument (to a string paper ready for the llm)
        if parsed_paper_text:
            paper_text = paper_argument[key]
        elif raw_xml_paper_text:
            import dataset_loader
            paper_text = dataset_loader.parse_raw_xml_string(paper_argument[key])
        elif paper_path:
            import dataset_loader
            paper_text = dataset_loader.load_paper_text_from_file_path(paper_argument[key])
        elif paper_url:
            import dataset_loader
            paper_text = dataset_loader.load_paper_text_from_url(paper_argument[key])
        else:
            raise ValueError

        # fill form
        outputs[key] =ff_iterator.fill_single_form(key="", paper_text=paper_text, pydantic_form=schema, return_dict_with_context=True)

    return outputs



if __name__ == "__main__":

    
    path = "/mnt/data/upcast/data/all_xmls/12093373_ascii_pmcoa.xml"
    path2= "/mnt/data/upcast/data/all_xmls/12095422_ascii_pmcoa.xml"
    import dataset_loader

    parsed_xml_paper_text = dataset_loader.load_paper_text_from_file_path(path)
    with open(path, "r") as f:
        raw_xml_paper_text = f.read()

    paper_path = path
    paper_url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/12093373/ascii"
    paper_url2= "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/12095422/ascii"


    from metadata_schemas.arxpr2_schema import Metadata_form as schema

    output = call_inference(
            schema,
            #
            # choose one to try out:
            #
            #parsed_paper_text = parsed_xml_paper_text,
            #raw_xml_paper_text = raw_xml_paper_text,
            #paper_path = paper_path,
            #paper_path = [paper_path, path2],
            #paper_url = paper_url,
            paper_url = {"paper1": paper_url, "paper2":paper_url2},
            #
            similarity_k = 5,
            field_info_to_compare = "choices",
            )

    print("output:")
    import pprint
    pprint.pprint(output)

