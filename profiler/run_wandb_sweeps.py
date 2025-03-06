import wandb
import dspy
from argparse import Namespace
import yaml
from types import SimpleNamespace

from load_modules import load_modules
from run_modules import FormFillingIterator

# Load llm now instead of for each run
model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
#model_id = "TheBloke/Mistral-7B-v0.1-GPTQ"
dspy_model = dspy.HFModel(model = model_id)
#dspy_model = None

PRELOADED_DATASET = None


def add_defaults(parameters):
    """ given parameters for a run, add default values for all fields that are not included (from arguments.yaml) """

    with open("arguments.yaml", "r") as f:
        argument_template = yaml.safe_load(f)

    for argtype in argument_template:
        for argname in argument_template[argtype]:
            if not argname in parameters:
                parameters[argname] = {
                        "value" : argument_template[argtype][argname]["default"]
                        }
    return parameters



def sweep_single_run():
    # function for the runs in the wandb sweep (one run of FormFillingIterator for a specific set of parameters)
    global PRELOADED_DATASET
    wandb.init(project= "upcast_profiler")
    
    args = wandb.config


    prepared_kwargs = load_modules(args, preloaded_dspy_model = dspy_model, preloaded_dataset = PRELOADED_DATASET)
    args = args._items
    args.pop("_wandb")
    # use floats in argstring to load results in run_modules
    if args["maxsum_factor"]==1:
        args["maxsum_factor"]= 1.0
    if args["mmr_param"]==1:
        args["mmr_param"]= 1.0
    dataset_name = args["dataset"]
    #args["load"] = False # REMOVE THIS!!
    args = SimpleNamespace(**args)

    if "documents" in prepared_kwargs and "labels" in prepared_kwargs:
        PRELOADED_DATASET = (prepared_kwargs["documents"], prepared_kwargs["labels"])



    score = FormFillingIterator(args, **prepared_kwargs)()

    wandb.log(score)


# define sets of parameters to test

best_choice_params = { # directly use bext match (no generation)
        "ff_model" :{"value" : "best_choice"},
        "context_shortener" :{"value" : "retrieval"},
        "chunk_info_to_compare" : {"values": [
            "direct",
            #"keybert",
            ]},
        "field_info_to_compare": {"value":"choices"},
    }

fullpaper_params = { # best baseline, put the whole thing in gpt
        "ff_model" :{"value" : "4om"},
        "context_shortener" : {"value" :"full_paper"},
    }

gpt_sota= { # gpt using choices for retrieval
        "ff_model" :{"values" : ["4om"]},
        "field_info_to_compare" : {"values":[
            "choices",
            "choice-list",
            ]},
        "include_choice_every" : {"values" :[
            #1, # 25 values
            2, # 12 values
            #3, # 8 values
            4, # 6 values
            #5, # 5 values
            6, # 4 values
            #8, # 3 vales
            12, # 2 values
            24, # 1 value
            ]},
        "similarity_k" : {"values": [10]},
        }
gpt_rag_params = { # gpt using field description for retrieval
        "ff_model" :{"values" : [
            "4om",
            ]},
        "field_info_to_compare" : {"values":[
            "description",
            ]},
        "similarity_k" : {"values": [10]},
        }

llama_sota= { # llama using choices for retrieval
        "ff_model" :{"values" : ["hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"]},
        "field_info_to_compare" : {"values":[
            "choices",
            "choice-list",
            ]},
        "include_choice_every" : {"values" :[
            #1, # 25 values
            2, # 12 values
            #3, # 8 values
            4, # 6 values
            #5, # 5 values
            6, # 4 values
            #8, # 3 vales
            12, # 2 values
            24, # 1 value
            ]},
        "chunk_size" : {"value" : 300},
        "similarity_k" : {"value" : 4},
        "sampler" : {"value" : "multi"},
        "sampler_temp" : {"value" : 0.001},
        }
mistral_sota= { # mistral using choices for retrieval
        "ff_model" :{"values" : ["hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"]},
        "ff_model" :{"values" : ["TheBloke/Mistral-7B-v0.1-GPTQ"]},
        "field_info_to_compare" : {"values":[
            "choices",
            "choice-list",
            ]},
        "include_choice_every" : {"values" :[
            #1, # 25 values
            2, # 12 values
            #3, # 8 values
            4, # 6 values
            #5, # 5 values
            6, # 4 values
            #8, # 3 vales
            12, # 2 values
            24, # 1 value
            ]},
        "chunk_size" : {"value" : 300},
        "similarity_k" : {"value" : 4},
        "sampler" : {"value" : "multi"},
        "sampler_temp" : {"value" : 0.001},
        }

llama_rag= { # llama using rag for retrieval
        "ff_model" :{"values" : ["hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"]},
        "field_info_to_compare" : {"values":[
            "description",
            ]},
        "chunk_size" : {"values" : [
            300,
            ]},
        "similarity_k" : {"values" : [
            4,
            ]},
        }
mistral_rag = { # mistral using description for retrieval
        "ff_model" :{"values" : ["TheBloke/Mistral-7B-v0.1-GPTQ"]},
        "field_info_to_compare" : {"values":[
            "description",
            ]},
        "chunk_size" : {"values" : [
            300,
            ]},
        "similarity_k" : {"values" : [
            4,
            ]},
        }

gpt_ontorag_params = {
        "ff_model" :{"values" : [
            "4om",
            ]},
        "field_info_to_compare" : {"values":[
            "onto-description",
            "onto-label",
            "onto-both",
            ]},
        "similarity_k" : {"values": [10]},
        }


llama_ontorag= {
        "ff_model" :{"values" : ["hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"]},
        "field_info_to_compare" : {"values":[
            "onto-description",
            "onto-label",
            "onto-both",
            ]},
        "chunk_size" : {"values" : [
            300,
            ]},
        "similarity_k" : {"values" : [
            4,
            ]},
        }
mistral_ontorag = {
        "ff_model" :{"values" : ["TheBloke/Mistral-7B-v0.1-GPTQ"]},
        "field_info_to_compare" : {"values":[
            "onto-description",
            "onto-label",
            "onto-both",
            ]},
        "chunk_size" : {"values" : [
            300,
            ]},
        "similarity_k" : {"values" : [
            4,
            ]},
        }

def run_sweep(parameters, dataset_length=0, sweep_count=1, method="grid", dataset = "arxpr2", name = None, fields_length = 0, mode = "train", log=True):
    # perform the wandb sweep, trying out sets of parameters and running "sweep_run"
    parameters["dataset_length"] = {"value" : dataset_length}
    parameters["fields_length"] = {"value" : fields_length}
    parameters["mode"] = {"value" : mode}
    parameters["log"] = {"value" : log}
    if type(dataset) is str:
        parameters["dataset"] = {"value" : dataset}
        name = f"{name}_{dataset}_{sweep_count}_{dataset_length}"
    if type(dataset) is list:
        parameters["dataset"] = {"values" : dataset}
        name = f"{name}__{sweep_count}_{dataset_length}.{fields_length}"
    parameters = add_defaults(parameters)
    
    
    sweep_configuration = {
        "name": name,
        "method": method, # random, grid (every config) or bayesian
        "metric": {"goal": "maximize", "name": "total_score"},
        "parameters": parameters,
    }
    
    
    sweep_id = wandb.sweep(sweep=sweep_configuration, project="upcast_profiler")

    wandb.agent(sweep_id, function=sweep_single_run, count=sweep_count)
    #wandb.teardown()

def run_test_sweeps():
    # call run_sweep for each set of parameters
    fl = 100
    #run_sweep(best_choice_params, 
    #          fields_length = fl,
    #          sweep_count = 1,
    #          mode = "test",
    #          name="best_choice",
    #          )
    #run_sweep(gpt_rag_params, 
    #          fields_length = fl,
    #          sweep_count = 1,
    #          mode = "test",
    #          name="gpt_rag",
    #          )
    run_sweep(gpt_sota, 
              fields_length = fl,
              sweep_count = 10,
              mode = "test",
              name="gpt_sota",
              )
    #run_sweep(fullpaper_params, 
    #          fields_length = fl,
    #          sweep_count = 1,
    #          mode = "test",
    #          name="gpt_fullpaper",
    #          )
    #run_sweep(gpt_ontorag_params, 
    #          fields_length = fl,
    #          sweep_count = 1,
    #          mode = "test",
    #          name="gpt_onto_test",
    #          )

    run_sweep(llama_sota, 
              fields_length = fl,
              sweep_count = 10,
              mode = "test",
              name="llama_sota",
              )
    #run_sweep(llama_rag, 
    #          fields_length = fl,
    #          sweep_count = 1,
    #          mode = "test",
    #          name="llama_rag",
    #          )
    #run_sweep(llama_ontorag, 
    #          fields_length = fl,
    #          sweep_count = 1,
    #          mode = "test",
    #          name="llama_onto_test",
    #          )

    global dspy_model
    del dspy_model
    import time
    time.sleep(60*5)
    model_id = "TheBloke/Mistral-7B-v0.1-GPTQ"
    dspy_model = dspy.HFModel(model = model_id)

    run_sweep(mistral_sota, 
              fields_length = fl,
              sweep_count = 10,
              mode = "test",
              name="misrtal_rag",
              )
    #run_sweep(mistral_rag, 
    #          fields_length = fl,
    #          sweep_count = 1,
    #          mode = "test",
    #          name="mistral_rag",
    #          )
    #run_sweep(mistral_ontorag, 
    #          fields_length = fl,
    #          sweep_count = 1,
    #          mode = "test",
    #          name="mistral_onto_test",
    #          )



# sets of parameters used for tuing
llama_decodetune= {
        "ff_model" :{"values" : ["hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"]},
        "field_info_to_compare" : {"values":[
            #"description",
            "choices",
            #"choice-list",
            ]},
        "chunk_size" : {"value" : 300},
        "similarity_k" : {"value" : 4},
        "sampler_beams": {"values" : [3,5]},
        }
llama_decodetune2= {
        "ff_model" :{"values" : ["hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"]},
        "field_info_to_compare" : {"values":[
            #"description",
            "choices",
            #"choice-list",
            ]},
        "chunk_size" : {"value" : 300},
        "similarity_k" : {"value" : 4},
        "sampler" : {"values" : ["multi"]},
        "sampler_temp" : {"values" : [0.001,0.01,0.05]},#, 0.1, 0.2,0.3, 0.5, 0.7]},
        }
mist_decodetune= {
        "ff_model" :{"values" : ["TheBloke/Mistral-7B-v0.1-GPTQ"]},
        "field_info_to_compare" : {"values":[
            #"description",
            "choices",
            #"choice-list",
            ]},
        "chunk_size" : {"value" : 300},
        "similarity_k" : {"value" : 4},
        "sampler_beams": {"values" : [3,5]},
        }
mist_decodetune2= {
        "ff_model" :{"values" : ["TheBloke/Mistral-7B-v0.1-GPTQ"]},
        "field_info_to_compare" : {"values":[
            #"description",
            "choices",
            #"choice-list",
            ]},
        "chunk_size" : {"value" : 300},
        "similarity_k" : {"value" : 4},
        "sampler" : {"values" : ["multi"]},
        "sampler_temp" : {"values" : [0.001,0.01,0.05]},#, 0.1, 0.2,0.3, 0.5, 0.7]},
        }

deepseek_rag= { 
        "ff_model" :{"values" : ["jakiAJK/DeepSeek-R1-Distill-Llama-8B_GPTQ-int4"]},
        "field_info_to_compare" : {"values":[
            "description",
            ]},
        "chunk_size" : {"values" : [
            300,
            ]},
        "similarity_k" : {"values" : [
            4,
            ]},
        }


if __name__ == "__main__":
    run_test_sweeps()
    quit()
    fl = 2
    run_sweep(deepseek_rag, 
              fields_length = fl,
              sweep_count = 1,
              #mode = "test",
              name="ds_rag",
              )
    quit()

    fl = 50
    run_sweep(gpt_ontorag_params, 
              fields_length = fl,
              sweep_count = 3,
              #mode = "test",
              name="gpt_onto",
              )
    run_sweep(llama_ontorag, 
              fields_length = fl,
              sweep_count = 3,
              #mode = "test",
              name="llama_onto",
              )
    run_sweep(mistral_ontorag, 
              fields_length = fl,
              sweep_count = 3,
              #mode = "test",
              name="mistral_onto",
              )
