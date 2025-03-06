





# UPCAST profiling

This is the upcast profiling tool, that aims to profile biomedical datasets based on the papers/documents presenting the datasets.



## How to run:

setup environment:
```
conda create --name upcast python==3.10.14
conda activate upcast
pip install -r profiler/requirements.txt
pip install --upgrade werkzeug==2.3.8
```
(werkzeug must be downgraded from the required version, causing a warning)

run on a simple test case:
```
cd profiler
python main.py --dataset simple_test --no-log
```

Look through `introduction.ipynb` for an introduction to how the different parts work.


For running inference - i.e. for analysing one or several papers for the purpose of the output instead of evaluation with labels,
use run_inference.call_inference()
## Quick file overview

Folders:
- `data`: for preparing the datasets - downloading papers and metadata + collecting metadata into one json file
- `ontologies`: some scripts with experiments for exploring the ontologies, + some ontology files.
- `llm_ui`: user interface app made early in the project (i.e. currently outdated, but may eventually be connected to the rest of the pipeline again)
- `profiler`: the llm pipeline
- `profile/metadata_schemas`: pydantic schemas describing the format of the output of the pipeline
- `profiler/context_shortening/`: for reducing the size of the context fed to the llm (meant as a generalisation of the retrieval concept in rag - can be regular retrieval, keybert based retrieval, just feeding the full paper, or using llm to summarice)
- `profiler/form_filling/`: for the generation part, with structured output handled in different ways according to the llm and the flied. Returns an object following the correct pydantic schema for the specified dataset.


Some file descriptions:
#####- `profiler/main.py`: assemples the pipeline and runs it according to the arguments.
- `profiler/load_modules.py`: assemples the pipeline according to the arguments
- `profiler/run_modules.py`: runs the pipeline from load_modules.
- `profiler/dataset_loader.py` for dataset loading
- `profiler/evaluation.py` to calculate the score
- `profiler/arguments.yaml`: defines all the arguments for used in main.py and run_wandb_sweeps.py. Run `python main.py --help` to list them.
- `profiler/run_wandb_sweeps.py`: uses sweeps from weights and biases for hyper parameter tuning and for running all the final tests on the test set. Calls `load_/run_modules.py`.
- `profiler/context_shortening/chunker_test.ipynb`: notebook with some testing of the chunker
- `profiler/optimize`: dspy prompt optimization - not updated in a while/currently not in use

Some missing files (in .gitignore)
- `profiler/openai_key.py`: the file with the openai api key
- `ontologies/efo.owl`: the ontology used for the keybert-retriever for study type - the file is too large for github.










## Experimental design

### Dataset creation

#### Retrieving data
(this is done in the data/fetch_data.py file)
- Querying EuropePMC, we found all the ID of all the papers referred to by arrayExpress
- For each ID, we attempt downloading the full text XML of the paper from PMC, as well as the metadata from ArrayExpress.
- This results in 14860 metadata json files, for one dataset each, and the XMLs for their papers.
- These json files are quite big, include both information on the whole dataset and each sample, and which fields are in the json vary from dataset to dataset. Example: https://www.ebi.ac.uk/biostudies/files/E-MTAB-8097/E-MTAB-8097.json

#### Consolidating to a usable dataset
- (in data/make_somple_json.count_fields():) Iterate through the jsons, finding all unique fields, count the number of times they appear, note some examples and if they are ontology terms, and save this info
- Next, I disregarded any fields with less than 2000 appearances, and manually chose interesting candidates among the other, choosing only those 
  - with valuable information about the dataset (disregarding e.g. sample-specific fields)
  - that can likely be found in the paper (disregarding e.g. file names)
  - where there value is somewhat predictable (disregarding e.g. titles and descriptions).
  - NOTE: the majority of the fields were removed, leaving 23 candidates
  - NOTE: this step should perhaps be redone with more well-defined criteria?
- The 23 candidates are analysed in [this notebook](data/visualize_fields.ipynb)
  - Only fields with reasonably distributed values were selected (disregarding e.g. developmental stage and strain where most of the values are unique, and software, where most values are the same) 
  - This resulted in the fields found in the metadata schema.
  - Fields where all/most occurances have a small set of values, were set as Literal fields (i.e. multiple choice/categorization) (with values like "other" for values outside the selected allowable ones), the others as free text (still limited by length to avoid full sentece answers etc).
- (in the remainder of data/make_somple_json) The jsons were then merged into one json with the selected fields. In cases where a paper is connected to several datasets, all the values are used.

#### Further restrictions upon loading the data
- On the profiler pipeline, for now, all fields with more than one value is ignored.

### Restricted prediction
- When generating answers for the field, we use restricted output.
  - For generation using llama3 or other open source models, Outlines is used to ensure the restrictions are followed. It works by setting the probabilities of disallowed tokens to 0 before sampling the generated tokens.
  This means, for integer fields the output will always be integer, for Literal fields it will always be one of the listed allowed answers, and for free-text it will follow the maximum length (which, when done this way instead of using max tokens, in most cases avoids halv sentences).
  - For generation through openai, we use their structured_output option, which works in the same way (masking disallowed tokens) but they have not implemented as many restrictions - e.g. they do not allow restricted field length, so this is ignored.

### Evaluation
- Each field is evaluated in the following manner depending on the type:
  - Integer field: 1 if correct, 0 if not (independent of how far it was from correct)
  - Literal field: 1 if correct, 0 if not (with allowed answers based ontology we could introduce more levels, for values closeby in the tree)
  - string fields: Similarity based on characters
- Literal and integer reported as accuracy, strings as similarity.
