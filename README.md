





# LLMDap (LLM-based Data Profiling)

LLMDap is a generic, domain-agnostic solution for automated metadata generation. It uses Large Language Models (LLM) and Retrieval-Augmented Generation (RAG) to produce consistent, high-quality metadata. This metadata aligns with domain ontologies, improving dataset discoverability.


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

## Publications
Further information can be found in the following publications:
- Shanshan Jiang, Sondre Sørbø, Phil Tinn, Shang Ferheng Karim and Dumitru Roman (2025). LLMDap: LLM-based Data Profiling and Sharing. Third Data Economy Workshop, 2025. https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/DEC/DEC25_5.pdf
- Phil Tinn, Sondre Sørbø, Shanshan Jiang, Konstantinos Voutetakis, Sotiris Moudouris Giounis, Eleftherios Pilalis, Olga Papadodima, Dumitru Roman (2025). Pre-Meta: Priors-augmented Retrieval for LLM-based Metadata Generation. Bioinformatics, 2025; https://doi.org/10.1093/bioinformatics/btaf519

