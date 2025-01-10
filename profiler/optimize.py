import numpy as np
import torch
import weave
import outlines
import dspy

import dataset_loader
import ega_metadata_schema
import form_filling
import RAG
import evaluation


def prepare_fullform_trainset(length):
    """ Make a set of vector store retrievers to use as input in the optimizatin step """
    weave.init(project_name = "upcast_profiler")

    all_documents, all_labels = dataset_loader.load_ega_data(length)

    # load llm form filler
    pydantic_form = ega_metadata_schema.Single_Metadata_form

    chat_model = "llama3.1:8b"
    embed_model = "llama3.1:8b"

    trainset = []

    # iterate through documents
    for key in all_documents:
        paper_text = all_documents[key]
        paper_labels = all_labels[key]

        # make vectorstore
        vs = RAG.VectorStoreWeave(document=paper_text, chat_model=chat_model, embed_model=embed_model)
        
        # get relevant chunk(s) from vectorstore
        retriever = vs.build_retriever()
        # retriever = vs.build_query_engine()

        trainset.append(
                dspy.Example(retriever=retriever, label = paper_labels).with_inputs("retriever")
                )
    return trainset


class StringOutputModule(dspy.Module):
    def __init__(self, generator):
        self.generator = generator
        self.predictor = dspy.Predict()
    @weave.op()
    def forward(self, *args, **kwargs):
        print("(kw)args):", args, kwargs)
        answer = self.answer_generator(self.predictor, *args, **kwargs)


def optimize_fullform():
    """
    Optimize the process of filling out the full form (as opposed to individual fields, which might work better?).
    The retrievers are fed as input. This means the making of the VectorDB is not optimized, but done beforehand.
    The prompt used for retrieval should be optimizer (not yet implemented at the time of writing),
    as well as the generation part.
    """

    # generate training set
    trainset = prepare_fullform_trainset(6)
    valset = trainset[3:]
    trainset= trainset[:3]

    method = "mipro"

    if method == "mipro":
        form_filler_program = form_filling.SequentialFormFiller(device="cuda:0", verbose = False)

        # initialize prompt generator
        regex = outlines.fsm.json_schema.STRING
        llm_model = form_filler_program.llm_model
        regex_generator = outlines.generate.regex(llm_model, regex)
        from dspy_x_outlines import OutlinesHFModel
        lm = OutlinesHFModel(llm_model, regex_generator, max_tokens=500)


        # initialize optimizer
        optimizer = dspy.teleprompt.MIPROv2(
                prompt_model = lm,
                metric=evaluation.score_any_prediction,
                verbose = True,
                track_stats = True,
                )

        # initialize networks
        pydantic_form = ega_metadata_schema.Single_Metadata_form
        form_filler_program.set_pydantic_form(pydantic_form)

        # compile (i.e. optimize)
        eval_kwargs = dict(num_threads=16, display_progress=True, display_table=0)
        compiled_program = optimizer.compile(
                form_filler_program,
                trainset = trainset,
                valset = valset,
                num_batches=4,
                max_bootstrapped_demos=3,
                max_labeled_demos=5,
                seed=42,
                program_aware_proposer=False, # NOTE not sure how much this helps. It does not work with multiline argument definitions in __init__ (of sequentialformfiller, i think)
                eval_kwargs = eval_kwargs
                )
        print(compiled_program)
        print("compiled_program")

    if method == "copro":
        # initialize prompt generator
        form_filler = form_filling.SequentialFormFiller(device="cuda:2") # this is defined just to get the llm
        regex = outlines.fsm.json_schema.STRING
        llm_model = form_filler.llm_model
        regex_generator = outlines.generate.regex(llm_model, regex)
        from dspy_x_outlines import OutlinesHFModel
        lm = OutlinesHFModel(llm_model, regex_generator, max_tokens=500)


        # initialize optimizer
        optimizer = dspy.teleprompt.COPRO(
                prompt_model = lm,
                metric=evaluation.score_any_prediction,
                )

        # initialize networks
        pydantic_form = ega_metadata_schema.Single_Metadata_form
        program = form_filling.SequentialFormFiller(device = "cuda:3")
        program.set_pydantic_form(pydantic_form)

        # compile (i.e. optimize)
        eval_kwargs = dict(num_threads=16, display_progress=True, display_table=0)
        compiled_program = optimizer.compile(
                program,
                trainset = trainset,
                eval_kwargs = eval_kwargs
                )
        print(compiled_program)
        print("compiled_program")

    if method == "fewshot":
        # initialize optimizer
        #fewshot_optimizer = dspy.teleprompt.BootstrapFewShotWithRandomSearch(metric=evaluation.score_any_prediction,
        fewshot_optimizer = dspy.teleprompt.BootstrapFewShot(metric=evaluation.score_any_prediction,
                                                       max_bootstrapped_demos=4, 
                                                       max_labeled_demos=16,
                                                       max_rounds=1,
                                                       max_errors=5, 
                                                       )

        # initialize networks
        pydantic_form = ega_metadata_schema.Single_Metadata_form
        student = form_filling.SequentialFormFiller(device = "cuda:3")
        teacher = form_filling.SequentialFormFiller(device = "cuda:2")
        student.set_pydantic_form(pydantic_form)
        teacher.set_pydantic_form(pydantic_form)

        # compile (i.e. optimize)
        compiled_program = fewshot_optimizer.compile(
                student = student,
                teacher = teacher,
                trainset = trainset,
                #valset = valset,
                )
        print(compiled_program)
        print("compiled_program")

    # TODO should save or something?



if __name__ == "__main__":
    optimize_fullform()
