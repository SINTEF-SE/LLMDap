import weave
import outlines
import dspy
import openai

import dataset_loader
import metadata_schemas 
import form_filling
import evaluation
import context_shortening

import nltk
nltk.download('averaged_perceptron_tagger_eng')


def set_openai_api_key():
    from openai_key import API_KEY
    openai.api_key = API_KEY


def remove_non_single_fields(labels):
    return [field for field in labels.keys() if len(labels[field]) != 1]

def remove_empty_fields(labels):
    return [field for field in labels.keys() if len(labels[field]) == 0]


@weave.op() # log args
def load_modules(args, preloaded_dspy_model = None, preloaded_dataset = None):
    """
    prepare arguments, then call fill_out_forms 
    preloaded_dspy_model can be inputted to avoid loading it in memory several times
    """

    # log arguments
    if args.log:
        weave.init(project_name = "upcast_profiler",
                   )

    # load llm
    model_is_openai = False
    use_best_choice_generator = False
    if args.ff_model == "4om": # openai model
        model_id = "gpt-4o-mini"
        model_is_openai = True
        set_openai_api_key()
    elif args.ff_model == "4o": # openai model
        model_id = "gpt-4o"
        model_is_openai = True
        set_openai_api_key()
    elif args.ff_model == "best_choice":
        use_best_choice_generator = True
    elif args.ff_model == "None": # do not load any model (used for retrieval evaluation)
        model_id = ""
        model_is_openai = True
    else: # huggingface model, with outlines
        # load HF llm through dspy
        if preloaded_dspy_model is None:
            model_kwargs = {}
            if args.ff_model == "llama3.1I-8b-q4":
                model_id = "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4"
            elif args.ff_model == "biolm":
                model_id = "aaditya/Llama3-OpenBioLLM-8B"
            elif args.ff_model == "ministral_gguf":
                model_id = "bartowski/Ministral-8B-Instruct-2410-GGUF"
                model_kwargs = {"gguf_file" : "Ministral-8B-Instruct-2410-Q4_K_M.gguf"}
            else:
                model_id = args.ff_model
            try:
                    dspy_model = dspy.HFModel(model = model_id, hf_device_map = "cuda:2", model_kwargs = model_kwargs)
            except RuntimeError:
                dspy_model = dspy.HFModel(model = model_id, hf_device_map = "cuda:0")
        else:
            dspy_model = preloaded_dspy_model
        hf_model = dspy_model.model
        hf_tokenizer = dspy_model.tokenizer

        # set some dspy model options
        #dspy_model.kwargs["max_tokens"]=args.max_tokens
        dspy_model.drop_prompt_from_output = True

        # define outlines llm and sampler
        outlines_llm = outlines.models.Transformers(model=hf_model, tokenizer=hf_tokenizer)
        if args.sampler == "greedy":
            outlines_sampler = outlines.samplers.GreedySampler()
        elif args.sampler == "beam":
            outlines_sampler = outlines.samplers.BeamSearchSampler(beams = args.sampler_beams)
        elif args.sampler == "multi":
            outlines_sampler = outlines.samplers.MultinomialSampler(
                    top_k=args.sampler_top_k,
                    top_p=args.sampler_top_k,
                    temperature=args.sampler_temp,
                    )
        else:
            raise ValueError




    if args.dataset == "arxpr2" and args.dataset_shuffle == "r":
        # do dynamic reloading+shuffling
        length = args.dataset_literal_length
        form_generator = metadata_schemas.get_shuffled_arxpr2(length = length)
        document_generator = dataset_loader.Arxpr_generator(version = "2_25", mode=args.mode)
        dataset_kwargs = dict(
                form_generator = form_generator,
                document_generator = document_generator,
                )
        pydantic_form = form_generator()
    elif args.dataset == "study_type" and args.dataset_shuffle == "r":
        # do dynamic reloading+shuffling
        length = args.dataset_literal_length
        form_generator = metadata_schemas.get_shuffled_arxpr2(length = length, only_shuffle_type = True)
        document_generator = dataset_loader.Studytype_generator(version = "2_25", mode=args.mode)
        dataset_kwargs = dict(
                form_generator = form_generator,
                document_generator = document_generator,
                )
        pydantic_form = form_generator(0)
    else:
        # load up front
        loader_kwargs = {"max_amount": args.dataset_length}
        if args.dataset == "arxpr":
            loader = dataset_loader.load_arxpr_data
            pydantic_form = metadata_schemas.arxpr_schema 
        elif args.dataset == "arxpr2":
            loader_kwargs["version"] = "2_25" # loaded dataset always 25, only pydantic form depends on literal_length and shuffling
            if args.dataset_shuffle == "s": #preshuffled
                raise NotImplementedError # just shiffle here instead
                #pydantic_form = metadata_schemas.arxpr2s_schemas[str(args.dataset_literal_length)] # TODO shuffle
            elif args.dataset_shuffle == "n": # not shuffled
                pydantic_form = metadata_schemas.arxpr2_schemas[str(args.dataset_literal_length)]
            elif args.dataset_shuffle.isdecimal(): #preshuffled
                length = int(args.dataset_literal_length)
                form_generator = metadata_schemas.get_shuffled_arxpr2(length = length)
                pydantic_form = form_generator(seed=args.dataset_shuffle)
            else:
                print(type(args.dataset_shuffle), args.dataset_shuffle)
                raise ValueError

            loader_kwargs["mode"] = args.mode #train or test
            loader = dataset_loader.load_arxpr_data
        elif args.dataset == "study_type":
            loader = dataset_loader.load_study_type_data
            pydantic_form = metadata_schemas.study_type_schema 
        elif args.dataset == "ega":
            loader = dataset_loader.load_ega_data
            pydantic_form = metadata_schemas.ega_schema
        elif args.dataset == "nhrf":
            loader = dataset_loader.load_nhrf_examples
            pydantic_form = metadata_schemas.nhrf_qa_schema
        elif args.dataset == "nhrf2":
            loader = dataset_loader.load_nhrf_examples2
            pydantic_form = metadata_schemas.nhrf_schema
        elif args.dataset == "nhrf3":
            loader = dataset_loader.load_nhrf_examples3
            pydantic_form = metadata_schemas.nhrf_qa_schema_2
        elif args.dataset == "simple_test":
            loader = dataset_loader.get_simple_test
            pydantic_form = metadata_schemas.arxpr_schema 
        else:
            raise ValueError
        if preloaded_dataset is None:

            all_documents, all_labels = loader(**loader_kwargs)
        else:
            all_documents, all_labels = preloaded_dataset

        dataset_kwargs = dict(
                documents = all_documents,
                labels = all_labels,
                )


    # set context_shortener
    if args.context_shortener == "rag":
        context_shortener = context_shortening.RAGShortener(
                embed_model = args.embedding_model,
                pydantic_form = pydantic_form,
                retriever_type = args.retriever_type,
                chunk_size = args.chunk_size,
                chunk_overlap = args.chunk_overlap,
                similarity_k = args.similarity_k,
                mmr_param = args.mmr_param,
                )
    elif args.context_shortener == "rerank":
        if model_is_openai:
            raise NotImplementedError
        context_shortener = context_shortening.Rerank(hf_model, hf_tokenizer)
    elif args.context_shortener == "reduce":
        if model_is_openai:
            raise NotImplementedError
        context_shortener = context_shortening.Reduce(
                hf_model,
                hf_tokenizer,
                temperature = args.reduce_temperature,
                chunk_size = args.chunk_size,
                chunk_overlap = args.chunk_overlap,
                max_tokens = args.reduce_max_tokens,
                )
    elif args.context_shortener == "full_paper":
        context_shortener = context_shortening.FullPaperShortener()
    elif args.context_shortener == "retrieval":
        if not args.dataset in ["study_type", "arxpr2"] and not args.field_info_to_compare=="description":
            raise ValueError

        context_shortener = context_shortening.Retrieval(
                chunk_info_to_compare = args.chunk_info_to_compare,
                field_info_to_compare = args.field_info_to_compare,
                include_choice_every = args.include_choice_every,
                embedding_model_id = args.embedding_model,
                pydantic_form = pydantic_form if args.include_choice_every==1 or args.dataset_shuffle!="r" else None, # if we are to reshufle and pick only some values, we do not specify the pydantic form here.
                n_keywords = args.n_keywords,
                top_k = args.similarity_k,
                chunk_size = args.chunk_size,
                chunk_overlap = args.chunk_overlap,
                mmr_param = args.mmr_param,
                maxsum_factor = args.maxsum_factor,
                keyphrase_range = (args.keyphrase_min, args.keyphrase_min + args.keyphrase_range_diff),
                )
    else:
        print(args.context_shortener)
        raise ValueError


    if model_is_openai:
        if args.context_shortener=="full_paper":
            form_filler = form_filling.OpenAIFormFiller(
                    model_id=model_id,
                    pydantic_form = pydantic_form,
                    listify_form = args.listed_output,
                    max_tokens = args.openai_ff_max_tokens,
                    verbose=False)#True)
        elif args.context_shortener in ["rag", "retrieval"]:
            form_filler = form_filling.OpenAISequentialFormFiller(
                    model_id=model_id,
                    pydantic_form = pydantic_form,
                    listify_form = args.listed_output,
                    max_tokens = args.openai_ff_max_tokens,
                    verbose=False)
        else:
            raise NotImplementedError
    elif use_best_choice_generator:
        form_filler = form_filling.DirectKeywordSimilarityFiller(
                pydantic_form=pydantic_form,
                listify_form=args.listed_output,
                verbose=False)

    else:
        # load llm form filler
        form_filler = form_filling.SequentialFormFiller(
                outlines_llm,
                outlines_sampler,
                pydantic_form=pydantic_form,
                listify_form=args.listed_output,
                answer_in_quotes=args.answer_in_quotes,
                max_tokens = args.outlines_ff_max_tokens)

    # remove fields?
    if args.remove_fields == "None":
        remove_fields = lambda x:[]
    elif args.remove_fields == "empty":
        remove_fields = remove_empty_fields
    elif args.remove_fields == "non-single":
        remove_fields = remove_non_single_fields
    else:
        print(args.remove_fields)
        raise ValueError



    prepared_kwargs = dict(
            context_shortener = context_shortener,
            form_filler = form_filler,
            evaluation_fnc=evaluation.score_general_prediction,
            remove_fields = remove_fields,
            **dataset_kwargs
            )
    return prepared_kwargs


