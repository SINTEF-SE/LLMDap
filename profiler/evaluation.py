from difflib import SequenceMatcher
import weave
import dspy
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.core.evaluation import SemanticSimilarityEvaluator
from llama_index.core.embeddings import resolve_embed_model
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings
import asyncio


def define_semantic_model(llm: str):

    Settings.embed_model = OllamaEmbedding(model_name=llm)
    
    evaluator = SemanticSimilarityEvaluator(
        # embed_model=embed_model,
        # similarity_mode=SimilarityMode.DEFAULT,  # Import error due to v0.11.0 llama-index-core ???
        similarity_threshold=0.6,
    )

    return evaluator


def similarity(a,b, a_func = max, b_func = sum, semantic=True):
    """ simple similarity measure (from short tests i guess its something like the ratio of letters in the longest that are in the shortest (probably with same order))
    its symmetric
    in any case, we should look more into better ways of diong this (llm? embedding? or other similarity measire https://stackoverflow.com/questions/17388213/find-the-similarity-metric-between-two-strings#17388505 )
    """
    # the max, sum combination is for predicting ALL the labels with ANY of the predictions
    # for predicting ANY label with ANY prediction (for disease), use max max.
    # a is pred, b i labels
    if type(b) == list:
        return b_func([similarity(a,l, a_func = a_func) for l in b])/len(b)
    if type(a) == list:
        return a_func([similarity(l, b) for l in a])


    if semantic:
        semantic_eval = define_semantic_model("llama3.1:8b")
        result = asyncio.run(semantic_eval.aevaluate(response=a.lower(), reference=b.lower()))        
        return result.score
    
    else:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()



@weave.op()
def score_ega_prediction(labels, filled_form, verbose = False):
    """ Score a prediction/label pair from the ega dataset, 
    using inclusion or similarity depending on the field"""
    if type(labels) == dspy.primitives.example.Example:
        labels = labels.label

    if verbose:
        print("PRED:\t", filled_form.date,       "\nLABELS:\t", labels["study_released"], "AND", labels["datasets_released"])
        print("")
        print("PRED:\t", filled_form.biosamples, "\nLABELS:\t", labels["num_samples"])
        print("")

        print("PRED:\t", filled_form.study_type, "\nLABELS:\t", labels["study_type"])
        print(similarity(filled_form.study_type, labels["study_type"]))
        print("")

        print("PRED:\t", filled_form.disease,    "\nLABELS:\t", labels["disease_by_keyword"], "AND", labels["disease_by_llm"])
        print([similarity(filled_form.disease, l) for l in [labels["disease_by_keyword"], labels["disease_by_llm"]]])
        print("")

        print("PRED:\t", filled_form.technology, "\nLABELS:\t", labels["technologies"])
        print([similarity(filled_form.technology, l) for l in labels["technologies"]])
        print("")

    score = 0

    # predict ANY
    if filled_form.date in [labels["study_released"], *labels["datasets_released"]]:
        score += 1

    # predict ANY
    if filled_form.biosamples in labels["num_samples"]:
        score += 1

    # predict ALL
    score += similarity(filled_form.study_type, labels["study_type"])
    # predict ANY
    score += similarity(filled_form.disease, [labels["disease_by_keyword"], labels["disease_by_llm"]], b_func=max)*2 #*2 since its devided ty len(b) (yes this should be done cleaner eventually)
    # predict ALL
    score += similarity(filled_form.technology, labels["technologies"])

    return score/5

@weave.op()
def score_general_prediction(labels, filled_form, verbose = False):
    """
    The evaluation function used for ARXPR dataset and others with same structure.
    This function evaluates a single paper (average is then calculated in main.py)
    One score is calculated per field, depending on field type.

    labels: dict (with field_name as keys) with lists. (see dataset_loader.py/get_simple_test for example)
    The list is of length 1 for now.
    filled_form: for now, also only have a single value per field

    returns a dict of score per field.
    This enables calculating mean score for each field
    """
    if type(labels) == dspy.primitives.example.Example:
        labels = labels.label

    scores = {}
    # iterate through fields
    for field in filled_form.__fields__:
        label = labels[field]

        # each paper only have labels for a subset of the fields.
        # we only calculate score for these
        if len(label):
            pred = getattr(filled_form, field)
            if pred is None:
                continue # this happens when loading (main.load_form), for unfilled fields
            #print(pred)
            #print(type(pred))
            #print(filled_form.schema()["properties"][field])

            # score is calculated according to type:
            if type(pred) is int:
                score = pred in [int(l) for l in label] #score is 1 if correct, 0 otherwise (independend of difference. I think this makes the most sense for out labels (i.e. if a year is wrong, the llm has not found it from the correct place)
            elif type(pred) is str:
                field_properties = filled_form.schema()["properties"][field]
                if (
                        "enum" in field_properties or # Literal / multiple choice
                        ("anyOf" in field_properties and "enum" in field_properties["anyOf"][0]) # Union(literal, None) (from loading)
                    ):
                    score = pred in label # 1 if correct, else 0
                else:
                    # for free text, similarity is calculated (could look into the way this is calculated)
                    score = max([similarity(l, pred) for l in label]) # score is from the best match with a label
            else:
                assert type(pred) == list, (pred, type(pred))
                # how to calculate this... f-score?
                raise NotImplementedError

            score = float(score) # convert bool to float
            scores[field] = score # add to dict
            if verbose:
                print("label:", label)
                print("pred:", pred)
                print("field score:", score)

            ## save (just for checking answer distribution - remove this)
            #with open("output_log.txt", "a") as file:
            #    file.write(f"{label}:{pred}\n")


    return scores
