from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from owlready2 import get_ontology


def get_subontology(
            mode, # label or description
            top_level_node_iri = "http://purl.obolibrary.org/obo/OBI_0000070", # node labeled "assay"
            ):
    """ get the description and/or labels of every node that is a descendant of the "assay" node (or another specified, on the efo ontology"""

    # Step 1: Load the ontology using OWLReady2 and extract concept labels
    ontology_path = "../ontologies/efo.owl"  # Path to your OWL file
    ontology = get_ontology(ontology_path).load()
    
    # Step 2: Prune ontology and get descriptions
    # Extract labels for each concept in the ontology
    top_level_node = ontology.search_one(iri = top_level_node_iri)
    descendants = top_level_node.descendants()
    ontology_descriptions = []
    for node in descendants:
        node_descriptions = node.IAO_0000115

        if len(node_descriptions) and mode in ["description", "both"]:
            ontology_descriptions.append(node_descriptions[-1]) # in 4 cases for the assay subontology, there are 2 or 3 descriptions. In all cases the last one is best (they are very similar)
        if not (len(node_descriptions) and mode == "description"):
            ontology_descriptions.append(node.label[0]) # there are 2 labels on some cases, (the if editors prefered label is different i think), but not among these ones without description
    #print(f"Ontology descriptions tail: {ontology_descriptions[-4:]}")
    return ontology_descriptions


def get_kw_model(model=None): return KeyBERT(model)
def get_keywords(text, kw_model, **kwargs): 
    keywords_and_scores = kw_model.extract_keywords(text, **kwargs)
    keywords = [x[0] for x in keywords_and_scores]
    scores = [x[1] for x in keywords_and_scores]
    return keywords, scores
#def get_embedding_model(embedding_model_id = 'all-MiniLM-L6-v2'): return SentenceTransformer(embedding_model_id)#, device = "cuda:1")
def get_embedding_model(embedding_model_id = 'all-MiniLM-L6-v2'): return SentenceTransformer(embedding_model_id, device = "cuda:1")
def get_similarity_matrix(kw_embeddings, target_embeddings):  return util.cos_sim(kw_embeddings, target_embeddings)



if __name__ == "__main__":

    # chessecake test / example
    text = "I like cheese. I love cake."
    targets = ["cheese", "cake", "cheesecake"] # to speed up testing

    kw_model = get_kw_model()
    kws, scores = get_keywords(text, kw_model, top_n =2)
                                  #use_maxsum=True, nr_candidates=4)
    print(kws, scores) # cheese and cake

    #kws, scores = get_keywords(text, kw_model, top_n =2, keyphrase_ngram_range=(2,2))
    #print(kws, scores) #  cheese love and love cake

    emb_model = get_embedding_model()
    kw_emb = emb_model.encode(kws)
    target_emb = emb_model.encode(targets)

    # using keybert
    similarity = get_similarity_matrix(kw_emb, target_emb)
    print(similarity) # cheese and cake are most relevant targets

    # no keybert - similarity between original text and targets
    text_emb = emb_model.encode([text])
    similarity = get_similarity_matrix(text_emb, target_emb)
    print(similarity) # cheescake is most relevant target (!)

