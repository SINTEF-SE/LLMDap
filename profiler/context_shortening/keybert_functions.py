from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util

import torch

def get_kw_model(model=None): return KeyBERT(model)
def get_keywords(text, kw_model, **kwargs): 
    keywords_and_scores = kw_model.extract_keywords(text, **kwargs)
    keywords = [x[0] for x in keywords_and_scores]
    scores = [x[1] for x in keywords_and_scores]
    return keywords, scores
def get_embedding_model(embedding_model_id = 'all-MiniLM-L6-v2'): return SentenceTransformer(embedding_model_id, device = "cuda:1" if torch.cuda.device_count()>1 else "cuda:0")
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

