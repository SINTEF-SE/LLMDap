import dspy
import pprint
import torch

from context_shortening import RAG
from context_shortening.chunking import chunk_by_headeres_and_clean
from context_shortening import keybert_functions
from context_shortening import get_ontology_descriptions




class ContextShortener():
    """ General / template context shortener.
    The context shortener reduces the context from the entire paper, to whatever will be fed as context to the llm prompt. This could be a summary, keywords, certain chunks etc."""
    def __init__(self):
        pass
    def set_document(self,document):
        self.document = document
    def set_pydantic_form(self, pydantic_form):
        pass
    def __call__(self, **kwargs):
        raise NotImplementedError


class FullPaperShortener(ContextShortener):
    """ output the whole document """
    def __call__(self, **kwargs):
        return self.document


class CreateRetrievalPromptSignature(dspy.Signature):
    # dspy signature (prompt template) for sequential form filling (i.e. one field at a time), field-agnistic.
    """
    You are a RAG prompt engineer working on retrieving specific details for filling out a form, using scientific papers as the documents.
    Make a retrieval prompt for finding the field described below
    """ 

    answer_field_name = dspy.InputField()
    answer_field_description = dspy.InputField()
    answer_field_examples = dspy.InputField()

    answer = dspy.OutputField(desc="String to be used for retrieving the above info from the context")

class RAGShortener(ContextShortener):
    """ Retrieval implemented in the RAG.py file """

    def __init__(self, embed_model, pydantic_form, retriever_type, chunk_size, chunk_overlap, similarity_k, mmr_param):
        self.embed_model = embed_model
        self.pydantic_form = pydantic_form
        self.set_description_retrieval_prompt() # default : use description for retrieval
        self.retriever_type = retriever_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.similarity_k = similarity_k
        self.mmr_param = mmr_param

    def generate_retrieval_prompt_using_llm(self, dspy_lm):
        """ Generate a retrieval prompt for each field using a dspy llm"""
        dspy.settings.configure(lm=dspy_lm)

        predictor = dspy.Predict(signature=CreateRetrievalPromptSignature)

        # iterate through fields
        fields = self.pydantic_form.__fields__
        retrieval_prompts = {}
        for fieldname in fields:
            field = fields[fieldname]
            
            retrieval_prompts[fieldname] = predictor(
                    answer_field_name = fieldname,
                    answer_field_examples = str(field.examples),
                    answer_field_description = field.description
                    ).answer

        print("\nretrieval prompts generated (Read through them and make sure they make sense!):")
        pprint.pprint(retrieval_prompts)
        self.retrieval_prompts = retrieval_prompts
        print("")

    def set_description_retrieval_prompt(self):
        """Set the destricption of each field to be the retrieval prompt"""

        # iterate through fields
        fields = self.pydantic_form.__fields__
        self.retrieval_prompts = {fieldname : fields[fieldname].description for fieldname in fields}
        print("retrieval prompts generated:")
        pprint.pprint(self.retrieval_prompts)

    def set_document(self,document):
        # make vectorstore
        vs = RAG.VectorStoreWeave(document=document,
                                  embed_model=self.embed_model,
                                  chunk_size = self.chunk_size,
                                  chunk_overlap = self.chunk_overlap,
                                  similarity_k = self.similarity_k,
                                  mmr_param = self.mmr_param,
                                  )

        # # Simple retriever
        # if self.retriever_type == "simple":
        #     self.retriever = vs.build_retriever()

        # # Fancy retriever
        # if self.retriever_type == "fusion":
        #     self.retriever = vs.build_fusion_retriever()

        # # Retriver with metadata (temporary)
        # if self.retriever_type == "metadata":
        #     self.retriever = vs.build_query_engine()

        self.retriever = vs.build_query_engine()


    def __call__(self, **kwargs):
        context_nodes = self.retriever.retrieve(self.retrieval_prompts[kwargs["answer_field_name"]])

        string_with_all_contexts = "\n...\n".join([node.get_text() for  node in context_nodes])
        return string_with_all_contexts
        #context = context_nodes[0].get_text()

        # print("\nRetrieved Context:")
        # print(context, "\n")

        #print("\n\nRetrieved Context Metadata:")
        #print(context_nodes[0].get_content(metadata_mode='all'))

        # TODO: rank context nodes by similarity match with filtered ontologies for relevant fields

        #return context



class Retrieval(ContextShortener):
    """ Retrieval implemented using keybert """
    def __init__(self, 
            *, # force keyword arguments since there are so many
            chunk_info_to_compare,
            field_info_to_compare,
            include_choice_every,
            embedding_model_id,
            n_keywords, 
            top_k, 
            chunk_size,
            chunk_overlap,
            pydantic_form = None,
            mmr_param = 1,
            maxsum_factor = 1,
            keyphrase_range = (1,1),
            ):
        self.chunk_info_to_compare = chunk_info_to_compare
        self.field_info_to_compare = field_info_to_compare
        self.include_choice_every = include_choice_every
        self.embedding_model_id = embedding_model_id
        self.pydantic_form = pydantic_form
        self.n_keywords = n_keywords # number of keywords to extract from each chunk
        self.top_k = top_k # number of chunks to merge and return
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        assert mmr_param == 1 or maxsum_factor == 1
        self.mmr_param = mmr_param
        self.maxsum_factor = maxsum_factor
        self.keyphrase_range = keyphrase_range


        # define embedding model through these version numbers (dont want to handle the long names through args and main.py...
        self.emb_model = keybert_functions.get_embedding_model(embedding_model_id)
        self.kw_model = keybert_functions.get_kw_model(self.emb_model)
        
        self.descriptions = {}
        self.target_emb = {}

        if not pydantic_form is None:
            self.set_pydantic_form(pydantic_form)




    def set_pydantic_form(self, pydantic_form):
        self.set_target_embeddings(pydantic_form)

    def set_target_embeddings(self, pydantic_form):

        if self.field_info_to_compare == "description" or self.field_info_to_compare.startswith("onto-"):
            if self.target_emb:
                return # no need to update, as the order of literal values are not used
        self.descriptions = {}
        self.target_emb = {}

        if self.field_info_to_compare == "choices":
            fields = pydantic_form.__fields__
            for fieldname in fields:
                field = fields[fieldname]
                field_type = field.annotation
                literal_values = field_type.__args__
                field_literal_skip_number = min(self.include_choice_every, len(literal_values))
                literal_values = literal_values[field_literal_skip_number-1::field_literal_skip_number] # only include every n'th value (srtating on n-1)
                self.descriptions[fieldname] = literal_values
                self.target_emb[fieldname] = self.emb_model.encode(self.descriptions[fieldname])
        elif self.field_info_to_compare == "choice-list": # choices but in a list, single string
            fields = pydantic_form.__fields__
            for fieldname in fields:
                field = fields[fieldname]
                field_type = field.annotation
                literal_values = field_type.__args__
                field_literal_skip_number = min(self.include_choice_every, len(literal_values))
                literal_values = literal_values[field_literal_skip_number-1::field_literal_skip_number] # only include every n'th value (srtating on n-1)
                self.descriptions[fieldname] = [str(literal_values)] # make list of length one,  with a string of the whole list of values
                self.target_emb[fieldname] = self.emb_model.encode(self.descriptions[fieldname])
        elif self.field_info_to_compare == "description": #field description (should be same as rag setup)
            fields = pydantic_form.__fields__
            for fieldname in fields:
                self.descriptions[fieldname] = fields[fieldname].description
                self.target_emb[fieldname] = self.emb_model.encode(self.descriptions[fieldname])

        elif self.field_info_to_compare.startswith("onto-"): # from ontology
            mode = self.field_info_to_compare[5:]
            assert mode in ["label", "description", "both"]
            fields = pydantic_form.__fields__
            for fieldname in fields:
                self.descriptions[fieldname] = get_ontology_descriptions.get_subontology_for_field(mode, fieldname)
                self.target_emb[fieldname] = self.emb_model.encode(self.descriptions[fieldname])
        else:
            print(field_info_to_compare)
            raise ValueError

    def set_document(self, document):
        self.chunks = chunk_by_headeres_and_clean(document, chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap, verbose=False)
        self.chunks = [chunk.text for chunk in self.chunks]

        self.keywordss = []
        self.keyword_scoress = []
        self.keyword_embeddingss = []
        self.indices_with_keywords = []
        if self.chunk_info_to_compare == "keybert":
            for (i, chunk) in enumerate(self.chunks):
                keywords, scores = keybert_functions.get_keywords(
                        chunk, 
                        self.kw_model, 
                        # kwargs
                        keyphrase_ngram_range = self.keyphrase_range,
                        top_n=self.n_keywords,
                        use_maxsum = self.maxsum_factor>1,
                        use_mmr = self.mmr_param<1,
                        diversity = self.mmr_param,
                        nr_candidates = int(self.n_keywords * self.maxsum_factor),
                        )

                embs = self.emb_model.encode(keywords)
  
                if len(keywords)==0: # short chunks may have no keyword. Note that this require som extra index handling
                    print(f"no keyword chunk: ***{chunk}***")
                    continue
                self.keywordss.append(keywords)
                self.keyword_scoress.append(scores)
                self.keyword_embeddingss.append(embs)
                self.indices_with_keywords.append(i)

        elif self.chunk_info_to_compare == "direct":
            # for each chunk, act as if there is a single kw, embed the chunk and give score 1 (can simplify if keybert is redundant)
            for (i, chunk) in enumerate(self.chunks):

                if len(chunk)==0: # not sure if this can happen or not
                    continue

                keywords, scores = [chunk], [1.0]
                embs = self.emb_model.encode(keywords)
  
                self.keywordss.append(keywords)
                self.keyword_scoress.append(scores)
                self.keyword_embeddingss.append(embs)
                self.indices_with_keywords.append(i)
        else:
            raise ValueError

    def __call__(self, **kwargs):
        fieldname = kwargs["answer_field_name"]

        chunk_scores = []
        for kw_i, chunk_i in enumerate(self.indices_with_keywords): # keyword indices and chunk indices can be different

            similarity = keybert_functions.get_similarity_matrix(
                    self.keyword_embeddingss[kw_i], self.target_emb[fieldname]
                    )

            chunk_scores.append(
                    # store score and index
                    (self.calculate_chunk_relevance(similarity, self.keyword_scoress[kw_i]), chunk_i) # store index of corresponding chunk in tuple with the score
                    )

        chunk_scores = sorted(chunk_scores, key = lambda x: -x[0]) # sort in decreasing order, by score

        # print chunks
        #for score, index in chunk_scores:
        #    print(score, self.chunks[index])

        chosen_chunks = [self.chunks[chunk_scores[i][1]] for i in range(min(len(self.chunks), self.top_k))]
        return "\n...\n".join(chosen_chunks)

    def get_similarity_matrices(self, fieldname):
        """this function returns the similarity matrix per chunk - for usage with direct keywords-based classification (as opposed to for retrieval/reranking)"""
        similarities = []

        for kw_i, chunk_i in enumerate(self.indices_with_keywords): # keyword indices and chunk indices can be different

            similarity = keybert_functions.get_similarity_matrix(
                    self.keyword_embeddingss[kw_i], self.target_emb[fieldname]
                    )
            similarities.append((similarity, self.keyword_scoress[kw_i]))
        return similarities



    def calculate_chunk_relevance(self, similarity, keyword_scores):

        # Reduce ontology value dimension NOTE this can be done in many different ways!
        #similarity_per_kw = similarity.mean(dim=1)
        similarity_per_kw = similarity.max(dim=1).values #  can also get indices, to get max description
        # reduce to single score # NOTE this can also be changed.
        keyword_scores = torch.Tensor(keyword_scores)
        product = similarity_per_kw.inner(keyword_scores)
        return product

