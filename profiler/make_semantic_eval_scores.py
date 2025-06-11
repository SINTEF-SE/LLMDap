
from evaluation import similarity, define_semantic_model
import json

with open("all_results/arxpr3_scores.json") as f:
    data = json.load(f)

choice_data = data["choice_log"]

def strjoin(a,b): return a+"::"+b
def strjoin(a,b): return (a,b)

semantic_score_data = {}

emb_models = ["all-minilm:l6-v2", "llama3.1:8b"]

similarity_db = {e:{} for e in emb_models}
def get_similarity(a,b,emb_model, semantic_model):
    global similarity_db
    if not strjoin(a,b) in similarity_db[emb_model]:
        sim_score= similarity(a,b, semantic=True, semantic_model=semantic_model)
        similarity_db[emb_model][strjoin(a,b)] = sim_score
        similarity_db[emb_model][strjoin(b,a)] = sim_score
    return similarity_db[emb_model][strjoin(a,b)]


for emb_model in emb_models:
    semantic_score_data[emb_model] = {}
    semantic_model = define_semantic_model(emb_model)
    print("emb model", emb_model)
    print("total_index_length:", len(choice_data))
    for run_key in choice_data:
        semantic_score_data[emb_model][run_key] = {}
        for field in choice_data[run_key]:
            semantic_score_data[emb_model][run_key][field] = [get_similarity(pair[0],pair[1], emb_model, semantic_model) for pair in choice_data[run_key][field]]
        print(len(semantic_score_data[emb_model]))
with open(f"all_results/arxpr3_semantic_scores.json", "w") as f:
    json.dump(semantic_score_data, f)

