import numpy as np
import torch
import weave
import pprint
import pydantic
from typing import Optional
import time
import json
import os
import openai

import form_filling
import context_shortening

def make_optional_model(model: pydantic.BaseModel) -> pydantic.BaseModel:
    """
    Make a new pydantic model where all the fields are optional.
    Node that string contraints, descriptions and examples are not kept
    (this is not needed for evaluation)
    """
    fields = {name: (Optional[field.annotation], None) for name, field in model.__fields__.items()}
    optional_model = pydantic.create_model(f'{model.__name__}Optional', **fields)
    return optional_model


def load_form(key, argstring, pydantic_form):
    if not argstring: # something wrong
        raise ValueError
    try:
        with open("all_results/"+key+".json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print("------ file not found")
        return None # file does not exist
    except json.decoder.JSONDecodeError as e:
        print("------ file corrupted(?)")
        print(e)
        return None # corrupted - loading fail
    try:
        data = data[argstring]
    except KeyError:
        return None # file does not contain a run with these arguments - loading fails
    if type(data) == str and data == "skipped":
        return "skipped"
    optional_form = make_optional_model(pydantic_form)
    print("------ load successfull")
    return optional_form(**data)

def save_form(key, argstring, form_dict):
    # to save form from each query (using argstring)
    if not argstring: # something wrong
        raise ValueError
    try:
        with open("all_results/"+key+".json") as f:
            data = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        data= {}
    data[argstring] = form_dict
    os.makedirs("all_results", exist_ok = True)
    with open("all_results/"+key+".json", "w") as f:
        json.dump(data, f)


def save_score(argstring, scores, index_log, choice_log, dataset):
    if not argstring: # something wrong
        raise ValueError
    try:
        with open(f"all_results/{dataset}_scores.json") as f:
            data = json.load(f)
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        data= {"scores":{}, "index_logs":{}, "choice_logs":choice_log}
    data["scores"][argstring] = scores
    data["index_logs"][argstring] = index_log
    if not "choice_log" in data:
        data["choice_log"] = {}
    data["choice_log"][argstring] = choice_log
    os.makedirs("all_results", exist_ok = True)
    with open(f"all_results/{dataset}_scores.json", "w") as f:
        json.dump(data, f)





class FormFillingIterator:
    def __init__(
        self,
        context_shortener, 
        form_filler, 
        documents=None, 
        form_generator=None,
        document_generator=None,
        labels=None, 
        evaluation_fnc=None, 
        remove_fields = lambda x:[], 
        argstring="", 
        save=True, 
        load = True, 
        fields_length = 0, 
        mode = "train",
        dataset_name = "",
        output_json_path = None):

        # make sure we have correct inputs
        if documents is None:
            assert labels is None
            assert not form_generator is None
            assert not document_generator is None
            assert fields_length>0
        else:
            assert not form_filler.pydantic_form is None
            assert form_generator is None
            assert document_generator is None

        self.context_shortener = context_shortener
        self.form_filler = form_filler
        self.documents = documents
        self.form_generator = form_generator
        self.document_generator = document_generator
        self.labels = labels
        self.evaluation_fnc = evaluation_fnc
        self.remove_fields = remove_fields
        self.argstring = argstring
        self.save = save
        self.load = load
        self.fields_length = fields_length
        self.mode = mode
        self.dataset_name = dataset_name
        self.output_json_path = output_json_path
        self.field_names = self.form_filler.pydantic_form.__fields__ 

        self.all_scores = {}
        self.index_log = {}
        self.choice_log = {}
        for field in self.field_names:
            self.all_scores[field] = []
            self.index_log[field] = []
            self.choice_log[field] = []
        #self.all_times = []
        self.skips = 0

        if documents is None:
            self.iterate = self._iterate_using_generator
        else:
            self.iterate = self._iterate_using_list

    @weave.op()
    def __call__(self):
        self.iterate()

        if self.documents is None or self.labels:
            return self.evaluate()
        return

    def _iterate_using_generator(self):

        while True:
            key, paper_labels = self.document_generator.get_next_labels()

            # get equal amount of predictions for each label
            # by removing labels for fields with enough predictions already
            skipped_fields = 0
            for field in paper_labels:
                if len(self.all_scores[field]) >= self.fields_length:
                    #if len(paper_labels[field]):
                    #    print("--- skipping field with enough preidictions:", field)
                    paper_labels[field] = []
                    skipped_fields += 1
            if skipped_fields >= len(self.all_scores): # all fields have the required number of predictions
                print("---------Enough predictions made")
                break

            if len(paper_labels) == len(self.remove_fields(paper_labels)):
                #print("!!! No usable labels, skippping paper")
                continue

            # now that most labels have been disgarded (via continue), we load document (which takes a bit of time)
            paper_text = self.document_generator.get_paper_text(key)

            pydantic_form = self.form_generator(seed = int(key)) # use key as seed to ensure unique seeds
            self.form_filler.re_set_pydantic_form(pydantic_form)
            self.context_shortener.set_pydantic_form(pydantic_form)

            filled_form = self.fill_single_form(key,paper_text, paper_labels)

            # log index
            self.log_index(filled_form, paper_labels, pydantic_form)

    def log_index(self,filled_form, paper_labels, pydantic_form):

        pydantic_fields = pydantic_form.__fields__
        for fieldname in paper_labels:
            if len(paper_labels[fieldname]):
                assert len(paper_labels[fieldname]) == 1

                literal_values = pydantic_fields[fieldname].annotation.__args__

                label_choice = paper_labels[fieldname][0]
                label_index = literal_values.index(label_choice)

                pred_choice = getattr(filled_form, fieldname)
                pred_index = literal_values.index(pred_choice)

                print((label_index, pred_index))
                self.index_log[fieldname].append((label_index, pred_index))
                self.choice_log[fieldname].append((label_choice, pred_choice))


    def _iterate_using_list(self):
        # iterate through documents
        for docnr, key in enumerate(self.documents):

            print("loading doc", key, ", nr", docnr, "/", len(self.documents))
            #start_time = time.time()
            paper_text = self.documents[key]
            if self.labels:
                paper_labels = self.labels[key]

                if self.fields_length:
                    # get equal amount of predictions for each label
                    # by removing labels for fields with enough predictions already
                    skipped_fields = 0
                    for field in paper_labels:
                        if len(self.all_scores[field]) >= self.fields_length:
                            #if len(paper_labels[field]):
                            #    print("--- skipping field with enough preidictions:", field)
                            paper_labels[field] = []
                            skipped_fields += 1
                    if skipped_fields >= len(self.all_scores): # all fields have the required number of predictions
                        print("---------Enough predictions made")
                        break

                if len(paper_labels) == len(self.remove_fields(paper_labels)):
                    #print("!!! No usable labels, skippping paper")
                    continue
            
                self.fill_single_form(key,paper_text, paper_labels)
            else:
                self.fill_single_form(key,paper_text)


    @weave.op()
    def fill_single_form(self, key, paper_text, paper_labels=None):
        pydantic_form = self.form_filler.pydantic_form

        filled_form = None

        if self.load:
            filled_form = load_form(key, self.argstring, pydantic_form)

            if not (filled_form is None or filled_form == "skipped"):
                # check all fields with labels have been filled
                field_missing = False 
                for field in filled_form.__fields__:
                    label = paper_labels[field]
                    # each paper only have labels for a subset of the fields.
                    # we only calculate score for these
                    if len(label):
                        pred = getattr(filled_form, field)
                        if pred is None:
                            field_missing = True
                            print("misssing field: ", field)
                if field_missing:
                    print("!! unloading document due to missing field(s)!!")
                    filled_form = None # un-load


        if filled_form is None or filled_form == "skipped":
        
            print("--------- setting document")
            self.context_shortener.set_document(paper_text)
        
            # fill out the form
            try:
                print("--------- generating")
                if not paper_labels is None:
                    filled_form = self.form_filler.forward(self.context_shortener, exclude_fields=self.remove_fields(paper_labels))
                else:
                    filled_form = self.form_filler.forward(self.context_shortener)
                if self.save:
                    save_form(key, self.argstring, filled_form.dict())
            except torch.OutOfMemoryError:
                print("OUT of memory, skipping")
                self.skips += 1
                if self.save:
                    save_form(key, self.argstring, "skipped")
                return
            except openai.BadRequestError as m:
                print("!! BAD REQUEST ERROR!!")
                print(m)
                print("skipping paper and filling zeros in score")
                # evaluate
                if self.labels:
                    for field in list(set(paper_labels.keys())-set(self.remove_fields(paper_labels))):
                        print("zeroing field:", field)
                        self.all_scores[field].append(0)
                    #all_times.append(time.time()-start_time)
                    return

        elif filled_form == "skipped":
            self.skips += 1
            return


        if not self.output_json_path is None:
            # unlike save_form, which is meant to save masses of queries in training/testing, this is the output file for inference
            # NOTE: This will overwrite!
            jsondata = {
                    "filled_form": filled_form.dict(),
                    "context" : self.form_filler.contexts,
                    }
            with open(self.output_json_path, "w") as f:
                json.dump(jsondata, f, indent=4)


        # evaluate
        if not paper_labels is None:
            scores = self.evaluation_fnc(paper_labels, filled_form, verbose=False)

            print("score:", scores)
            print("\n")
            for field in scores:
                self.all_scores[field].append(scores[field])
            #all_times.append(time.time()-start_time)

            #print number of scores in each field
            for fn in self.all_scores:
                print(fn, len(self.all_scores[fn]), end= "; ")
            print("")

        return filled_form

    def evaluate(self):

        if self.mode == "test":
            save_score(self.argstring, self.all_scores, self.index_log, self.choice_log, dataset = self.dataset_name)
        #print("________printing final scores:")
        #pprint.pprint(all_scores)
        means_by_field = {}
        for field in self.all_scores:
            print(field, np.mean(self.all_scores[field]))
            means_by_field[field] = np.mean(self.all_scores[field])

        # calculate mean score
        final_score = []
        final_accuracy = []
        final_similarity = []
        for field in self.all_scores:
            scores = self.all_scores[field]
            final_score.extend(scores)

            field_properties = self.form_filler.pydantic_form.schema()["properties"][field]
            if (
                    field_properties["type"] == "integer" or
                    (field_properties["type"] == "string" and "enum" in field_properties)
                    ):
                print(field, " -- accuacy")
                final_accuracy.extend(scores)
            else:
                final_similarity.extend(scores)
                print(field, " -- similarity ")
            #print(field_properties["type"], "enum" in field_properties, field_properties)
        print("all scores:", final_score)
        print("length:", len(final_score))
        print("mean", np.mean(final_score))
        
        info_to_log = means_by_field
        info_to_log["final_scores"] = final_score
        info_to_log["total_score"] = np.mean(final_score)
        info_to_log["total_accuracy"] = np.mean(final_accuracy)
        info_to_log["total_similarity"] = np.mean(final_similarity)
        #info_to_log["seconds"] = np.mean(all_times)
        info_to_log["papers_skipped"] = self.skips

        return info_to_log
