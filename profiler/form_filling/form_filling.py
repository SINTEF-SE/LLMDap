import torch
from pydantic import constr
from pydantic_core._pydantic_core import ValidationError as pydantic_ValidationError
import pydantic
import typing
import weave
import json
import copy
import pprint
import numpy as np
from difflib import SequenceMatcher
import openai


from form_filling import listify_pydantic
from form_filling import regex_handling



def make_FormFillPrompt(context, 
                        answer_field_name=None, 
                        answer_field_type=None, 
                        answer_field_description=None, 
                        listed_answer = False, 
                        prompt_for_reasoning=False,
                        reason=None
                        ):
    # get prompt for sequential form filling (i.e. one field at a time), field-agnistic.
    prompt = f"""
You are to fill out values of a form based on some context from a part of a scientific paper or document.
Use only the context to reply. If the answer is not in the context directly, make a qualified guess based on what is in the context.
"""
    if listed_answer:
        prompt += "Answer in list form, with as many answers as fitting.\n"
    prompt += "\n"

    # Add context
    prompt += f"context: {{{{{{{context}}}}}}}\n\n"

    # Add field info
    if not answer_field_name is None:
        prompt += f"answer field name: {{{{{{{answer_field_name}}}}}}}\n\n"
    if not answer_field_type is None:
        prompt += f"answer field type: {{{{{{{answer_field_type}}}}}}}\n\n"
    if not answer_field_description is None:
        prompt += f"answer field description: {{{{{{{answer_field_description}}}}}}}\n\n"

    # Add reasoning step
    if prompt_for_reasoning:
        prompt += "Reasoning step: {{{Let's think step by step. "
        return prompt
    if not reason is None:
        prompt += f"Reasoning step: {{{{{{{reason}}}}}}}\n\n"

    prompt += "Answer: {{{"
    return prompt




def get_constraints_from_field(field):
    """ get field_type, and any min/max length if its a constr """
    field_type = field.annotation
    metadata = field.metadata

    # determine any constraints
    if len(metadata):
        constraints = metadata[0]
        assert field_type == str # this is the only constrained field implemented for now
        assert type(constraints) == pydantic.types.StringConstraints
        min_l, max_l = constraints.min_length, constraints.max_length
    else:
        min_l, max_l = None, None
    return field_type, min_l, max_l



class FieldFiller:
    """
    Module for filling out a single field in a pydantic form.
    One FieldFiller is made per schema field, but used for multiple documents.
    document shortener is fed in the forwards fuction.
    """
    def __init__(self, answer_generator, answer_in_quotes=False, listify = False, verbose=False):

        self.answer_generator = answer_generator
        self.answer_in_quotes = answer_in_quotes
        self.listify = listify
        self.verbose = verbose

    def forward(self, prompt_input, context, field_type):

        # generate answer
        prompt_input["context"] = context
        answer = self.answer_generator(make_FormFillPrompt(**prompt_input))
        #print("generated answer", answer, type(answer))
        assert type(answer) is str

        if self.listify:
            assert answer[0] == "["
            assert answer[-1] == "]"

            # remmove space after comma
            answer = answer.replace(", ",",")

            # remove any last comma
            if answer[-2] == ",":
                answer = answer[:-2]+"]"

            # add double quotes so its json parsable
            if not self.answer_in_quotes:
                answer = answer.replace("[",'["')
                answer = answer.replace(",",'","')
                answer = answer.replace("]",'"]')

            # parse
            try:
                answers = json.loads(answer)
            except json.decoder.JSONDecodeError as e:
                print("unparsed:")
                print(answer)
                print([answer])
                print(type(answer))
                print(not self.answer_in_quotes)
                print("")
                raise e

            answers = [self.parse_single_output(field_type, answer) for answer in answers]
            return answers

        else:
            if self.answer_in_quotes:
                answer=answer[1:-1] # remove quotes. Not nescessary with listify since its done by json.loads
            return parse_single_output(field_type, answer)

def parse_single_output(field_type, stringoutput, answer_in_quotes = None):
    # parse string output into the given type
    try:
        if getattr(field_type, "__origin__", None) is typing.Literal:
            output = str(stringoutput)
        else:
            output = field_type(stringoutput)
    except ValueError as e:
        print("\n\n\n")
        print("!!!")
        print("WARNING: FAILED TO READ STRING:", stringoutput, "inserting empty value instead")
        print("Check regex rules - they seem to allow unparsable output for class:", field_type)
        print("answer in quotes?", answer_in_quotes)
        #print(e)
        raise e
        print("\n\n\n")
        output = field_type()
    return output

class SequentialFormFiller:
    """
    Class for iterating through a pydantic schema, and predict each field sequentially.
    Uses outlines to ensure correct field types.
    Dspy is wrapped around outlines, to enable optimization.
    """
    def __init__(self,
                 outlines_llm,
                 outlines_sampler,
                 pydantic_form = None,
                 listify_form = False,
                 answer_in_quotes = True,
                 max_tokens = 50,
                 verbose = False
                 ):
        self.llm_model = outlines_llm
        self.sampler = outlines_sampler
        self.verbose = verbose
        self.max_tokens = max_tokens

        self.answer_in_quotes=answer_in_quotes
        self.listify_form = listify_form
        if not pydantic_form is None:
            self.set_pydantic_form(pydantic_form)

    def set_pydantic_form(self, pydantic_form):
        """ Prepares generator for each field typ in the pydantic form """
        self.pydantic_form = pydantic_form

        self.outlines_generators = {}

        if self.verbose:
            print("Generating regex generators...")

        # iterate through fields
        fields = pydantic_form.__fields__
        for fieldname in fields:
            field_type, min_l, max_l = get_constraints_from_field(fields[fieldname])

            # only make a new generator if it is not equal to one already generated
            if not (field_type, min_l, max_l) in self.outlines_generators:
                outlines_generator = regex_handling.make_constrained_generator(
                        llm_model=self.llm_model,
                        field_type=field_type,
                        min_l=min_l,
                        max_l=max_l,
                        answer_in_quotes=self.answer_in_quotes,
                        listify_form = self.listify_form,
                        sampler = self.sampler)
                self.outlines_generators[(field_type, min_l, max_l)] = outlines_generator

        if self.verbose:
            print("Finished generating regex generators.")

        self.prepare_field_fillers()

    def prepare_field_fillers(self):
        self.field_fillers = {}
        fields = self.pydantic_form.__fields__
        for fieldname in fields:
            field = fields[fieldname]
            field_type, min_l, max_l = get_constraints_from_field(field)
            generator = self.outlines_generators[(field_type, min_l, max_l)]
            self.field_fillers[fieldname] = FieldFiller(
                    answer_generator = generator,
                    verbose = self.verbose,
                    answer_in_quotes = self.answer_in_quotes,
                    listify = self.listify_form,
                    )

    def re_set_pydantic_form(self,pydantic_form):
        """after shufling literal values, there is no need to remake field filleds since they do not use order"""
        if self.pydantic_form is None:
            self.set_pydantic_form(pydantic_form)
        self.pydantic_form = pydantic_form


    @weave.op()
    def forward(self, get_context, exclude_fields = []):

        pydantic_form = get_subschema(self.pydantic_form, exclude_fields = exclude_fields)

        fields = pydantic_form.__fields__
        output_dict = {}
        self.contexts = {}

        # iterate through fields
        if self.verbose:
            print("--INFO--:starting to iterate through fields")
        for fieldname in fields:
            field = fields[fieldname]
            field_type = field.annotation


            # make prompt input
            prompt_input = {
                           "context":None,
                           "answer_field_name":fieldname,
                           "answer_field_description":field.description,
                           "answer_field_type":str(field_type),
                           "listed_answer":self.listify_form,
                            }
            context = get_context(**prompt_input)
            self.contexts[fieldname] = context

            # generate output
            output = self.field_fillers[fieldname].forward(prompt_input, context, field_type)

            output_dict[fieldname] = output

        if self.verbose:
            print("--INFO--: fields iterated")

        # listify form
        if self.listify_form:
            pydantic_form = listify_pydantic.conlistify_pydantic_model(pydantic_form, min_length=1)
        else:
            pydantic_form = pydantic_form

        if self.listify_form:
            output = pydantic_form(**output_dict)
        else:
            # remove weave stuff that raises erros for pydantic validator (i.e. change type from weave.trace.box.boxedstr to str)
            try:
                output = pydantic_form(**{name : val.__str__() for name, val in output_dict.items()})
            except pydantic_ValidationError: # outlines seem to allow non-allowable strings in rare occasions. Workaround: just choose closest allowable answer

                fields = pydantic_form.__fields__
                output_dict = {name : val.__str__() for name, val in output_dict.items()}
                for name, predicted_string in output_dict.items():

                    field = fields[name]
                    field_type = field.annotation
                    if typing.get_origin(field_type) == typing.Literal: # i have only seen this problem in Literal fields
                        allowed_answers = field_type.__args__
                        if not predicted_string in allowed_answers: # only alter the field(s) with the problem
                            
                            best_similarity  = -1.0
                            best_ans = None
                            for ans in allowed_answers:
                                similarity = SequenceMatcher(None, ans, predicted_string).ratio()
                                if similarity > best_similarity:
                                    best_similarity = similarity
                                    best_ans = ans

                            output_dict[name] = best_ans
                            
                            print("!!!!! Failed to generate allowable answer. Finding closest allowable answer instead")
                            print("Predicted answer:", predicted_string, "while closest match is", best_ans)
                            # log this 
                            with open("output_log.txt", "a") as file:
                                file.write(f"\nfound closest match: {predicted_string} -> {best_ans}\n")
                output = pydantic_form(**output_dict)

        torch.cuda.empty_cache()
        return output

#    def deepcopy(self):
#        """ avoid copying llm_model """
#        if self.verbose:
#            print("--INFO--: deepcopy called")
#        lm = self.llm_model
#        self.llm_model = None
#        copy = super().deepcopy()
#        self.llm_model = lm
#        copy.llm_model = lm
#        return copy
#
#    def reset_copy(self):
#        """ avoid copying llm_model """
#        if self.verbose:
#            print("--INFO--: reset_copy called :)")
#        lm = self.llm_model
#        self.llm_model = None
#        copy = super().reset_copy()
#        self.llm_model = lm
#        copy.llm_model = lm
#        return copy


def get_subschema(original_schema: pydantic.BaseModel, exclude_fields: list = [], remove_maxlength= False):
    """Get a pydantic form with fewer fields"""

    # Extract the fields from the original schema
    original_fields = original_schema.__annotations__
    # Filter the fields based on the provided list
    new_fields = {}
    for field in original_fields:
        if not field in exclude_fields:

            properties = original_schema.schema()["properties"][field]
            #print(properties)
            if remove_maxlength:
                # this is needed for openai api
                if "maxLength" in properties:
                    properties.pop("maxLength") # maxlength is simply ignored

                new_fields[field] = (original_schema.__fields__[field].annotation, pydantic.Field(**properties))
            else:
                new_fields[field] = (original_fields[field], pydantic.Field(**properties))

    # Create a new model with the filtered fields
    NewSchema = pydantic.create_model('NewSchema', **new_fields)
    return NewSchema

###
# openai
### 


class OpenAIFormFiller:
    """
    """
    def __init__(self,
                 model_id,
                 pydantic_form=None,
                 listify_form = False,
                 max_tokens = 50,
                 verbose = False
                 ):
        self.model_id = model_id
        self.verbose = verbose
        self.listify_form = listify_form


        if not pydantic_form is None:
            self.set_pydantic_form(pydantic_form)

    def set_pydantic_form(self, pydantic_form):
        self.pydantic_form = pydantic_form
    def re_set_pydantic_form(self, pydantic_form):
        if self.pydantic_form is None:
            self.set_pydantic_form(pydantic_form)
        self.pydantic_form = pydantic_form

    @weave.op()
    def forward(self, get_context, exclude_fields = []):

        pydantic_form = get_subschema(self.pydantic_form, exclude_fields = exclude_fields, remove_maxlength= True)


        output_dict = {}

        # listify form
        if self.listify_form:
            pydantic_form = listify_pydantic.conlistify_pydantic_model(pydantic_form, min_length=1)
            raise NotImplementedError # is this just straight forward? I guess examples and descriptions need some tweeking.
                                      # find out if this is used first, or we need do state them in prompt
                                      # ( in that case we can make a listed prompt i guess)
        else:
            pydantic_form = pydantic_form



        # prepare context
        context = get_context()
        self.contexts = context
        if self.verbose:
            print("context:")
            print(context)
            #print("schema:")
            #print(pydantic_form.schema())


        # generate answer
        completion = openai.beta.chat.completions.parse(
                                                        model = self.model_id,
                                                        messages = [
                                                            {
                                                                "role":"user",
                                                                "content": make_FormFillPrompt(context = context, listed_answer=self.listify_form),
                                                            }
                                                        ],
                                                        response_format = pydantic_form,
                                                        )
        if len(completion.choices) > 1:
            print(completion)
            print(completion.model_dump)
            print(completion.choices)
            quit()
        answer = completion.choices[0].message.content
        print("------------called openai, model=", completion.model)


        
        if self.verbose:
            print("!")
            print("!")
            print("!")
            print("answer generated!")
            print(type(answer))
            print(answer)

        output_dict = json.loads(answer)
        output = pydantic_form(**output_dict)

        return output 



###
# openai sequential
###

@weave.op()
def openAIFieldFiller(prompt_input, # used for retrieval and generation
                      context,
                      field_type,
                      model_id,
                      subschema, # used for generation, and NOT retrieval
                      listify=False,
                      verbose=False
                      ):

        # retireve chunks
        if verbose:
            print("    --INFO--: retrieving context")
            print("              Form input    : ", prompt_input)

        # prepare context
        prompt_input["context"] = context

        # generate answer
        completion = openai.beta.chat.completions.parse(
                                                        model = model_id,
                                                        messages = [
                                                            {
                                                                "role":"user",
                                                                "content": make_FormFillPrompt(**prompt_input, listed_answer=listify),
                                                            }
                                                        ],
                                                        response_format = subschema,
                                                        )
        if len(completion.choices) > 1:
            print(completion)
            print(completion.model_dump)
            print(completion.choices)
            quit()
        answer = completion.choices[0].message.content
        print("------------called openai, model=", completion.model)

        if listify:
            raise NotImplementedError
        else:
            output_dict = json.loads(answer)
            if len(output_dict) != 1:
                print("!!!!!", output_dict)
                raise ValueError
            for key in output_dict:
                value = output_dict[key]
            return value



class OpenAISequentialFormFiller:
    """
    """
    def __init__(self,
                 model_id,
                 pydantic_form = None,
                 listify_form = False,
                 max_tokens = 50,
                 verbose = False
                 ):
        self.model_id = model_id
        self.verbose = verbose
        self.max_tokens = max_tokens
        self.listify_form = listify_form
        if not pydantic_form is None:
            self.set_pydantic_form(pydantic_form)

    def set_pydantic_form(self, pydantic_form):
        self.pydantic_form = pydantic_form
    def re_set_pydantic_form(self, pydantic_form):
        if self.pydantic_form is None:
            self.set_pydantic_form(pydantic_form)
        self.pydantic_form = pydantic_form

    @weave.op()
    def forward(self, get_context, exclude_fields = []):

        pydantic_form = get_subschema(self.pydantic_form, exclude_fields = exclude_fields)

        fields = pydantic_form.__fields__
        output_dict = {}
        self.contexts = {}

        # iterate through fields
        if self.verbose:
            print("--INFO--:starting to iterate through fields")
        for fieldname in fields:
            field = fields[fieldname]
            field_type = field.annotation


            # make prompt input
            prompt_input = {
                           "context":None, 
                           "answer_field_name":fieldname, 
                           "answer_field_description":field.description, 
                           "answer_field_type":str(field_type),
                            }

            context = get_context(**prompt_input)
            self.contexts[fieldname] = context
        
            all_other_fields = list(self.pydantic_form.__fields__.keys())
            all_other_fields.remove(fieldname)
            subschema = get_subschema(self.pydantic_form, exclude_fields = all_other_fields, remove_maxlength= True) # for generation, remove the stuff openai cant handle

            # generate output
            output = openAIFieldFiller(
                      prompt_input = prompt_input,
                      context = context,
                      field_type = field_type,
                      model_id = self.model_id,
                      subschema = subschema,
                      listify=self.listify_form,
                      verbose=self.verbose,
                      )

            output_dict[fieldname] = output

        if self.verbose:
            print("--INFO--: fields iterated")

        # listify form
        if self.listify_form:
            pydantic_form = listify_pydantic.conlistify_pydantic_model(pydantic_form, min_length=1)
        else:
            pydantic_form = pydantic_form

        # make pytantic object and return
        if self.listify_form:
            output = pydantic_form(**output_dict)
        else:
            output = pydantic_form(**{name : val.__str__() for name, val in output_dict.items()})
        torch.cuda.empty_cache()
        return output



class DirectKeywordSimilarityFiller:
    """
    Replaces the sequential form filler when using the direct keyword similarity approach.
    Retrieves similarity matrices instead of chunks, uses the best match as answer instead of generating using llm
    """
    def __init__(self,
                 pydantic_form = None,
                 listify_form = False,
                 order = np.inf, # max norm works quite a bit better than sum/1- or 2-norm
                 verbose = False,
                 ):
        self.verbose = verbose
        self.order = order
        if listify_form:
            raise NotImplementedError # not yet (but could be relatively easy)
        self.listify_form = listify_form
        if not pydantic_form is None:
            self.set_pydantic_form(pydantic_form)
    def set_pydantic_form(self, pydantic_form):
        self.pydantic_form = pydantic_form
    def re_set_pydantic_form(self, pydantic_form):
        if self.pydantic_form is None:
            self.set_pydantic_form(pydantic_form)
        self.pydantic_form = pydantic_form


    @weave.op()
    def forward(self, context_shortener, exclude_fields = []):

        pydantic_form = get_subschema(self.pydantic_form, exclude_fields = exclude_fields)

        fields = pydantic_form.__fields__
        output_dict = {}

        # iterate through fields
        if self.verbose:
            print("--INFO--:starting to iterate through fields")
        for fieldname in fields:
            field = fields[fieldname]
            field_type = field.annotation



            # generate output
            output = self.get_best_answer_for_field(context_shortener, fieldname, field_type)
            self.contexts = {} # entire paper used

            output_dict[fieldname] = output

        if self.verbose:
            print("--INFO--: fields iterated")

        # listify form
        if self.listify_form:
            pydantic_form = listify_pydantic.conlistify_pydantic_model(pydantic_form, min_length=1)
        else:
            pydantic_form = pydantic_form

        if self.listify_form:
            output = pydantic_form(**output_dict)
        else:
            # remove weave stuff that raises erros for pydantic validator (i.e. change type from weave.trace.box.boxedstr to str)
            output = pydantic_form(**{name : val.__str__() for name, val in output_dict.items()})
        torch.cuda.empty_cache()
        return output

    def get_best_answer_for_field(self, context_shortener, fieldname, field_type):

        target_keywords = context_shortener.descriptions[fieldname] # strings to match (e.g. ontology node labels or allowed answers)
        target_keywords = [t.lower() for t in target_keywords] # to lower, to match the allowed answers
        if getattr(field_type, "__origin__", None) is typing.Literal:
            allowed_answers = field_type.__args__


            for ans in allowed_answers:
                assert ans in target_keywords # if not all allowed answers are in the targets, it will not be possible to predict them (could still try predicting the others in certain cases i guess - e.g. ignoring "other", then predict other if best match is not good (future work)
            allowed_indices = []
            for i, kw in enumerate(target_keywords):
                if kw in allowed_answers:
                    allowed_indices.append(i)
        else:
            allowed_indices = list(range(len(target_keywords)))


        # get similarities
        similarities = context_shortener.get_similarity_matrices(fieldname)

        # prep similarities:
        prepared_similarities = []
        for (similarity, kw_scores) in similarities:
            
            # only keep allowed answers
            similarity = similarity[:,allowed_indices]

            # clip minimum to 0, to ignore negatives when using norms later
            similarity = similarity.clip(min=0)

            # adjust for keyword scores
            kw_scores = torch.Tensor(kw_scores)
            similarity = torch.matmul(similarity.T, kw_scores)

            prepared_similarities.append(similarity)

        # to numpy array
        prepared_similarities = np.array([sim.numpy() for sim in prepared_similarities])

        # calculate which answer/node matches the chunks best
        best_match_index = self.calculate_best_match(prepared_similarities)
        best_string = target_keywords[allowed_indices[best_match_index]]
        #print("best string:", best_string)
        return best_string

    def calculate_best_match(self, similarities):
        scores_per_node = np.linalg.norm(similarities, ord=self.order, axis=0)
        am = np.argmax(scores_per_node)
        return am




