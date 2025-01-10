from dsp.modules.lm import LM
import dspy
import outlines
import weave

from form_filling import regex_handling


def openai_to_hf(**kwargs):
    hf_kwargs = {}
    for k, v in kwargs.items():
        if k == "n":
            hf_kwargs["num_return_sequences"] = v
        elif k == "frequency_penalty":
            hf_kwargs["repetition_penalty"] = 1.0 - v
        elif k == "presence_penalty":
            hf_kwargs["diversity_penalty"] = v
        elif k == "max_tokens":
            hf_kwargs["max_new_tokens"] = v
        elif k == "model":
            pass
        else:
            hf_kwargs[k] = v

    return hf_kwargs


class OutlinesHFModel(LM):
    def __init__(
        self,
        outlines_model,
        generator,
        model_string = "unspecified",
        max_tokens=20,
        kwargs = {}
    ):
        """wrapper for Hugging Face models

        Args:
            model (str): HF model identifier to load and use
            checkpoint (str, optional): load specific checkpoints of the model. Defaults to None.
            is_client (bool, optional): whether to access models via client. Defaults to False.
            hf_device_map (str, optional): HF config strategy to load the model.
                Recommeded to use "auto", which will help loading large models using accelerate. Defaults to "auto".
            model_kwargs (dict, optional): additional kwargs to pass to the model constructor. Defaults to empty dict.
        """

        super().__init__(model_string)
        self.provider = "hf"

        self.model = outlines_model
        self.generator = generator
        self.tokenizer = self.model.tokenizer.tokenizer
        self.device = self.model.device
    
        self.history = []


        if type(generator.sampler)==outlines.samplers.GreedySampler:
            kwargs["temperature"] = 0
        elif type(generator.sampler)==outlines.samplers.BeamSearchSampler:
            kwargs["temperature"] = 0
        elif type(generator.sampler)==outlines.samplers.MultinomialSampler:
            kwargs["temperature"] = generator.sampler.temperature
        kwargs["max_tokens"] = max_tokens
        self.kwargs = kwargs


    def basic_request(self, prompt, **kwargs):
        raw_kwargs = kwargs
        kwargs = {**self.kwargs, **kwargs}
        response = self._generate(prompt, **kwargs)

        history = {
            "prompt": prompt,
            "response": response,
            "kwargs": kwargs,
            "raw_kwargs": raw_kwargs,
        }
        self.history.append(history)

        return response

    @weave.op
    def _generate(self, prompt, **kwargs):
        kwargs = {**openai_to_hf(**self.kwargs), **openai_to_hf(**kwargs)}
        # print(prompt)
        if isinstance(prompt, dict):
            try:
                prompt = prompt["messages"][0]["content"]
            except (KeyError, IndexError, TypeError):
                print("Failed to extract 'content' from the prompt.")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        if 'temperature' in kwargs and kwargs['temperature'] == 0.0:
            kwargs['do_sample'] = False
        
        #outputs = self.model.generate(**inputs, **kwargs)
        #print(kwargs)
        outputs = self.generator(prompt, max_tokens=kwargs["max_new_tokens"])

        #print("GENERATED OUTPUT:", outputs)

        if type(self.generator.sampler)==outlines.samplers.BeamSearchSampler:
            #print("n tokens generated:", [len(self.tokenizer.encode(output)) for output in outputs]) 
            outputs = outputs[0] # greedy sample among the beams
        #else:
        #    print("n tokens generated:", len(self.tokenizer.encode(outputs))) 
        #print("tokens generated:", self.tokenizer.encode(outputs))
        #print(type(outputs))

        #completions = [{"text": c} for c in self.tokenizer.batch_decode(outputs, skip_special_tokens=True)]
        completions = [{"text": outputs}]

        response = {
            "prompt": prompt,
            "choices": completions,
        }
        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        if kwargs.get("n", 1) > 1 or kwargs.get("temperature", 0.0) > 0.1:
            kwargs["do_sample"] = True

        response = self.request(prompt, **kwargs) # this will call basic_request
        return [c["text"] for c in response["choices"]]


def make_dspy_generator(outlines_llm, outlines_generator, max_tokens = 20):
    """ get a dspy generator from outlines llm and generator. """
    lm = OutlinesHFModel(outlines_llm, outlines_generator, max_tokens = max_tokens)

    # Define the predictor.
    @weave.op()
    def predict(dspy_predictor, **prompt_input):
        dspy.settings.configure(lm=lm)
        return dspy_predictor(**prompt_input).answer
    return predict


def make_constrained_generator(field_type, llm_model, min_l, max_l, answer_in_quotes, listify_form, sampler = None):
    """
    make an outlines generator restricted to a specific type*, potentially with constraints, using regex to describe output restrictions 
    * the output is always a string, but a parsable one
    """

    # get regex
    regex = regex_handling.make_regex_string(field_type, min_l, max_l, answer_in_quotes, listify_form)

    # make outlines generators
    generator = outlines.generate.regex(llm_model, regex, sampler=sampler)
    return generator
