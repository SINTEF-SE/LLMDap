import transformers
import torch



class LLM:
    def __init__(self):
        self.load_aaditya_llm(device = "cuda:0")

    def ask(self,prompt,max_tokens=50, temperature=0.1):
        generation_config = transformers.GenerationConfig(
                    max_new_tokens=max_tokens, # amount of tokens, increase if you want more than one word output
                    eos_token_id=self.terminators,
                    pad_token_id=self.terminators[0], # would be inferred and printed a msg
                    do_sample=True,
                    temperature=temperature,
                    top_p=0.9,
                    )
        inputs = torch.tensor([self.pipeline.tokenizer(prompt).data["input_ids"]]).to(self.device)
        outputs = self.pipeline.model.generate(
                    inputs,
                    generation_config,
                    )
        outputs = self.pipeline.tokenizer.decode(outputs[0])
        return outputs

    def load_aaditya_llm(self, device):
        self.device = device
        self.softmax = torch.nn.Softmax(dim=0)

        model_id = "aaditya/OpenBioLLM-Llama3-8B"
        
        self.pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_id,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device=self.device,
                    )
        
        
        self.terminators = [
                    self.pipeline.tokenizer.eos_token_id,
                    self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]
