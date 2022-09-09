from abc import ABC
from typing import Optional, Any, Union
from dataclasses import dataclass
from time import sleep
import os

import torch
import torch.nn.functional as F
import numpy as np
import openai
from transformers import PreTrainedModel, PreTrainedTokenizer, AutoModelForCausalLM, AutoTokenizer
from tqdm import trange


@dataclass
class Batch:
    model_name: str
    texts: list[str]
    logprobs: Optional[np.ndarray] = None
    token_logprobs: Optional[list[list[float]]] = None
    

    def __len__(self):
        return len(self.texts)

    def __add__(self, other):
        assert self.model_name == other.model_name
        if self.logprobs is not None and other.logprobs is not None:
            merged_logprobs = np.concatenate([self.logprobs, other.logprobs], axis=0)
        elif self.logprobs is None and other.logprobs is None:
            merged_logprobs = None
        else:
            raise TypeError()
        return Batch(
            texts=self.texts + other.texts, 
            model_name=self.model_name,
            logprobs=merged_logprobs
        )


class LanguageModel(ABC):

    def get_logprobs(self: Batch) -> np.ndarray:
        pass

    def sample(self, num_samples: int = 32, save_logprobs: bool = True) -> Batch:
        pass

class GPT3(LanguageModel):
    model_name: str = "text-davinci-002"
    max_tokens: int = 16
    total_tokens_used: int = 0
    batch_size: 8

    def __init__(self, model_name: Optional[str] = "text-davinci-002", max_tokens: int = 16, batch_size: int = 32):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.total_tokens_used = 0
        self.batch_size = batch_size
        openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def get_logprobs(self, batch: Batch) -> np.ndarray:
        assert all(len(text) > 0 for text in batch.texts)
        response = openai.Completion.create(
            model=self.model_name,
            prompt=batch.texts,
            max_tokens=0,
            temperature=1,
            logprobs=1,
            echo=True
        )
        self.total_tokens_used += response.usage.total_tokens
        token_logprobs = [response.choices[i].logprobs.token_logprobs[1:] for i in range(len(batch))]
        sequence_logprobs = [np.asarray(logprobs).sum() for logprobs in token_logprobs]
        return np.stack(sequence_logprobs, axis=0)

    def sample(self, num_samples: int = 32, save_logprobs: bool = True) -> Batch:
        batch = Batch(model_name=self.model_name, texts=[], logprobs=[] if save_logprobs else None)
        for _ in trange(num_samples//self.batch_size or 1): 
            minibatch_size = min(self.batch_size, num_samples)
            while True:
                try:
                    response = openai.Completion.create(
                        model=self.model_name,
                        n=minibatch_size,
                        temperature=1, 
                        logprobs=1 if save_logprobs else None,
                        echo=True,
                        max_tokens=self.max_tokens
                    )
                except openai.error.RateLimitError:
                    sleep(30)
                else:
                    break
            self.total_tokens_used += response.usage.total_tokens
            texts = [response.choices[i].text for i in range(minibatch_size)]
            if save_logprobs:
                token_logprobs = [response.choices[i].logprobs.token_logprobs[1:] for i in range(minibatch_size)]
                sequence_logprobs = [np.asarray(logprobs).sum() for logprobs in token_logprobs]
                logprobs = np.stack(sequence_logprobs, axis=0)
            else:
                logprobs = None
            batch += Batch(
                model_name=self.model_name,
                texts=texts,
                logprobs=logprobs,
                token_logprobs=token_logprobs
            )
        return batch

class HFModel(LanguageModel):
    model_name: str
    hf_model: PreTrainedModel
    hf_tokenizer: PreTrainedTokenizer
    max_tokens: int = 128
    generate_batch_size: int = 32
    eval_batch_size: int = 32
    total_tokens_used: int = 0
    device: torch.device

    def __init__(
        self, 
        model_name: str, 
        hf_model: PreTrainedModel, 
        hf_tokenizer: PreTrainedTokenizer, 
        max_tokens: int = 128, 
        generate_batch_size = 32, 
        eval_batch_size = 32,
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.model_name = model_name
        self.hf_model = hf_model
        self.hf_tokenizer = hf_tokenizer
        self.max_tokens = max_tokens
        self.generate_batch_size = generate_batch_size
        self.eval_batch_size = eval_batch_size
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.total_tokens_used = 0
        self.hf_model.to(device)
        if hf_tokenizer.pad_token is None:
            hf_tokenizer.pad_token = hf_tokenizer.eos_token
    
    @classmethod
    def from_pretrained(cls, model_name: str, tokenizer_name: Optional[str], device: Optional[Union[str, torch.device]] = None, model_kwargs: Optional[dict[str, Any]] = {}, **kwargs) -> 'HFModel':
        return HFModel(
            model_name=model_name,
            hf_model=AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs),
            hf_tokenizer=AutoTokenizer.from_pretrained(model_name or tokenizer_name),
            device=device,
            **kwargs
        )

    def sample(self, num_samples: int = 32, save_logprobs: bool = True) -> Batch:
        assert num_samples % self.generate_batch_size == 0
        batch = Batch(model_name=self.model_name, texts=[], logprobs=[] if save_logprobs else None)
        for _ in range(num_samples//self.generate_batch_size or 1): 
            output = self.hf_model.generate(
                text=[self.hf_tokenizer.bos_token]*self.generate_batch_size,
                do_sample=True,
                top_k=0,
                top_p=1,
                min_length=2,
                num_return_sequences=self.generate_batch_size,
                max_length=self.max_tokens,
                padding='max_length',
                return_dict_in_generate=True,
                output_scores=save_logprobs
            )
            texts = self.hf_tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
            if save_logprobs:
                logits = torch.stack(output.scores, dim=1)
                attention_mask = output.sequences != self.hf_tokenizer.pad_token_id
                logprobs = self._get_logprobs_from_logits(
                    input_ids=output.sequences[:, 1:, None], 
                    logits=logits, 
                    mask=attention_mask[:, 1:]
                ).cpu().numpy()
            else:
                logprobs = None
            batch += Batch(model_name=self.model_name, texts=texts, logprobs=logprobs)
        return batch

    def get_logprobs(self, batch: Batch) -> np.ndarray:
        logprobs: list[np.ndarray] = []
        for i in trange(0, len(batch), self.eval_batch_size):
            current_indices = slice(i, i+self.eval_batch_size)
            inputs = self.hf_tokenizer(
                text=[self.hf_tokenizer.bos_token + text for text in batch.texts[current_indices]],
                padding=True,
                max_length=self.max_tokens,
                return_tensors="pt"
            ).to(self.device)
            with torch.inference_mode():
                logits = self.hf_model.forward(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask
                ).logits
                logprobs_minibatch = self._get_logprobs_from_logits(
                    input_ids=inputs.input_ids[:, 1:, None], 
                    logits=logits[:, :-1], 
                    mask=inputs.attention_mask[:, :-1]
                ).cpu().numpy()
            logprobs.append(logprobs_minibatch)
        return np.concatenate(logprobs, axis=0)

    def _get_logprobs_from_logits(self, input_ids: torch.LongTensor, logits: torch.FloatTensor, mask: torch.LongTensor) -> torch.FloatTensor:
        log_probs = F.log_softmax(logits, dim=2)
        input_token_logprobs = log_probs.gather(2, input_ids).squeeze(dim=2)
        # masking out logprobs of padding tokens
        input_token_logprobs = torch.where(mask.bool(), input_token_logprobs, torch.zeros_like(input_token_logprobs))  
        return input_token_logprobs.double().sum(dim=1)


def evaluate_forward_kl(hf_model: PreTrainedModel, num_samples: int = 1024, max_tokens: int = 32):
    gpt3 = GPT3(max_tokens=max_tokens)
    gpt3_batch = gpt3.sample(num_samples=num_samples, save_logprobs=True)
    hf_logprobs = hf_model.get_logprobs(gpt3_batch)
    print('total_tokens_used', gpt3.total_tokens_used)
    return (gpt3_batch.logprobs - hf_logprobs).mean()

