import numpy as np

from kl_gpt3 import Batch, GPT3, HFModel

def test_batch_merge():
    batch_1 = Batch(
        text=['Hello'],
        model_name='text-davinci-002',
        logprobs=np.asarray([-2.0])
    )
    batch_2 = Batch(
        text=['Hello', 'Hi'],
        model_name='text-davinci-002',
        logprobs=np.asarray([-3.0, -4.0])
    )
    concat_batch = batch_1 + batch_2
    concat_batch.model_name == 'text-davinci-002'
    assert len(concat_batch) == 3
    assert concat_batch.logprobs.shape == (3,)
    assert concat_batch.texts[0] == batch_1.texts[0]
    assert concat_batch.logprobs[2] == batch_2.logprobs[1]

def test_gpt3_gives_consistent_logprobs():
    gpt3 = GPT3()
    batch = gpt3.sample(num_samples=10)
    logprobs = gpt3.get_logprobs(batch)
    assert np.abs(batch.logprobs[0] - logprobs[0]) < 1

def test_hf_gives_consistent_logprobs():
    hf_model = HFModel.from_pretrained('gpt2', max_tokens=128, device=0, generate_batch_size=1, eval_batch_size=1)
    batch = hf_model.sample(num_samples=2)
    assert np.abs(batch.logprobs - hf_model.get_logprobs(batch)) < 1
