from kl_gpt3.kl_gpt3 import GPT3


def compute_kl(model_a: GPT3, model_b: GPT3) -> float:
    model_a_samples = model_a.sample(num_samples=64, save_logprobs=True)
    model_b_logprobs = model_b.get_logprobs(model_a_samples)
    return (model_a_samples.logprobs - model_b_logprobs).mean()


cushman = GPT3(model_name='code-cushman-001', max_tokens=64)
davinci = GPT3(model_name='text-davinci-002', max_tokens=64)
print(f'KL(cushman, davinci) = {compute_kl(cushman, davinci):.3f}')
print(f'KL(davinci, cushman) = {compute_kl(davinci, cushman):.3f}')
