# kl-gpt3

# Quickstart

```bash
pip install git+https://github.com/tomekkorbak/kl-gpt3.git
OPENAI_API_KEY=sk-YOURKEY
```

```python
from transformers import AutoModelForCausalLM
from kl_gpt3.kl_gpt3 import HFModel, evaluate_forward_kl

gpt2 = AutoModelForCausalLM.from_pretrained('gpt2')
kl = evaluate_forward_kl(gpt2, max_tokens=32, num_samples=4)
print(kl)
```

# todos
- [ ] handle gpt3 api timeout nicely
- [ ] docstrings
- [ ] add tests
