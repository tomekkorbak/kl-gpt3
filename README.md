# kl-gpt3

# Quickstart
```bash
pip install git+https://github.com/tomekkorbak/kl-gpt3.git
OPENAI_API_KEY=sk-YOURKEY
```

```python
from kl_gpt3.kl_gpt3 import HFModel evaluate_forward_kl

hf_model = HFModel.from_pretrained('gpt2', max_tokens=32)
kl = evaluate_forward_kl(hf_model, max_tokens=32, num_samples=2048)
print(kl)
```

# todos
[ ] approximate KL
[ ] handle gpt3 api timeout nicely
[ ] docstrings
[ ] add tests