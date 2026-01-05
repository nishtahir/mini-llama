# Mini LLaMA Codelab

Build your own LLama! This codelab provides a comprehensive walkthrough for building your own decoder only llama model with rotary positional embeddings and grouped-query attention, trained on WikiText. It aims to demonstrate simple but effective practices that can be transfered over to other models and projects. 

This guide assumes some datascience background and that you are comfortable with Python and PyTorch.

## Objectives and Outcomes
- Understand how a LLaMA-style config is represented for Hugging Face tooling.
- Implement the model (RoPE, attention, feedforward, Transformer block) and wire it into `PreTrainedModel`.
- Train on a small WikiText slice, log loss, and export weights/tokenizer/config.
- Run greedy and beam-search generation on the saved checkpoint.

## Prerequisites
- Python 3.12+, a recent PyTorch build (GPU via CUDA or Apple MPS; CPU works but is slower). I trained a copy on an M1 Macbook Pro 32GB
- Enough disk for the WikiText dataset download (~650 MB).

## Step 0 - Project Skeleton
1) Start by creating and activating your virtual environment. 

```
python3 -m venv .venv
source .venv/bin/activate
```

2) Upgrade `pip` and install `uv`

```
pip install --upgrade pip
pip install uv
```

`uv` is a build tool and package manager that will be used to create and manage the project. 

Run `uv init` to initialize the project and create a python package. The folder i'm in is called `mini-llama`. `.` tells `uv` to use the current folder

```
uv init . --package
```

This creates a directory structure with the following files


```
pyproject.toml    # Project manifest. Contains things like dependencies.
src/              # scr folder, your package goes here.
  mini_llama/     # your package.
    __init__.py
    main.py
.python-version.  # Contains the version of python you are using for other tools
```

> Note: Keeping the package under `src/` avoids importing the working directory accidentally.

## Step 1: Setup
1) Open the `pyproject.toml`. This is the project manifest. It contains information about your package. Python version constraints and dependencies.

2) Use `uv` to install dependencies we'll be using

```
uv add click datasets torch transformers
```

3) Notice that `uv` adds the dependencies to the project manifest along with the versions that were installed.

4) Next create `config_minillama.py` and `modeling_minillama.py`

```
touch src/mini_llama/conifg.py src/mini_llama/modeling_minillama.py
```

## Step 2: Define the Model Config
Configs capture hyperparameters and serialization for Hugging Face tooling. They provide values to `AutoModel` and `from_pretrained` reconstruct your model.

1) Edit `config_minillama.py` to include the following config definition 

```python
from transformers import AutoConfig, PretrainedConfig

class MiniLlamaConfig(PretrainedConfig):
    model_type = "mini-llama"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = 8,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout

MiniLlamaConfig.register_for_auto_class(AutoConfig)
```
- `PretrainedConfig` carries defaults; `register_for_auto_class` lets `AutoConfig` discover it.
- `dim`, `n_heads`, and `n_layers` define model size; `n_kv_heads` enables grouped-query attention (GQA).
- `multiple_of` controls the SwiGLU hidden size rounding.

The config class is initialized with defaults, these will apply only in situations where a user creates a config without passing those parameters. They should ideally be realistic but don't have to be exactly what you plan to train.

## Step 3: Positional Encoding Helpers
Rotary Position Embeddings (RoPE) encode positions by rotating query/key vectors in complex space. We precompute frequencies to keep things fast.

1) Edit `modeling_minillama.py`. Include the following.

```python
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, xq_.size(1), 1, xq_.size(-1))
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```
- `precompute_freqs_cis` builds complex exponentials for all time steps up to `end`.
- `apply_rotary_emb` reshapes queries/keys to complex pairs, rotates them, and returns real-valued tensors.


## Step 4: Normalization and Feedforward Blocks
LLaMA variants use RMSNorm and a SwiGLU-style feedforward

1) Edit `modeling_minillama.py`. Include the following.

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight
```
- `_norm` computes root-mean-square normalization.
- `weight` is a learnable scale; casting keeps precision on lower-precision devices.

2) Edit `modeling_minillama.py`. Include the following.

```python
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, multiple_of):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
```
- Hidden size is reduced by `2/3` then rounded to a multiple for efficiency.
- Two projections (`w1`, `w3`) feed the gated SwiGLU, `w2` projects back to the model dimension.

> Note: what would other activation functions here do to gradient flow compared to SiLU?

## Step 5: Attention with Grouped-Query Heads
Multi-head attention projects queries/keys/values; GQA shares keys/values across head groups to save compute.

1) Edit `modeling_minillama.py`. Include the following.

```python
class Attention(nn.Module):
    def __init__(self, args: MiniLlamaConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        if args.dim % args.n_heads != 0:
            raise ValueError("Model dimension must be divisible by number of heads.")
        head_dim = args.dim // args.n_heads
        if head_dim % 2 != 0:
            raise ValueError("Head dimension must be even to apply rotary embeddings.")
        if args.n_kv_heads > args.n_heads:
            raise ValueError("n_kv_heads must be less than or equal to n_heads.")
        if args.n_heads % args.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads for GQA.")
        self.head_dim = head_dim

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        if self.n_kv_heads < self.n_heads:
            xk = torch.repeat_interleave(xk, self.n_heads // self.n_kv_heads, dim=2)
            xv = torch.repeat_interleave(xv, self.n_heads // self.n_kv_heads, dim=2)

        xq, xk, xv = xq.transpose(1, 2), xk.transpose(1, 2), xv.transpose(1, 2)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        output = torch.matmul(F.softmax(scores.float(), dim=-1).type_as(xq), xv)

        return self.wo(output.transpose(1, 2).contiguous().view(bsz, seqlen, -1))
```
- `repeat_interleave` shares keys/values when `n_kv_heads` < `n_heads`.
- Masking uses `-inf` on future positions for causality.

> Consider: how does lowering `n_kv_heads` change memory and quality? Try `n_kv_heads = 1` to mimic MHA with shared keys/values.

## Step 6: Transformer Block and Model Assembly
Now we're going to add a Transformer block that uses the modules we made earler. The Transformer block applies attention then feedforward with residuals and RMSNorm; the full model will stack blocks and computes logits.

1) Edit `modeling_minillama.py` to include the Transformer block module.

```python
class TransformerBlock(nn.Module):
    def __init__(self, layer_id, args):
        super().__init__()
        self.layer_id = layer_id
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args.dim, 4 * args.dim, args.multiple_of)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis, mask):
        h = x + self.attention(self.attention_norm(x), freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out
```
- Pre-norm style: normalize before sublayers, then add residuals.

Finally we assemble the model which incorporates the Tranformer block.

2) Edit `modeling_minillama.py` to include the assembled model.

```python
class MiniLlama(PreTrainedModel):
    config_class = MiniLlamaConfig

    def __init__(self, config: MiniLlamaConfig):
        super().__init__(config)
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(i, config) for i in range(config.n_layers)]
        )
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads, config.max_seq_len * 2
        )

    def forward(self, tokens, targets=None):
        bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seqlen].to(h.device)

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, freqs_cis, mask)

        logits = self.output(self.norm(h))

        if targets is not None:
            return logits, F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )
        return logits, None
```
- RoPE tensors are precomputed once, sliced per sequence length.
- The causal mask blocks attention to future tokens.
- Optional loss lets training/inference share the same forward.

# Step 7: CLI entrypoint

Now we're going to setup a CLI and training script to train our model

1) Replace the content of `main.py` wit the following

```python
import click

@click.group()
def cli():
    pass


if __name__ == "__main__":
    cli()
```

This sets up a cli entrypoint using click.

2) Edit `pyproject.toml` to add the entrypoint

```
[project.scripts]
minillama = "mini_llama.main:cli"
```

This adds a project script you can run for the terminal. 

3) Make the script available by syncing with uv

```
uv sync
```

You can now execute your cli by running `minillama`.

We're going to add a CLI command with a training script to train the model. Having a CLI provides some advantages:
- Clear a clear and consistent interface for interacting with the script
- Sensible defaults that inform how you expect it to be used
- Customization options for the user to fascilitate experimentation

# Step: Training

4) Edit `main.py`

```python
def get_batch(data: list[int], batch_size: int, block_size: int, device):
    # We're picking random sequences from our data that fill the context window (block_size)

    # Random start incides
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # x is our training sequence
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )

    # y is the target sequence shifted over by 1
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    return x.to(device), y.to(device)


@cli.command() # Note this references cli which we defined earlier, not click
@click.option("--batch-size", default=2, help="Batch size")
@click.option("--max-steps", default=10000, help="Limit steps for testing")
@click.option("--dim", default=1024, help="Model embedding dimension")
@click.option("--n-layers", default=6, help="Number of transformer blocks")
@click.option("--n-heads", default=8, help="Number of attention heads")
@click.option("--output", default="target/minillama", help="")
@click.option("--lr", default=0.0001, help="")
@click.option("--log-every", default=50, help="")
@click.option("--tokenizer", default="gpt2", help="")
def train(
    *,
    batch_size: int,
    max_steps: int,
    n_layers: int,
    n_heads: int,
    dim: int,
    output: str,
    lr: float,
    log_every: int,
    tokenizer: str,
):
    device = torch.device("mps")

    # Setup tokenizer. We're borrowing GPT
    # but you can train your own BPE tokenizer easily
    # This gives you control over vocab size
    pretrained_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    pretrained_tokenizer.pad_token = pretrained_tokenizer.eos_token

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    ds = ds["train"]
    ds = ds.take(100000)
    # Bulk concat training text
    text = "\n".join(ds["text"])

    # Normally you tokenize a batch as you fill the context window
    # but we can get away with tokenizing the entire sequence.
    tokens = pretrained_tokenizer.encode(text)
    train_data = np.array(tokens)

    # Instantiate the config. The values here get saved in config.json when exported
    config = MiniLlamaConfig(
        vocab_size=pretrained_tokenizer.vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        max_seq_len=pretrained_tokenizer.model_max_length,
    )

    # Model architecture is instantiated using values from the config
    model = MiniLlama(config)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    for step in range(max_steps):
        xb, yb = get_batch(train_data, batch_size, config.max_seq_len, device)

        # Forward pass
        logits, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % log_every == 0:
            print(f"Step {step} | Loss: {loss.item():.4f}")

    # Save the pretrained components.
    config.save_pretrained(output)
    model.save_pretrained(output)
    pretrained_tokenizer.save_pretrained(output)
```
- CLI options expose key hyperparameters and additional training parameters.
- Click does the work of mapping the cli options to function arguments so you
just need to focus on making sure things get routed properly.

> Note: Not everything has to appear in the CLI. It's important to identify things that are likely to change frequently knobs you want to expose for experimentation. Sometimes the best CLI option is no CLI option. 

- `ix` picks random start positions; `x` is the context, `y` is shifted targets.
- Shifting by one trains causal next-token prediction.
- Training loop: forward → loss → backward → optimizer step; logging every `log_every`.
- Save config, weights, and tokenizer to a single folder for later loading.


You should be able to view this CLI using the built in help menu

5) View the `minillama train` help menu

```
minillama train --help
Usage: minillama train [OPTIONS]

Options:
  --batch-size INTEGER  Batch size
  --max-steps INTEGER   Limit steps for testing
  --dim INTEGER         Model embedding dimension
  --n-layers INTEGER    Number of transformer blocks
  --n-heads INTEGER     Number of attention heads
  --output TEXT
  --lr FLOAT
  --log-every INTEGER
  --tokenizer TEXT
  --help                Show this message and exit.
```

6) Launch a training job using the CLI

```
minillama train
```

This might take a while but, the dataset should get pulled and the model loaded and training should begin shortly. 

7) Lauch a training job with adjusted training parameters

```
minillama train --n-layers 4
```

After a completed training session, the output should be `target/minillama` (unless specified otherwise). This should contain the artifacts from training

8) View training artifacts in `target/minillama`

```
target/minillama
├── config.json
├── config_minillama.py
├── merges.txt
├── modeling_minillama.py
├── model.safetensors
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── vocab.json
```

# Step 9: Sampling - Greedy sampling 
Iteratively sample next tokens from the model’s probability distribution.

1) Edit `main.py` to include the following

```python
@cli.command()
@click.option("--model", default="target/minillama", help="Checkpoint to use")
@click.option(
    "--prompt", default="The capital of", help="Seed text to start generation"
)
@click.option(
    "--max-new-tokens", default=30, show_default=True, help="Tokens to generate"
)
def infer(model: str, prompt: str, max_new_tokens: int):
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model = AutoModel.from_pretrained(model, trust_remote_code=True)
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    # Simple generation loop inline
    for _ in range(max_new_tokens):
        logits, _ = model(input_ids)
        logits = logits[:, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, idx_next), dim=1)

    print(tokenizer.decode(input_ids[0].tolist()))
```

- Loads the saved artifacts; seeds the generation with `"The"`.
- Samples one token at a time, always conditioning on the full sequence.
- `torch.multinomial` draws from the probability distribution instead of always taking argmax.

> Note: The model is loaded using the standard Auto classes, this is due to the registration performed earlier. Transformers checks the directory we point it to for config.json which points it to config_minillama.py and modeling_minillama.py. Those are then dynamically loaded which is why `trust_remote_code` is necessary. Since external code is executed this is inherently dangerous - thus requiring the opt-in flag. 

> Please don't trust any random model on the internet without vetting the authors

2) Run `minillama infer` to generate output using your trained model

```
minillama infer
```

# Step 10: Sampling Beam
Beam sampling keeps multiple candidate sequences, expanding by top tokens each step to balance exploration and quality.


1) Edit `main.py`

```python
@cli.command()
@click.option("--model", default="target/minillama", help="Checkpoint to use")
@click.option(
    "--prompt", default="The capital of", help="Seed text to start generation"
)
@click.option("--beam-width", default=10, show_default=True, help="Number of beams")
@click.option(
    "--max-new-tokens", default=30, show_default=True, help="Tokens to generate"
)
@click.option("--top-k", default=50, help="Top k tokens to sample from")
def infer_beam(
    model: str,
    prompt: str,
    beam_width: int,
    max_new_tokens: int,
    top_k: int,
):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model = AutoModel.from_pretrained(model, trust_remote_code=True).to(device)
    model.eval()

    input_ids = torch.tensor(
        [tokenizer.encode(prompt)], dtype=torch.long, device=device
    )

    beams = [(input_ids, 0.0)]
    max_seq_len = getattr(
        model.config,
        "max_seq_len",
        getattr(model.config, "max_position_embeddings", None),
    )
    steps = max_new_tokens
    if max_seq_len is not None:
        steps = min(max_new_tokens, max_seq_len - input_ids.shape[1])

    with torch.no_grad():
        for _ in range(max(steps, 0)):
            # Evaluate all beams in a single forward pass
            beam_tokens = torch.cat([seq for seq, _ in beams], dim=0)
            logits, _ = model(beam_tokens)
            log_probs = torch.nn.functional.log_softmax(logits[:, -1, :], dim=-1)

            candidates = []
            for beam_idx, (seq, score) in enumerate(beams):
                top_log_probs, top_idx = torch.topk(log_probs[beam_idx], top_k)
                for log_prob, token_id in zip(top_log_probs, top_idx):
                    next_seq = torch.cat([seq, token_id.view(1, 1)], dim=1)
                    candidates.append((next_seq, score + log_prob.item()))

            candidates.sort(key=lambda x: x[1], reverse=True)
            beams = candidates[:beam_width]

    best_sequence, _ = beams[0]
    print(tokenizer.decode(best_sequence[0].tolist()))
```

2) Run a beam inference

```
minillama infer-beam
```

3) Compare this with the output of `minillama infer` from earlier

Beam searches can yeild much better cumulative predictions but are much more computationally expensive. Good sampling strategies can make even bad models usable but there are pros and cons.


## What You Learned
- How to express a model’s hyperparameters with `PretrainedConfig` so it can be saved/loaded with Hugging Face utilities.
- How RoPE, GQA attention, RMSNorm, and SwiGLU feedforward combine inside a decoder-only Transformer.
- How to wire a minimal training loop with random contiguous batches for next-token prediction.
- How to checkpoint and reload for both greedy sampling and beam search.

## Next Steps
- Swap in a custom tokenizer (e.g., SentencePiece) to control vocab size and coverage.
- Add evaluation: perplexity on a held-out split, or qualitative prompts.
- Try mixed precision (`torch.autocast`) for faster training on GPU.
- Explore scaling laws: systematically increase `dim`/`layers` and record loss vs. compute.
- Implement the generate function in modeling_minillama.py with an optimized function for inference.
- Run the project using vllm's transformers backend 
