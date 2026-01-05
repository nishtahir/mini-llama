from datasets.load import load_dataset
from mini_llama.config_minillama import MiniLlamaConfig
from mini_llama.modeling_minillama import MiniLlama
import click
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


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


@click.group()
def cli():
    pass


@cli.command()
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


if __name__ == "__main__":
    cli()
