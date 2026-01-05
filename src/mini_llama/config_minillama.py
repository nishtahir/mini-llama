from transformers import AutoConfig, PretrainedConfig


class MiniLlamaConfig(PretrainedConfig):
    model_type = "mini-llama"

    def __init__(
        self,
        *,
        vocab_size: int = 32000,
        dim: int = 512,  # Embedding dimension
        n_layers: int = 8,  # Number of transformer blocks
        n_heads: int = 8,  # Attention heads
        n_kv_heads: int = 8,  # Key/Value heads (for Grouped Query Attention)
        multiple_of: int = 256,  # For SwiGLU hidden layer size
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
