from tensor import DType, Tensor, TensorView, Shape

struct KVCache:
    var keys: Tensor[DType.float16]
    var values: Tensor[DType.float16]
    var filled: Int
    var max_seq_len: Int
    var num_layers: Int
    var num_kv_heads: Int
    var head_dim: Int

    def __init__(out self, num_layers: Int, max_seq_len: Int, num_kv_heads: Int, head_dim: Int):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.filled = 0

        var key_shape = Shape.from_params(4, num_layers, max_seq_len, num_kv_heads, head_dim)
        var val_shape = Shape.from_params(4, num_layers, max_seq_len, num_kv_heads, head_dim)

        self.keys = Tensor[DType.float16](key_shape)
        self.values = Tensor[DType.float16](val_shape)

    fn append(inout self, layer: Int, k: TensorView[DType.float16], v: TensorView[DType.float16]) raises:
        if self.filled >= self.max_seq_len:
            raise Error("KV cache overflow")
        if layer < 0 or layer >= self.num_layers:
            raise Error("Invalid layer index")

    fn get(ref self, layer: Int, seq_len: Int) raises -> (TensorView[DType.float16], TensorView[DType.float16]):
        if layer < 0 or layer >= self.num_layers:
            raise Error("Invalid layer index")
        if seq_len > self.filled:
            raise Error("Invalid seq_len")
        let dummy_view = TensorView[DType.float16]()
        return (dummy_view, dummy_view)

    fn clear(inout self):
        self.filled = 0

    fn can_fit(ref self, new_tokens: Int) -> Bool:
        return self.filled + new_tokens <= self.max_seq_len