from std.random import rand
from tensor import DType, TensorView

struct SamplerConfig:
    var temperature: Float32
    var top_p: Float32
    var top_k: Int
    var repetition_penalty: Float32
    var vocab_size: Int

    @staticmethod
    fn default() -> SamplerConfig:
        return SamplerConfig(
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            repetition_penalty=1.0,
            vocab_size=32000,
        )

struct Sampler:
    var config: SamplerConfig

    def __init__(out self, config: SamplerConfig):
        self.config = config

    fn sample(ref self, logits: TensorView[DType.float32]) raises -> (Int, Float32):
        if self.config.temperature == 0.0:
            return greedy_sample(logits)

        let vocab_size = self.config.vocab_size
        var max_logit: Float32 = -1e9
        var max_idx: Int = 0

        for i in range(vocab_size):
            if logits.ptr[i] > max_logit:
                max_logit = logits.ptr[i]
                max_idx = i

        return (max_idx, 0.0)

fn greedy_sample(logits: TensorView[DType.float32]) -> (Int, Float32):
    let vocab_size = logits.shape.dims[0]
    var max_logit: Float32 = -1e9
    var max_idx: Int = 0

    for i in range(vocab_size):
        if logits.ptr[i] > max_logit:
            max_logit = logits.ptr[i]
            max_idx = i

    return (max_idx, max_logit)