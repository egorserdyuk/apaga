# Apaga - Universal Local AI Inference Library for Mojo

Apaga is a high-performance local inference engine for AI models, written in Mojo.

## Features

- **Universal Model Support** - Load GGUF, SafeTensors, and APG formats
- **Multiple Quantizations** - FP16, INT8, INT4, and custom quantization schemes
- **Hardware Abstraction** - CPU and SIMD-accelerated backends
- **Memory Optimization** - Arena-based allocation and weight pooling
- **Inference Runtime** - KV cache management and optimized generation

## Usage

```mojo
from apaga import Model, Session, generate

let model = Model("model.gguf")
let session = Session(model)

let config = GenerateConfig(max_tokens=256, temperature=0.7)
let output = generate(session, "Hello world!", config)
print(output)
```

## Modules

| Module | Description |
|--------|-------------|
| `tensor` | Tensor, Shape, DType, TensorView types |
| `mem` | Memory arena, weight buffers, pool |
| `hal` | Hardware abstraction (CPU, SIMD) |
| `runtime` | Inference runtime, KV cache, sampler |
| `loaders` | Model loaders (GGUF, SafeTensors, APG) |
| `quant` | Quantization engine |
| `ops` | Operations registry |
| `kernels` | Compute kernels (GEMM, norm, activation) |
| `public` | Public API |

## Development

```bash
mojo run test_apaga.mojo
```

## Version

0.1.0