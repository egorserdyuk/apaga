# Apaga - Universal Local AI Inference Library for Mojo

## Dev environment tips
- This is a Mojo project using Python 3.14+
- Mojo code lives in `.mojo` files across subdirectories (tensor, mem, hal, runtime, etc.)
- Use `mojo run test_apaga.mojo` to run the test file
- Check `__init__.mojo` files for module exports

## Project structure
- `tensor/` - Tensor, Shape, DType, View types
- `mem/` - Memory arena, weight buffers, pool
- `hal/` - Hardware abstraction layer (CPU, SIMD, thread pool)
- `runtime/` - Inference runtime (sampler, session, KV cache, plan)
- `loaders/` - Model loaders (GGUF, SafeTensors, APG)
- `quant/` - Quantization engine and parameters
- `ir/` - Intermediate representation and passes
- `ops/` - Operations registry and dispatch
- `kernels/` - Compute kernels (GEMM, norm, activation)
- `public/` - Public API (Model, Session, Generate)

## Testing
- Run tests with: `mojo run test_apaga.mojo`
- Tests are in `test_apaga.mojo` at the root

## PR instructions
- Title format: [apaga] <Title>
- Run `mojo run test_apaga.mojo` before committing