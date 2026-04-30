# apaga — Architecture Document

> **Universal Local AI Inference Library for Mojo**
> *OS-agnostic · Hardware-agnostic · Minimal dependencies · Maximum portability*

---

## Table of Contents

1. [Vision & Design Philosophy](#1-vision--design-philosophy)
2. [Lessons from Prior Art](#2-lessons-from-prior-art)
3. [Layer Map Overview](#3-layer-map-overview)
4. [Layer 0 — Hardware Abstraction (HAL)](#4-layer-0--hardware-abstraction-hal)
5. [Layer 1 — Memory Subsystem](#5-layer-1--memory-subsystem)
6. [Layer 2 — Tensor Engine](#6-layer-2--tensor-engine)
7. [Layer 3 — Kernel Library](#7-layer-3--kernel-library)
8. [Layer 4 — Operator Registry & Dispatch](#8-layer-4--operator-registry--dispatch)
9. [Layer 5 — Model IR & Graph Engine](#9-layer-5--model-ir--graph-engine)
10. [Layer 6 — Quantization Engine](#10-layer-6--quantization-engine)
11. [Layer 7 — Model Loaders](#11-layer-7--model-loaders)
12. [Layer 8 — Inference Runtime](#12-layer-8--inference-runtime)
13. [Layer 9 — Public API](#13-layer-9--public-api)
14. [Mojo-Specific Design Decisions](#14-mojo-specific-design-decisions)
15. [Portability Matrix](#15-portability-matrix)
16. [Dependency Policy](#16-dependency-policy)
17. [Performance Engineering Playbook](#17-performance-engineering-playbook)
18. [Testing & Validation Strategy](#18-testing--validation-strategy)
19. [Roadmap to Ollama-like System](#19-roadmap-to-ollama-like-system)
20. [Directory Structure](#20-directory-structure)
21. [Glossary](#21-glossary)

---

## 1. Vision & Design Philosophy

### 1.1 What apaga is

**apaga** is a purely local, hardware-agnostic AI inference engine written in Mojo. It runs neural network models — primarily transformer-based LLMs, but also vision, audio, and multimodal models — without depending on cloud services, Python runtimes, or vendor-specific SDKs at its core.

The name is meant to evoke something self-contained: it runs where it is, with what it has.

### 1.2 The four governing constraints

Every design decision must be evaluated against these four constraints in priority order:

```
1. CORRECTNESS   — numerically identical outputs to reference implementations
2. PORTABILITY   — compiles and runs on any OS/hardware target without code changes
3. PERFORMANCE   — within 2× of a tuned vendor-specific solution on that hardware
4. SIMPLICITY    — a single developer can understand any layer in one day
```

When constraints conflict, higher-numbered ones yield to lower-numbered ones. Performance never justifies breaking portability. Simplicity never justifies incorrect results.

### 1.3 What apaga is NOT

- Not a training framework. Backpropagation is out of scope.
- Not a cloud inference server. `apaga` is a library, not a daemon.
- Not a model hub. It loads models; it does not store or distribute them.
- Not a Python library. Python bindings are a convenience layer, not the foundation.

### 1.4 Design tenets

**Zero-cost abstractions.** Every abstraction in apaga must compile away. If a layer boundary introduces overhead at runtime, it is architecturally wrong.

**Explicit over implicit.** Memory allocation, device placement, and precision are always explicit decisions surfaced to the caller, never hidden inside library internals.

**Fail loudly at load time, never at inference time.** All shape checks, device compatibility checks, and quantization validation happen during model loading. Inference should never panic.

**Layered, not monolithic.** Each layer has a stable interface and can be replaced or tested independently. A user who only needs the tensor engine should be able to use it without pulling in the model loader.

**Additive extension, not subtractive compatibility.** New hardware backends, new model formats, and new quantization schemes are added by registering new implementations, never by forking or patching existing code paths.

---

## 2. Lessons from Prior Art

### 2.1 What ONNX gets right

- **Graph IR with typed, versioned operators.** Separating the model description from execution means the same `.onnx` file runs on CPU, GPU, or a new NPU without re-exporting.
- **Opset versioning.** Operators carry a version number so old models still parse correctly on new runtimes. apaga must do the same.
- **Shape inference as a first-class pass.** ONNX's shape inference pass runs before any kernel is called, surfacing shape errors early. apaga will require a shape propagation pass before any execution plan is built.

**What ONNX gets wrong (lessons for apaga):**

- ONNX's protobuf dependency makes it heavy and hard to parse in constrained environments. apaga will use a flat-binary IR (see §9).
- ONNX Runtime's plugin model is complex. apaga backend registration must be simpler.
- ONNX has no first-class quantization story at the IR level. apaga's IR has quantization metadata built in from day one.

### 2.2 What PyTorch gets right

- **Eager mode for debugging, compiled mode for deployment.** apaga mirrors this: a `debug` execution mode that runs ops one by one and validates shapes/values, and a `compiled` mode that fuses and optimizes.
- **Dispatch key table.** PyTorch's dispatcher selects the right kernel implementation based on device + dtype + layout. apaga's operator registry is a simplified version of this.
- **Memory format awareness (NCHW vs NHWC).** Tensor layout matters enormously for performance. apaga tensors carry their layout as a compile-time or runtime attribute.

**What PyTorch gets wrong:**

- The Python runtime dependency is the obvious one. apaga has none.
- PyTorch's `torch.compile` stack is a labyrinth. apaga's compilation pipeline must be readable.
- PyTorch's memory management is reference-counted with complex ownership in C++. Mojo's ownership model gives us better tools.

### 2.3 What llama.cpp gets right

- **Pure C/C++ with zero required external dependencies.** This is the gold standard for portability. apaga targets the same level.
- **Quantization-first design.** llama.cpp was built around quantized weights, not added later. apaga's tensor type system natively understands quantized types.
- **GGUF format.** A single flat-file format that is mmap-able, self-describing, and endian-aware. apaga's native model format (`.apg`) is spiritually similar.
- **K/V cache as a first-class object.** Not an implementation detail buried in a transformer class, but a named, managed resource.

**What llama.cpp gets wrong:**

- Operator implementations and the runtime are tightly coupled — hard to add a new backend without touching core code. apaga enforces a strict separation.
- No graph IR means no compiler-level optimizations (operator fusion, constant folding). apaga has a full graph pass pipeline.

### 2.4 What TensorFlow Lite / TFLite gets right

- **FlatBuffers for the model format.** Zero-copy deserialization. apaga's `.apg` format is FlatBuffers-inspired (though self-implemented to avoid the dependency).
- **Delegate model for hardware acceleration.** A clean "delegate" interface that any hardware vendor can implement. apaga's backend interface is directly inspired by this.
- **Selective build.** Only the operators you use are compiled in. apaga supports this via compile-time feature flags.

### 2.5 What MLC/TVM gets right

- **Operator fusion as a compiler pass.** Fusing element-wise ops into the preceding matmul is not a hand-coded trick but a systematic graph transformation. apaga will have a pass pipeline for this.
- **Targeting the hardware's native vector width automatically.** apaga's SIMD abstraction must auto-adapt to AVX-512, NEON, SVE, or whatever is available.

---

## 3. Layer Map Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         LAYER 9: PUBLIC API                          │
│              apaga.Session  ·  apaga.Model  ·  apaga.generate()      │
├─────────────────────────────────────────────────────────────────────┤
│                      LAYER 8: INFERENCE RUNTIME                      │
│         ExecutionPlan  ·  KVCache  ·  Sampler  ·  StreamWriter       │
├──────────────────────────────────┬──────────────────────────────────┤
│      LAYER 7: MODEL LOADERS      │    LAYER 6: QUANTIZATION ENGINE   │
│  GGUF · ONNX · SafeTensors · APG │  INT4/8 · GPTQ · AWQ · FP16/BF16 │
├──────────────────────────────────┴──────────────────────────────────┤
│                    LAYER 5: MODEL IR & GRAPH ENGINE                  │
│          GraphNode  ·  OpGraph  ·  Shape Inference  ·  Pass Pipeline │
├─────────────────────────────────────────────────────────────────────┤
│                  LAYER 4: OPERATOR REGISTRY & DISPATCH               │
│          OperatorDef  ·  KernelSelector  ·  DispatchTable            │
├─────────────────────────────────────────────────────────────────────┤
│                      LAYER 3: KERNEL LIBRARY                         │
│        GEMM · Attention · Norm · Activation · Conv · Embedding       │
├─────────────────────────────────────────────────────────────────────┤
│                      LAYER 2: TENSOR ENGINE                          │
│          DType  ·  Shape  ·  Stride  ·  TensorView  ·  Tensor        │
├─────────────────────────────────────────────────────────────────────┤
│                    LAYER 1: MEMORY SUBSYSTEM                         │
│       Arena  ·  Pool  ·  DeviceBuffer  ·  UnifiedMemory  ·  Mmap     │
├─────────────────────────────────────────────────────────────────────┤
│                  LAYER 0: HARDWARE ABSTRACTION (HAL)                 │
│         CPU (x86/ARM/RISC-V)  ·  GPU (CUDA/Metal/ROCm/Vulkan)        │
└─────────────────────────────────────────────────────────────────────┘
```

**Key rule:** Code in layer N may only call code in layers 0..N−1. No upward dependencies. This is enforced structurally by the module layout.

---

## 4. Layer 0 — Hardware Abstraction (HAL)

### 4.1 Purpose

The HAL is the only layer that knows what hardware the library is running on. Everything above it is hardware-agnostic. The HAL exposes a small, stable interface. It never leaks vendor types (`cudaStream_t`, `MTLDevice*`, etc.) into any higher layer.

### 4.2 CPU HAL

```
hal/
  cpu/
    caps.mojo          # Runtime CPU capability detection
    simd.mojo          # Portable SIMD width/type abstraction
    thread_pool.mojo   # Work-stealing thread pool
    affinity.mojo      # Core affinity and NUMA awareness
```

**CPU Capability Detection** runs once at startup and populates a `CPUCaps` struct:

```
struct CPUCaps:
    arch: CPUArch              # X86_64 | ARM64 | RISCV64 | WASM32
    vendor: CPUVendor          # Intel | AMD | Apple | Qualcomm | ...
    # x86
    has_avx2: Bool
    has_avx512f: Bool
    has_avx512_vnni: Bool
    has_amx_int8: Bool
    # ARM
    has_neon: Bool
    has_sve: Bool
    has_sve2: Bool
    has_dotprod: Bool
    has_i8mm: Bool
    # Common
    cache_line_bytes: Int       # Always 64 on modern hardware
    l1d_size_kb: Int
    l2_size_kb: Int
    l3_size_kb: Int
    physical_cores: Int
    logical_cores: Int
```

**SIMD Abstraction.** The kernel layer uses `SIMD[dtype, width]` natively in Mojo, but the HAL provides the canonical widths for each dtype on each CPU:

```
fn simd_width[dtype: DType]() -> Int:
    # Returns the native SIMD width for the best available ISA
    # e.g., 16 for Float32 on AVX-512, 8 on AVX2, 4 on NEON
```

**Thread Pool.** apaga ships with its own minimal work-stealing thread pool. It does not depend on OpenMP or TBB. It has three operations: `submit(task)`, `wait_all()`, and `shutdown()`. The number of worker threads defaults to physical core count and is configurable.

### 4.3 GPU HAL

```
hal/
  gpu/
    device.mojo        # Abstract Device interface
    stream.mojo        # Command stream / queue abstraction
    buffer.mojo        # GPU memory buffer
    event.mojo         # Synchronization primitives
    backends/
      cuda/            # NVIDIA GPU via libcuda.so (dlopen'd)
      metal/           # Apple GPU via Metal (macOS/iOS only)
      rocm/            # AMD GPU via libhip.so (dlopen'd)
      vulkan/          # Vulkan Compute fallback
      cpu_fallback/    # Pure CPU implementation of the Device interface
```

**Critical design decision: dlopen instead of link-time dependency.**

GPU backends are loaded at runtime via `dlopen`/`LoadLibrary`. If the CUDA runtime is not present, `apaga` silently falls back to the next available backend. This is how TensorFlow Lite's delegate model works and why llama.cpp can ship a single binary. There is **no compile-time dependency on CUDA, Metal, or ROCm headers** in the core library.

**The `Device` interface (simplified):**

```
trait Device:
    fn name(self) -> String
    fn allocate(self, bytes: Int, alignment: Int) -> DeviceBuffer raises
    fn free(self, buf: DeviceBuffer)
    fn copy_to_device(self, src: UnsafePointer[UInt8], dst: DeviceBuffer, bytes: Int) raises
    fn copy_to_host(self, src: DeviceBuffer, dst: UnsafePointer[UInt8], bytes: Int) raises
    fn synchronize(self) raises
    fn capabilities(self) -> DeviceCaps
```

**The `DeviceCaps` struct:**

```
struct DeviceCaps:
    device_type: DeviceType    # CPU | CUDA | Metal | ROCm | Vulkan
    total_memory_bytes: Int
    compute_units: Int
    max_shared_memory_bytes: Int
    supports_fp16: Bool
    supports_bf16: Bool
    supports_int8: Bool
    supports_int4: Bool
    warp_size: Int             # 32 for NVIDIA, 64 for AMD, 1 for CPU
```

### 4.4 Backend selection policy

At startup, apaga enumerates all backends in this order and selects the best available:

```
1. CUDA (if NVIDIA GPU detected and libcuda present)
2. Metal (if Apple Silicon / AMD GPU on macOS)
3. ROCm (if AMD GPU detected and libhip present)
4. Vulkan Compute (if Vulkan loader present)
5. CPU (always available, never fails)
```

The user can override via `APAGA_DEVICE=cpu|cuda:0|metal:0|rocm:0` environment variable or the API.

---

## 5. Layer 1 — Memory Subsystem

### 5.1 The fundamental problem

Inference has a predictable memory access pattern: weights are loaded once and read many times; activations are allocated and freed in a stack-like LIFO pattern; the KV cache grows incrementally. Generic heap allocators (malloc) are bad at all three of these patterns. apaga has three specialized allocators.

### 5.2 Arena Allocator

Used for activations during a forward pass. An arena allocates a large slab of memory upfront and gives out sub-slices with pointer bump. Freeing is O(1): reset the bump pointer. No individual `free()` calls.

```
struct Arena:
    var base: UnsafePointer[UInt8]
    var offset: Int
    var capacity: Int
    var device: Device

    fn allocate(inout self, bytes: Int, alignment: Int = 64) -> UnsafePointer[UInt8] raises
    fn reset(inout self)   # Free everything at once
    fn checkpoint(self) -> Int   # Save current offset
    fn restore(inout self, checkpoint: Int)  # Roll back to checkpoint
```

The arena is sized at model load time based on the max activation footprint computed during the shape propagation pass.

### 5.3 Pool Allocator

Used for tensors whose lifetimes are less predictable than arena-style. Maintains per-size free lists. Objects are returned to the pool, not to the OS. This eliminates fragmentation for the common case of repeated same-shape tensor allocations across inference calls.

### 5.4 Weight Buffer (mmap-based)

Model weights are large and read-only. They are loaded via `mmap`, which lets the OS page them in on demand and page them out under memory pressure. This is the same approach GGUF uses. The weight buffer is not an allocator; it is a view over an mmap'd file.

```
struct WeightBuffer:
    fn map(path: Path, offset: Int, size: Int) -> WeightBuffer raises
    fn as_ptr(self) -> UnsafePointer[UInt8]  # Always aligned to 64B
    fn unmap(owned self)
```

**Critical rule:** Weights are never copied into heap memory on load. They stay mmap'd for the lifetime of the model. This allows loading 70B models on machines with less RAM than the model's file size, relying on OS virtual memory.

### 5.5 Unified Memory (for GPU backends)

When running on a CUDA or Metal GPU, some allocations need to be accessible from both CPU and GPU without explicit copies (e.g., sampling logic that runs on CPU uses logits computed on GPU). The HAL's unified memory allocation path handles this:

```
struct UnifiedBuffer:
    fn allocate(device: Device, bytes: Int) -> UnifiedBuffer raises
    fn cpu_ptr(self) -> UnsafePointer[UInt8]
    fn gpu_ptr(self) -> DeviceBuffer
    fn prefetch_to_device(self) raises
    fn prefetch_to_host(self) raises
```

### 5.6 Memory lifecycle rules

| Allocation type | Allocator | Lifetime |
|---|---|---|
| Model weights | WeightBuffer (mmap) | Model lifetime |
| KV cache | Pool | Session lifetime |
| Activations (forward pass) | Arena | Single forward pass |
| Sampler output | UnifiedBuffer | Single token |
| Runtime metadata | System heap | Process lifetime |

---

## 6. Layer 2 — Tensor Engine

### 6.1 DType

apaga's dtype system covers all precisions used in local inference, with precise semantics:

```
@value
struct DType:
    # Floating point
    Float32    # IEEE 754 single
    Float16    # IEEE 754 half
    BFloat16   # Google Brain float
    Float8E4M3 # FP8, NV/ARM format (emerging)
    Float8E5M2 # FP8, exponent-heavy

    # Integer (for quantized weights)
    Int32
    Int16
    Int8
    Int4       # Packed: 2 values per byte
    Int2       # Packed: 4 values per byte (experimental)
    UInt8
    UInt4      # Packed

    # Special
    Bool
```

**Int4 and Int2 packing** are not primitive types in hardware. They are a storage format: two Int4 values packed into one UInt8. The kernel layer handles pack/unpack transparently.

### 6.2 Shape and Stride

```
struct Shape:
    var dims: InlineArray[Int, MAX_DIMS]  # MAX_DIMS = 8, stack-allocated
    var ndim: Int

    fn numel(self) -> Int
    fn is_contiguous(self, strides: Stride) -> Bool
    fn broadcast_to(self, other: Shape) -> Shape raises
    fn __eq__(self, other: Shape) -> Bool
```

```
struct Stride:
    var data: InlineArray[Int, MAX_DIMS]
    var ndim: Int

    @staticmethod
    fn row_major(shape: Shape) -> Stride      # C-order
    @staticmethod
    fn col_major(shape: Shape) -> Stride      # Fortran-order
    @staticmethod
    fn nhwc(shape: Shape) -> Stride           # For vision models
```

Both `Shape` and `Stride` are stack-allocated with a maximum of 8 dimensions. No heap allocation for tensor metadata. This is a hard limit that dramatically simplifies memory management.

### 6.3 TensorView (non-owning)

`TensorView` is the fundamental unit of computation. It is a fat pointer: a pointer to data plus metadata. It does **not** own the data.

```
struct TensorView[dtype: DType]:
    var ptr: UnsafePointer[Scalar[dtype]]
    var shape: Shape
    var stride: Stride
    var device: DeviceID

    fn is_contiguous(self) -> Bool
    fn numel(self) -> Int
    fn element_at(self, *indices: Int) -> Scalar[dtype]  # Debug only
    fn slice(self, dim: Int, start: Int, end: Int) -> TensorView[dtype]
    fn reshape(self, new_shape: Shape) -> TensorView[dtype] raises
    fn as_contiguous(self, arena: Arena) -> TensorView[dtype]  # Copies if needed
```

**The rule:** All kernel functions take and return `TensorView`. They never take ownership of data.

### 6.4 Tensor (owning)

`Tensor` wraps a `TensorView` and adds lifetime management. Its backing memory comes from one of the allocators in Layer 1.

```
struct Tensor[dtype: DType]:
    var view: TensorView[dtype]
    var _allocator: AllocatorRef   # Reference to the allocator that owns the data

    @staticmethod
    fn from_arena(arena: Arena, shape: Shape) -> Tensor[dtype] raises
    @staticmethod
    fn from_pool(pool: Pool, shape: Shape) -> Tensor[dtype] raises
    @staticmethod
    fn from_weight_buffer(wb: WeightBuffer, offset: Int, shape: Shape) -> Tensor[dtype]

    fn __del__(owned self):
        # Returns memory to pool if pool-allocated; no-op for arena/mmap
```

### 6.5 Type coercion and precision promotion

apaga has explicit rules for type promotion, unlike NumPy's implicit widening:

```
fn promote_dtype(a: DType, b: DType) -> DType:
    # Never implicitly narrows precision
    # Int4 op Float16 -> Float16
    # Float16 op Float32 -> Float32
    # BFloat16 op Float16 -> Float32  (both get widened)
```

Kernels declare their accepted input dtypes explicitly. The dispatch system (Layer 4) inserts cast operations automatically when the input dtype does not match the kernel's native dtype.

---

## 7. Layer 3 — Kernel Library

### 7.1 Structure

```
kernels/
  gemm/
    gemm_f32.mojo       # Reference impl
    gemm_f16.mojo
    gemm_bf16.mojo
    gemm_int8.mojo
    gemm_int4.mojo      # Dequantize-then-multiply
    gemm_avx512.mojo    # x86 AVX-512 fast path
    gemm_neon.mojo      # ARM NEON fast path
    gemm_cuda.mojo      # CUDA kernel launcher (FFI)
    gemm_metal.mojo     # Metal kernel launcher (FFI)
  attention/
    sdpa.mojo           # Scaled dot-product attention (reference)
    sdpa_flash.mojo     # FlashAttention-2 style (tiled)
    sdpa_causal.mojo    # Causal mask optimization
    sdpa_grouped.mojo   # Grouped Query Attention (GQA)
    sdpa_sliding.mojo   # Sliding window attention
  norm/
    rms_norm.mojo
    layer_norm.mojo
    group_norm.mojo
  activation/
    silu.mojo
    gelu.mojo
    relu.mojo
    geglu.mojo
  embedding/
    embed.mojo          # Token embedding lookup
    rope.mojo           # Rotary Position Embeddings
    alibi.mojo          # ALiBi position bias
  conv/
    conv1d.mojo
    conv2d.mojo
    depthwise_conv2d.mojo
  reduce/
    softmax.mojo
    topk.mojo
    argmax.mojo
  elementwise/
    add.mojo
    mul.mojo
    cast.mojo
```

### 7.2 Kernel contract

Every kernel function follows a strict contract:

1. **Pure function.** No global state. No allocation (all memory pre-allocated, passed in).
2. **TensorView in, TensorView out.** Memory ownership is never transferred.
3. **Shape-validated at compile time where possible, runtime where not.**
4. **A reference implementation always exists** (usually `_f32.mojo`) that is correct but not necessarily fast. Fast paths are additions on top.
5. **Kernels declare their supported dtypes** in a static metadata struct.

```
struct KernelMeta:
    var name: StringLiteral
    var supported_dtypes: StaticTuple[DType, MAX_DTYPES]
    var supported_devices: StaticTuple[DeviceType, MAX_DEVICES]
    var min_alignment: Int
```

### 7.3 GEMM — the most important kernel

GEMM (General Matrix-Matrix Multiply) is the dominant operation in transformer inference. Its performance determines the overall throughput.

apaga uses a **micro-kernel tiling** strategy (the same as BLIS):

```
GEMM(A[M,K], B[K,N]) -> C[M,N]:
  Partition M, N, K into tiles: MC×KC, KC×NC
  Pack A tile into L2-local buffer (row-major, aligned)
  Pack B tile into L1-local buffer (column-major for SIMD)
  Inner micro-kernel: register-tiled accumulate
    MR×NR tiles (e.g., 6×16 for AVX-512 Float32)
    Fully unrolled, all values in registers
```

The tile sizes `MC`, `NC`, `KC`, `MR`, `NR` are chosen based on `CPUCaps.l1d_size_kb` and `l2_size_kb` at runtime. There is no hardcoded cache size.

### 7.4 Attention — FlashAttention-2 style

Standard attention materializes the full attention score matrix `[seq_len, seq_len]`, which is O(seq_len²) memory. For 128K context windows this is prohibitive.

apaga implements tiled attention (FlashAttention-2 algorithm):

```
SDPA_Flash(Q[B,H,S,D], K[B,H,S,D], V[B,H,S,D]) -> O[B,H,S,D]:
  For each query tile Q_tile[BLOCK_Q, D]:
    For each key-value tile K_tile[BLOCK_KV, D], V_tile[BLOCK_KV, D]:
      scores = Q_tile @ K_tile.T / sqrt(D)
      Apply causal mask if required
      Running softmax (numerically stable, online update)
      Accumulate O_tile
  Finalize normalization
```

Memory: O(BLOCK_Q × BLOCK_KV) per head per thread, not O(S²). BLOCK_Q and BLOCK_KV default to 64 and are tunable.

### 7.5 Rotary Position Embeddings (RoPE)

RoPE is the dominant position encoding in modern LLMs (LLaMA, Mistral, Gemma). It is computed on the fly during attention, not stored as a lookup table. apaga applies RoPE in the attention kernel to avoid materializing the rotated Q/K tensors:

```
rope(x[S, D], position_ids[S], theta: Float32) -> x_rotated[S, D]:
  For each position i in [0, S):
    For each dimension pair (d, d+D/2):
      freq = 1 / (theta ^ (2*d / D))
      angle = position_ids[i] * freq
      x_rotated[i, d]     = x[i, d] * cos(angle) - x[i, d+D/2] * sin(angle)
      x_rotated[i, d+D/2] = x[i, d] * sin(angle) + x[i, d+D/2] * cos(angle)
```

RoPE scaling (for extended context: YaRN, dynamic NTK) is a parameterized variant of the above and is handled via a `RoPEConfig` struct passed to the kernel.

---

## 8. Layer 4 — Operator Registry & Dispatch

### 8.1 The dispatch problem

The same logical operation (e.g., "matrix multiply") has many implementations:
- Float32 on CPU with AVX-512
- Float16 on CPU with AVX-512 + FP16C
- Int4 dequantize-and-multiply on CPU
- Float16 on CUDA
- BFloat16 on Metal

The dispatch layer selects the right implementation at runtime without the caller knowing about the alternatives.

### 8.2 OperatorKey

```
struct OperatorKey:
    var op_name: StringLiteral   # e.g., "matmul", "rms_norm", "rope"
    var dtype: DType
    var device: DeviceType
    var layout: TensorLayout     # ROW_MAJOR | COL_MAJOR | NHWC | ...

    fn __hash__(self) -> UInt64
    fn __eq__(self, other: OperatorKey) -> Bool
```

### 8.3 KernelFn type

```
alias KernelFn = fn(
    inputs: Span[TensorView[DType]],
    outputs: Span[TensorView[DType]],
    attrs: AttributeMap,
    ctx: KernelContext,
) raises -> None
```

`KernelContext` carries: the current `Device`, an `Arena` for temporary scratch allocations, and a `StreamHandle` for GPU async dispatch.

### 8.4 OperatorRegistry

```
struct OperatorRegistry:
    var table: HashMap[OperatorKey, KernelFn]

    fn register(inout self, key: OperatorKey, fn: KernelFn)
    fn dispatch(self, key: OperatorKey) -> KernelFn raises
    fn best_for(self, op_name: StringLiteral, dtype: DType, device: DeviceType) -> KernelFn raises
```

`best_for` implements the fallback chain:

```
1. Exact match: (op, dtype, device)
2. Dtype widening: (op, promote(dtype), device)   — e.g., Int4 → Int8
3. Device fallback: (op, dtype, CPU)
4. Reference: (op, Float32, CPU)
5. Error: no kernel found
```

### 8.5 Auto-cast insertion

When the graph engine (Layer 5) resolves operators, it checks whether the declared edge dtype matches the kernel's required dtype. If not, it inserts a `cast` operation automatically. This happens once at plan-build time, not at each inference call.

---

## 9. Layer 5 — Model IR & Graph Engine

### 9.1 The graph as the unit of deployment

A model in apaga is a directed acyclic graph (DAG) of operators. Weights are constant tensors in this graph. The graph is the intermediate representation between the model loader (Layer 7) and the inference runtime (Layer 8).

### 9.2 GraphNode

```
struct GraphNode:
    var id: NodeID
    var op_name: StringLiteral
    var attrs: AttributeMap         # e.g., {n_heads: 32, head_dim: 128}
    var input_edges: List[EdgeID]
    var output_edges: List[EdgeID]
    var inplace_pairs: List[(Int, Int)]   # (input_idx, output_idx) pairs that alias
```

### 9.3 EdgeID and TypedEdge

```
struct TypedEdge:
    var id: EdgeID
    var dtype: DType
    var shape: Shape               # Known after shape propagation
    var is_constant: Bool          # True for weight tensors
    var constant_data: Optional[WeightRef]  # If is_constant
```

### 9.4 OpGraph

```
struct OpGraph:
    var nodes: List[GraphNode]
    var edges: List[TypedEdge]
    var inputs: List[EdgeID]       # Graph input placeholders
    var outputs: List[EdgeID]      # Graph output edges
    var metadata: ModelMetadata    # Name, version, author, license

    fn topological_order(self) -> List[NodeID] raises
    fn validate(self) raises        # Type check, no cycles, no dangling edges
```

### 9.5 Pass Pipeline

Before the graph is handed to the runtime, it runs through a configurable sequence of transformation passes. Each pass takes an `OpGraph` and returns a modified `OpGraph`.

**Built-in passes (run in this order):**

```
1. ShapeInferencPass
   Propagates concrete shapes through the graph starting from known input shapes.
   After this pass, every edge in the graph has a fully resolved Shape.
   Errors on unsatisfiable shapes.

2. ConstantFoldingPass
   Evaluates subgraphs where all inputs are constants at load time.
   E.g., precomputes RoPE cos/sin tables, position embeddings.

3. DeadNodeEliminationPass
   Removes nodes whose outputs are never consumed.

4. DtypeInsertionPass
   Inserts Cast nodes wherever edge dtype doesn't match kernel requirements.
   Uses the dispatch table's fallback chain to determine the required dtype.

5. OperatorFusionPass
   Merges element-wise sequences into a single fused kernel.
   Also fuses: matmul + bias + activation (common in attention/MLP layers).
   Produces FusedNode entries with a generated fused kernel.

6. MemoryPlanningPass
   Assigns an arena offset to every non-constant edge.
   Maximizes buffer reuse via liveness analysis.
   Output: activation arena size needed for a single forward pass.

7. ExecutionOrderPass
   Determines the final execution order, considering:
   - Topological constraints
   - Cache locality (schedule ops that share live data together)
   - Parallelism opportunities (mark independent op sequences)
```

Passes can be selectively disabled for debugging:

```
var config = PassConfig {
    .enable_fusion = True,
    .enable_constant_folding = True,
    .enable_memory_planning = True,
    .debug_dump_after_each_pass = False,
}
```

### 9.6 The `.apg` Native Format

apaga has a native binary model format for distribution. It is deliberately simple.

**File layout:**

```
[MAGIC: 8 bytes "APAGAMDL"]
[VERSION: u32]
[HEADER_SIZE: u64]           ← byte offset of start of header section
[WEIGHTS_OFFSET: u64]        ← byte offset of start of weights section
[GRAPH_OFFSET: u64]          ← byte offset of start of graph section

[HEADER SECTION]             ← JSON metadata (model name, arch, quant info, etc.)
[GRAPH SECTION]              ← Serialized OpGraph (custom flat binary)
[WEIGHTS SECTION]            ← Packed weight tensors, 64-byte aligned
```

Key properties:
- The weights section is mmap-able directly: tensors in the weights section need not be copied.
- The graph section is read entirely into memory (it is small: usually <1MB).
- The header is JSON so it can be inspected with standard tools (`cat model.apg | head -c 1024`).
- Multi-file models (sharded) are represented as multiple `.apg` files sharing the same header.

---

## 10. Layer 6 — Quantization Engine

### 10.1 Why quantization is not an afterthought

A 70B parameter model at Float32 requires 280 GB of memory. At Int4 it requires 35 GB. Quantization is not an optimization; it is a requirement for local inference of frontier models.

apaga treats quantized dtypes as first-class citizens in the type system, not as post-hoc compression applied to a Float32 tensor.

### 10.2 Quantization schemes

apaga supports the following schemes:

| Scheme | Description | Primary use |
|---|---|---|
| **RTN** | Round-to-nearest, per-channel scale | Baseline, always available |
| **GPTQ** | Gradient-based PTQ, per-group scale+zero | LLM weights, 4-bit |
| **AWQ** | Activation-aware weight quantization | LLM weights, 4-bit, better accuracy |
| **SmoothQuant** | Activation quantization (W8A8) | Int8 activations |
| **GGUF Q4_K_M** | llama.cpp k-quants | GGUF format compatibility |
| **BitsAndBytes NF4** | NormalFloat4, used by QLoRA | HuggingFace compat |

### 10.3 QuantParams (per-tensor metadata)

```
struct QuantParams:
    var scheme: QuantScheme
    var group_size: Int           # 32, 64, 128 (or -1 for per-channel)
    var scales: TensorView[Float16]   # Shape: [n_groups] or [n_channels]
    var zeros: TensorView[Float16]    # Shape: [n_groups] or [n_channels] (may be None)
    var bits: Int                 # 4 or 8
```

`QuantParams` live alongside the weight tensor in the graph. The GEMM kernel that operates on Int4 weights receives `QuantParams` as a kernel attribute.

### 10.4 Dequantization strategies

There are two places dequantization can happen:

**Strategy A: Dequantize-on-the-fly (preferred for memory-bound scenarios)**
The GEMM kernel dequantizes Int4→Float16 in registers, immediately before multiplying. No full dequantized copy is ever materialized. This is the approach llama.cpp uses and is optimal for memory-bandwidth-bound regimes.

**Strategy B: Pre-dequantize (preferred for compute-bound scenarios)**
A separate `dequantize` operator runs before GEMM, producing a full Float16 weight tensor. This is better when the weight will be multiplied many times (large batch sizes). apaga's `OperatorFusionPass` automatically decides between A and B based on the static batch size hint in the `SessionConfig`.

### 10.5 Quantization of activations (W8A8)

Weight-only quantization (W4A16, W8A16) is the default. Full quantization of activations (W8A8) provides additional speedup on hardware with efficient Int8 GEMM (e.g., NVIDIA Ampere and later with tensor cores, Apple ANE, Qualcomm HTP).

Activation quantization requires per-token dynamic scale computation: the scale is computed from the max absolute value of each activation row. This is a fast O(N) scan that runs before the quantized GEMM.

---

## 11. Layer 7 — Model Loaders

### 11.1 Loader interface

```
trait ModelLoader:
    fn can_load(path: Path) -> Bool  # Static method
    fn load(path: Path, config: LoadConfig) -> OpGraph raises
```

`LoadConfig` specifies: target dtype (override the stored dtype), device hint, memory budget.

### 11.2 GGUF Loader

GGUF is the format used by llama.cpp and Ollama. It is the most important format to support for immediate compatibility with the existing local AI ecosystem.

The GGUF loader:
1. Reads the GGUF header to extract model metadata and hyperparameters.
2. Maps the file into memory (mmap).
3. Constructs the `OpGraph` from the architecture name (e.g., `llama`, `mistral`, `phi3`) and the hyperparameters — **GGUF does not store a graph, only weights and metadata**, so apaga maintains a registry of architecture-to-graph constructors.
4. Attaches weight tensors as constant edges in the graph.
5. Translates GGUF quantization types (Q4_K, Q6_K, etc.) into apaga `QuantParams`.

```
architectures/
  llama.mojo      # LLaMA 1/2/3 family
  mistral.mojo    # Mistral 7B / Mixtral 8x7B
  phi3.mojo       # Microsoft Phi-3
  gemma.mojo      # Google Gemma
  qwen2.mojo      # Alibaba Qwen2
  starcoder.mojo  # BigCode StarCoder
```

Each architecture file is a function `build_graph(meta: GGUFMeta) -> OpGraph`.

### 11.3 ONNX Loader

The ONNX loader parses `.onnx` protobuf files and constructs an `OpGraph`. This handles the non-LLM model zoo (vision, audio, classification).

**Protobuf dependency problem:** ONNX files are protobuf-encoded. apaga ships a minimal self-contained protobuf3 wire-format decoder (~400 lines of Mojo) that handles the subset of protobuf used by ONNX, without depending on the protobuf library.

Unsupported ONNX opsets are reported as load-time errors with an actionable message (e.g., "opset 18 operator `GridSample` is not implemented; see https://github.com/... to contribute").

### 11.4 SafeTensors Loader

HuggingFace SafeTensors format: JSON header + flat binary tensor data. Simple to implement and safe (no arbitrary code execution on load, unlike Pickle). Used for loading HuggingFace models with a separate `config.json`.

The SafeTensors loader reads `config.json` to determine the architecture, then uses the same architecture registry as the GGUF loader to construct the graph.

### 11.5 APG Loader (native)

The native `.apg` format is loaded directly: the graph section is deserialized into an `OpGraph`, and the weights section is mmap'd. This path is the fastest and the one used by the future Ollama-like system after a one-time conversion step.

---

## 12. Layer 8 — Inference Runtime

### 12.1 ExecutionPlan

After the graph has passed through all transformation passes, it is compiled into an `ExecutionPlan`: a flat, ordered list of `DispatchedOp` structs that can be executed with minimal overhead.

```
struct DispatchedOp:
    var kernel: KernelFn            # Direct function pointer, no dynamic dispatch
    var inputs: InlineArray[TensorRef, MAX_OP_INPUTS]
    var outputs: InlineArray[TensorRef, MAX_OP_OUTPUTS]
    var attrs: AttributeMap
    var thread_group: Int           # For parallel dispatch
```

`TensorRef` is either a direct pointer (for weights, which are fixed) or an arena offset (for activations, resolved at inference time).

Building an `ExecutionPlan` from an `OpGraph` is expensive (it runs all passes). It happens once at model load time. Inference calls run the pre-built plan.

### 12.2 KVCache

The KV cache is the key data structure for efficient autoregressive generation. Without it, the attention operation would re-compute all past key/value projections on every new token.

```
struct KVCache:
    var keys: Tensor[Float16]     # Shape: [n_layers, max_seq_len, n_kv_heads, head_dim]
    var values: Tensor[Float16]   # Shape: [n_layers, max_seq_len, n_kv_heads, head_dim]
    var filled: Int               # How many positions are currently filled
    var max_seq_len: Int

    fn append(inout self, layer: Int, k: TensorView[Float16], v: TensorView[Float16]) raises
    fn get(self, layer: Int, seq_len: Int) -> (TensorView[Float16], TensorView[Float16])
    fn clear(inout self)
    fn can_fit(self, new_tokens: Int) -> Bool
```

**KVCache memory layout choices:**

- Separate key and value tensors (vs interleaved) improves hardware prefetcher performance during attention.
- `Float16` storage regardless of the model's weight dtype — KV caches are activations, not weights; they don't need the same compression.
- Paged KV cache (future): for multi-request serving, KV cache pages can be shared across requests with the same prefix (system prompt). This is `PagedAttention` (vLLM). It is out of scope for v1 but the `KVCache` interface is designed to allow it.

### 12.3 Session

```
struct Session:
    var model: Arc[OpGraph]
    var plan: ExecutionPlan
    var kv_cache: KVCache
    var activation_arena: Arena
    var device: Device
    var config: SessionConfig

    fn generate(
        inout self,
        tokens: Span[Int32],
        config: GenerateConfig,
        on_token: fn(Int32, Float32) -> Bool,  # token, logprob; return False to stop
    ) raises

    fn embed(inout self, tokens: Span[Int32]) -> Tensor[Float16] raises
    fn logits(inout self, tokens: Span[Int32]) -> Tensor[Float32] raises
    fn reset(inout self)   # Clear KV cache
```

`SessionConfig` includes:
- `max_seq_len: Int` — determines KV cache size
- `n_threads: Int` — overrides thread pool size
- `batch_size_hint: Int` — informs fusion decisions (Strategy A vs B for dequant)
- `compute_dtype: DType` — override compute precision (e.g., force Float32 for highest accuracy)

### 12.4 Sampler

Sampling is intentionally separated from the forward pass. The forward pass produces logits; the sampler produces the next token.

```
struct Sampler:
    var config: SamplerConfig

    fn sample(self, logits: TensorView[Float32], rng: inout RNG) -> (Int32, Float32) raises
```

```
struct SamplerConfig:
    var temperature: Float32      # 0.0 = greedy, 1.0 = no scaling
    var top_p: Float32            # Nucleus sampling threshold
    var top_k: Int                # Top-k filtering (0 = disabled)
    var repetition_penalty: Float32
    var vocab_size: Int
```

The sampler runs on the CPU regardless of where the forward pass ran. Logits are copied from GPU to CPU (via unified memory or explicit copy) before sampling.

### 12.5 Streaming / token callback

apaga does not have a concept of "the full generated string." It operates token-by-token. The `on_token` callback in `Session.generate` is called after each token is sampled. The caller is responsible for converting token IDs to text using the tokenizer.

The tokenizer is **not** part of apaga core. It is a separate, optional module (`apaga.tokenizer`) that wraps sentencepiece or tiktoken via FFI for completeness, but apaga's core can be used without it.

---

## 13. Layer 9 — Public API

### 13.1 Design goals

The public API must be:
- **Three lines to run a model.** A user should not need to understand any lower layer.
- **Explicit about resources.** Session creation, device selection, and memory limits are visible.
- **Extensible.** A power user can drop down to any lower layer.

### 13.2 High-level API

```mojo
from apaga import Model, Session, GenerateConfig

fn main() raises:
    # Load model (runs all passes, builds ExecutionPlan)
    let model = Model.from_file("llama3-8b.gguf")

    # Create a session (allocates KV cache, arena)
    var session = Session(model, max_seq_len=4096)

    # Generate
    let config = GenerateConfig(temperature=0.8, top_p=0.95, max_tokens=512)
    session.generate(
        tokens=[1, 15043, 3186],  # "<s> Hello world"
        config=config,
        on_token=fn(token: Int32, logprob: Float32) -> Bool:
            print(token, end="")
            return True  # continue
    )
```

### 13.3 Configuration hierarchy

```
LoadConfig         ← How to load and transform the model file
  ├── target_dtype        (override weight dtype)
  ├── device_hint         (preferred backend)
  └── pass_config         (which passes to run)

SessionConfig      ← Resource allocation for a session
  ├── max_seq_len
  ├── n_threads
  └── compute_dtype

GenerateConfig     ← Parameters for a single generation call
  ├── SamplerConfig
  ├── max_tokens
  └── stop_tokens
```

### 13.4 Error handling

apaga uses Mojo's `raises` mechanism. Errors propagate via `raises`; there are no silent failures. Error types:

```
ApagaError (base)
├── LoadError         ← File not found, unsupported format, corrupt data
├── ShapeError        ← Incompatible shapes during inference
├── DeviceError       ← GPU OOM, device lost, unsupported operation
├── QuantError        ← Unsupported quantization scheme
└── ConfigError       ← Invalid configuration values
```

All errors include a `context` chain showing the call stack within apaga, plus a `suggestion` field for common mistakes.

### 13.5 Python bindings

Mojo's Python interop allows exposing apaga as a Python package with zero additional wrapper code for most cases. The bindings module (`apaga/python/`) wraps the Mojo structs in Python-friendly classes:

```python
import apaga

model = apaga.Model.from_file("llama3-8b.gguf")
session = apaga.Session(model, max_seq_len=4096)
for token, logprob in session.stream("Hello, world!", temperature=0.8):
    print(token, end="", flush=True)
```

The Python API is a thin convenience wrapper. All computation happens in Mojo.

---

## 14. Mojo-Specific Design Decisions

### 14.1 Ownership and lifetimes

Mojo's ownership system maps directly onto apaga's memory model:

- `WeightBuffer` is always passed by `borrowed` reference — it is never moved or copied.
- `Arena` is `inout` in all kernel calls — kernels may advance its bump pointer.
- `KVCache` is `inout` in `Session.generate` — it is modified in place.
- `TensorView` is a `@register_passable` value type — cheap to copy (it's just a fat pointer).

This eliminates entire classes of bugs that plague C++ inference engines (use-after-free of weight buffers, double-free of GPU allocations).

### 14.2 Compile-time specialization with `@parameter`

Mojo's `@parameter` decorator enables zero-overhead dispatch for dtype and device at compile time in hot paths:

```mojo
@parameter
fn gemm_dispatch[dtype: DType, device: DeviceType](
    A: TensorView[dtype], B: TensorView[dtype], C: TensorView[dtype]
):
    @parameter
    if dtype == DType.float32 and device == DeviceType.CPU:
        gemm_f32_avx512(A, B, C)
    elif dtype == DType.float16 and device == DeviceType.CUDA:
        gemm_f16_cublas(A, B, C)
    # ... etc.
```

When the dtype and device are known at compile time (they usually are for the hot inner loop of a pre-built execution plan), this compiles to a direct function call with no branch overhead.

### 14.3 SIMD in kernels

Mojo's first-class SIMD support makes the kernel implementations both readable and fast:

```mojo
fn rms_norm[dtype: DType, width: Int](
    x: TensorView[dtype], weight: TensorView[dtype], eps: Float32
) -> TensorView[dtype]:
    let n = x.shape.dims[0]
    var sum_sq = SIMD[Float32, width](0)
    for i in range(0, n, width):
        let v = x.load[width](i).cast[Float32]()
        sum_sq += v * v
    let rms = rsqrt(sum_sq.reduce_add() / n + eps)
    for i in range(0, n, width):
        let v = x.load[width](i).cast[Float32]()
        result.store[width](i, (v * rms).cast[dtype]())
```

This is readable as scalar code but compiles to fully vectorized SIMD.

### 14.4 Compile-time feature flags

apaga uses Mojo's conditional compilation to exclude GPU backends on platforms where they are irrelevant:

```mojo
@parameter
if has_cuda:
    from apaga.hal.gpu.backends.cuda import CUDADevice
    registry.register_backend(CUDADevice())
```

A `--no-gpu` build flag produces a binary with no GPU code at all, suitable for embedded or server-without-GPU deployments.

### 14.5 Interoperability with C

Many high-performance kernels (cuBLAS, Apple Accelerate, Intel oneMKL) are C libraries. Mojo's C FFI is used to call these when they provide better performance than apaga's own kernels:

```mojo
from sys.ffi import external_call

fn cublas_sgemm(...):
    external_call["cublasSgemm_v2", NoneType](handle, ...)
```

The FFI call is wrapped in a kernel that matches apaga's `KernelFn` signature. The surrounding infrastructure (dispatch, type checking, arena management) is pure Mojo.

---

## 15. Portability Matrix

| Platform | CPU Backend | GPU Backend | Status |
|---|---|---|---|
| Linux x86_64 | ✅ AVX2 / AVX-512 | ✅ CUDA (dlopen) | Tier 1 |
| Linux ARM64 | ✅ NEON / SVE | ✅ CUDA Jetson (dlopen) | Tier 1 |
| macOS ARM64 (Apple Silicon) | ✅ NEON / AMX | ✅ Metal | Tier 1 |
| macOS x86_64 | ✅ AVX2 | ⚠️ Metal (AMD eGPU only) | Tier 2 |
| Windows x86_64 | ✅ AVX2 / AVX-512 | ✅ CUDA (dlopen) | Tier 2 |
| Windows ARM64 | ✅ NEON | ⬜ DirectML (future) | Tier 3 |
| Linux RISC-V | ✅ scalar | ⬜ none | Tier 3 |
| WebAssembly | ✅ WASM SIMD | ⬜ WebGPU (future) | Experimental |
| Android ARM64 | ✅ NEON / dotprod | ⬜ Vulkan (future) | Experimental |

**Tier definitions:**
- **Tier 1:** Actively tested in CI, all features supported, performance tuned.
- **Tier 2:** Tested on PR merge, functional correctness guaranteed, performance not tuned.
- **Tier 3:** Best-effort, contributed by community, scalar fallback only.
- **Experimental:** Not in CI, may not compile.

---

## 16. Dependency Policy

### 16.1 Zero mandatory runtime dependencies

apaga's core (Layers 0–8) compiles and runs with **zero external runtime dependencies**. The standard library and the OS are not "dependencies" in this sense.

This is a hard rule, not a goal. Every proposed dependency must pass the following test: *Can the person using apaga on a fresh Linux install with only the Mojo compiler reproduce a working inference run without installing anything else?* If no, the dependency is not allowed in core.

### 16.2 Allowed optional dependencies (via dlopen/LoadLibrary)

These are loaded at runtime if present. Their absence is not an error — apaga falls back gracefully:

| Library | Purpose | Fallback |
|---|---|---|
| `libcuda.so` | CUDA GPU compute | CPU backend |
| `libcublas.so` | cuBLAS GEMM | apaga's own GEMM |
| `Metal.framework` | Apple GPU compute | CPU backend |
| `libhip.so` | ROCm GPU compute | CPU backend |
| `libvulkan.so` | Vulkan compute | CPU backend |
| `libaccelerateffi.dylib` | Apple Accelerate BLAS | apaga's own GEMM |
| `libmkl_rt.so` | Intel MKL | apaga's own GEMM |

### 16.3 Optional loader dependencies

The ONNX loader requires parsing protobuf3. apaga ships its own minimal protobuf3 decoder. No `libprotobuf` required.

The SafeTensors loader requires JSON parsing. apaga ships its own minimal JSON parser. No `libjson` or similar required.

### 16.4 Python bindings dependency

The Python bindings (`apaga.python`) require the Mojo-Python interop runtime. They are not part of `apaga` core and are compiled separately.

---

## 17. Performance Engineering Playbook

### 17.1 Profiling before optimizing

Every performance claim in apaga must be backed by a reproducible benchmark. The benchmarking framework (`apaga/bench/`) provides:

- Token throughput (tokens/second) for prefill and decode phases separately
- Memory bandwidth utilization
- Kernel-level breakdowns via hardware performance counters (perf events on Linux, Instruments on macOS)

**Prefill** (processing the prompt) is compute-bound: matrix × matrix.
**Decode** (generating one token at a time) is memory-bandwidth-bound: matrix × vector.

These two phases have different bottlenecks and different optimization strategies. Always measure them separately.

### 17.2 The memory bandwidth wall

In decode phase, the GPU (or CPU) is waiting for data from DRAM, not doing arithmetic. The roofline model tells us the theoretical maximum tokens/second based on:

```
max_tokens_per_sec = memory_bandwidth_GB_s / (model_size_GB / tokens_per_batch)
```

For a 7B Int4 model (3.5GB) on an M2 MacBook (100 GB/s memory bandwidth):
```
max_tokens_per_sec ≈ 100 / 3.5 ≈ 28.5 tok/s (single-token decode)
```

Every optimization in decode phase must be evaluated against this ceiling. Closing the gap to within 80% of the roofline is the performance target.

### 17.3 Tiling and cache utilization

The GEMM kernel is designed to keep data in L1/L2 cache by tiling. Tile sizes are chosen to satisfy:

```
A_tile + B_tile + C_tile ≤ L1D_cache_size * 0.75   (75% to avoid evictions)
```

This is computed at runtime from `CPUCaps.l1d_size_kb`.

### 17.4 Prefetch and pipelining

For sequential memory-bandwidth-bound ops (embedding lookup, activation functions), software prefetch intrinsics (`__builtin_prefetch` equivalent in Mojo) are inserted 2–3 cache lines ahead of the current position.

### 17.5 Batch size and throughput tradeoffs

| Batch size | Regime | Bottleneck | Strategy |
|---|---|---|---|
| 1 | Memory-bandwidth-bound | DRAM bandwidth | Minimize weight size (quantize) |
| 4–16 | Transitional | Mixed | GQA (reduce KV size) |
| 32+ | Compute-bound | Arithmetic units | Maximize FLOP utilization |

apaga's `SessionConfig.batch_size_hint` lets the user tell the runtime which regime to optimize for. The pass pipeline uses this to choose between dequant strategies and kernel implementations.

---

## 18. Testing & Validation Strategy

### 18.1 Numerical correctness

The hardest problem in an inference engine is silent numerical errors. A model may produce plausible-sounding text that is numerically slightly wrong. The test suite addresses this at every layer:

**Layer 2 (Tensor):** Property-based tests verify that reshape, broadcast, and slice produce correct results for random shapes.

**Layer 3 (Kernels):** Every kernel has a reference test comparing its output to a known-good PyTorch implementation, with tolerance based on dtype:

```
Float32 → atol=1e-5, rtol=1e-5
Float16 → atol=1e-3, rtol=1e-3
BFloat16 → atol=2e-3, rtol=2e-3
Int8 (dequant) → atol=0.02, rtol=0.02
Int4 (dequant) → atol=0.05, rtol=0.05
```

**Layer 5 (Passes):** Round-trip tests verify that a graph produces identical outputs before and after each pass.

**Layer 8 (Runtime):** End-to-end golden tests run a known model on a known prompt and compare token outputs exactly to a reference run. These use small models (e.g., GPT-2 124M) to keep CI fast.

### 18.2 Hardware-in-the-loop tests

CI runs on:
- GitHub Actions Linux x86_64 (CPU only)
- Self-hosted macOS ARM64 (CPU + Metal)
- Self-hosted Linux x86_64 with NVIDIA GPU (CPU + CUDA)

GPU tests are tagged and skipped if the hardware is unavailable.

### 18.3 Regression benchmarks

Every PR runs the benchmarking suite on Tier 1 platforms. A >5% regression in tokens/second on any benchmark blocks the PR. Improvements are celebrated in the PR description.

### 18.4 Fuzzing the loaders

Model loaders are the highest-risk code path for security and stability. All loaders are fuzz-tested with mutated input files (truncated, bit-flipped, oversized fields). No loader may panic or read out of bounds — it must return a `LoadError`.

---

## 19. Roadmap to Ollama-like System

apaga is designed as the inference engine component of a larger system. The following outlines how to build on top of apaga to create a self-contained local AI platform.

### 19.1 Component map

```
┌─────────────────────────────────────────────────────────┐
│                   USER-FACING LAYER                      │
│  CLI (apaga-cli)  ·  REST API Server  ·  Desktop App     │
├─────────────────────────────────────────────────────────┤
│                   MANAGEMENT LAYER                       │
│  ModelStore  ·  Converter  ·  Scheduler  ·  Sessions     │
├─────────────────────────────────────────────────────────┤
│                  ← apaga core library →                  │
└─────────────────────────────────────────────────────────┘
```

### 19.2 ModelStore

The ModelStore manages local model files (download, verify checksums, list, delete). It is a standalone module that depends on apaga only for format detection. Models are stored in `~/.apaga/models/` and can be pulled from a registry URL.

### 19.3 Converter

The Converter is a one-time tool that takes a GGUF, SafeTensors, or ONNX file, runs the full pass pipeline, and writes the result as an `.apg` file. The `.apg` file includes the pre-computed execution plan, so model loading in production is instant (no passes to run, no shape inference).

```
apaga convert llama3-8b.gguf --quant q4_k_m --output llama3-8b.apg
```

### 19.4 REST API Server

A thin HTTP server wrapping `apaga.Session`. Implements the OpenAI `/v1/chat/completions` and `/v1/completions` API so existing tools (Open WebUI, LangChain, etc.) work without changes.

```
apaga serve llama3-8b.apg --port 11434
```

### 19.5 Session scheduler

For the multi-user / multi-model case, a scheduler manages a pool of sessions, supports continuous batching (processing multiple requests simultaneously during decode), and handles context eviction when the KV cache is full.

---

## 20. Directory Structure

```
apaga/
├── ARCHITECTURE.md          ← This document
├── README.md
├── LICENSE
├── mojoproject.toml         ← Mojo package manifest
│
├── apaga/                   ← Core library source
│   ├── __init__.mojo        ← Public API re-exports
│   ├── hal/                 ← Layer 0
│   │   ├── cpu/
│   │   └── gpu/
│   ├── memory/              ← Layer 1
│   │   ├── arena.mojo
│   │   ├── pool.mojo
│   │   ├── weight_buffer.mojo
│   │   └── unified.mojo
│   ├── tensor/              ← Layer 2
│   │   ├── dtype.mojo
│   │   ├── shape.mojo
│   │   ├── view.mojo
│   │   └── tensor.mojo
│   ├── kernels/             ← Layer 3
│   │   ├── gemm/
│   │   ├── attention/
│   │   ├── norm/
│   │   ├── activation/
│   │   ├── embedding/
│   │   ├── conv/
│   │   ├── reduce/
│   │   └── elementwise/
│   ├── dispatch/            ← Layer 4
│   │   ├── registry.mojo
│   │   ├── key.mojo
│   │   └── selector.mojo
│   ├── graph/               ← Layer 5
│   │   ├── node.mojo
│   │   ├── edge.mojo
│   │   ├── graph.mojo
│   │   ├── passes/
│   │   │   ├── shape_inference.mojo
│   │   │   ├── constant_folding.mojo
│   │   │   ├── fusion.mojo
│   │   │   ├── memory_planning.mojo
│   │   │   └── execution_order.mojo
│   │   └── format/
│   │       └── apg.mojo
│   ├── quant/               ← Layer 6
│   │   ├── params.mojo
│   │   ├── rtn.mojo
│   │   ├── gptq.mojo
│   │   ├── awq.mojo
│   │   └── gguf_compat.mojo
│   ├── loaders/             ← Layer 7
│   │   ├── gguf/
│   │   │   ├── loader.mojo
│   │   │   ├── parser.mojo
│   │   │   └── architectures/
│   │   ├── onnx/
│   │   │   ├── loader.mojo
│   │   │   └── proto3_mini.mojo
│   │   ├── safetensors/
│   │   └── apg/
│   ├── runtime/             ← Layer 8
│   │   ├── plan.mojo
│   │   ├── kv_cache.mojo
│   │   ├── session.mojo
│   │   └── sampler.mojo
│   ├── api.mojo             ← Layer 9: Public API
│   └── python/              ← Python bindings
│       └── bindings.mojo
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── golden/
│   └── fuzz/
│
├── bench/
│   ├── gemm_bench.mojo
│   ├── attention_bench.mojo
│   └── e2e_bench.mojo
│
├── tools/
│   ├── converter/           ← GGUF/ONNX → APG converter
│   └── inspector/           ← APG file inspector (like gguf-dump)
│
└── examples/
    ├── llm_chat.mojo
    ├── image_classify.mojo
    └── embed_text.mojo
```

---

## 21. Glossary

| Term | Definition |
|---|---|
| **APG** | apaga's native binary model format |
| **Arena** | Bump-pointer allocator; freed all at once |
| **AWQ** | Activation-Aware Weight Quantization; a post-training quantization method |
| **Backend** | A hardware-specific implementation registered with the HAL |
| **DType** | Data type of tensor elements (Float32, Int4, BFloat16, etc.) |
| **DispatchedOp** | A kernel + its resolved inputs/outputs; the unit of execution |
| **ExecutionPlan** | The flat, ordered, ready-to-execute form of an OpGraph |
| **Flash Attention** | Memory-efficient tiled attention algorithm avoiding O(S²) materialization |
| **GQA** | Grouped Query Attention; reduces KV cache size by sharing heads |
| **GGUF** | GGML Unified Format; llama.cpp's binary model format |
| **HAL** | Hardware Abstraction Layer; apaga's Layer 0 |
| **KV Cache** | Stores past key/value projections to avoid recomputation in autoregressive decoding |
| **Micro-kernel** | The innermost register-tiled loop in a GEMM implementation |
| **mmap** | Memory-mapped file I/O; allows files to be accessed as memory without copying |
| **OpGraph** | Directed acyclic graph of operators; apaga's model IR |
| **Pass** | A graph transformation function (shape inference, fusion, etc.) |
| **Prefill** | Processing the input prompt tokens (compute-bound) |
| **Decode** | Generating one output token at a time (memory-bandwidth-bound) |
| **RoPE** | Rotary Position Embedding; the dominant position encoding in modern LLMs |
| **RTN** | Round-to-Nearest; simplest quantization scheme |
| **Roofline model** | Theoretical performance ceiling based on compute or memory bandwidth limit |
| **SafeTensors** | HuggingFace's safe, Pickle-free model serialization format |
| **SIMD** | Single Instruction, Multiple Data; vectorized CPU arithmetic |
| **TensorView** | Non-owning fat-pointer view of tensor data |
| **Unified Memory** | Memory accessible from both CPU and GPU without explicit copy |
| **W4A16** | 4-bit weights, 16-bit activations; most common LLM quantization regime |
| **W8A8** | 8-bit weights and activations; used for highest-throughput integer inference |

---

*apaga architecture document · revision 0.1.0*
*All design decisions in this document are provisional until the corresponding layer passes its test suite.*
