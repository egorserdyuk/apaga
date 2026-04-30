from ..cpu.caps import get_global_cpu_caps

alias SIMDWidth = Int

alias SIMD_WIDTH_SCALAR = 1
alias SIMD_WIDTH_128 = 16
alias SIMD_WIDTH_256 = 32
alias SIMD_WIDTH_512 = 64

fn simd_width_for_dtype(dtype: Int) -> Int:
    caps = get_global_cpu_caps()
    let elem_size = _dtype_to_bytes(dtype)

    if caps.is_x86():
        if caps.has_avx512f:
            return 512 // elem_size
        elif caps.has_avx2:
            return 256 // elem_size
    elif caps.is_arm():
        if caps.has_sve:
            return 128 // elem_size
        return 128 // elem_size

    return 128 // elem_size

fn _dtype_to_bytes(dtype: Int) -> Int:
    if dtype == 0:
        return 4
    elif dtype == 1:
        return 2
    elif dtype == 2:
        return 2
    elif dtype == 3:
        return 1
    elif dtype == 4:
        return 1
    return 4