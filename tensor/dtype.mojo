comptime DTYPE_FLOAT32: Int = 0
comptime DTYPE_FLOAT16: Int = 1
comptime DTYPE_BFLOAT16: Int = 2
comptime DTYPE_FLOAT8E4M3: Int = 3
comptime DTYPE_FLOAT8E5M2: Int = 4
comptime DTYPE_INT32: Int = 5
comptime DTYPE_INT16: Int = 6
comptime DTYPE_INT8: Int = 7
comptime DTYPE_INT4: Int = 8
comptime DTYPE_INT2: Int = 9
comptime DTYPE_UINT8: Int = 10
comptime DTYPE_UINT4: Int = 11
comptime DTYPE_BOOL: Int = 12

struct DType(Copyable):
    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    @staticmethod
    fn float32() -> DType:
        return DType(DTYPE_FLOAT32)

    @staticmethod
    fn float16() -> DType:
        return DType(DTYPE_FLOAT16)

    @staticmethod
    fn bf16() -> DType:
        return DType(DTYPE_BFLOAT16)

    @staticmethod
    fn int32() -> DType:
        return DType(DTYPE_INT32)

    @staticmethod
    fn int16() -> DType:
        return DType(DTYPE_INT16)

    @staticmethod
    fn int8() -> DType:
        return DType(DTYPE_INT8)

    @staticmethod
    fn int4() -> DType:
        return DType(DTYPE_INT4)

    @staticmethod
    fn uint8() -> DType:
        return DType(DTYPE_UINT8)

    @staticmethod
    fn uint4() -> DType:
        return DType(DTYPE_UINT4)

    @staticmethod
    fn bool() -> DType:
        return DType(DTYPE_BOOL)

    fn bytes_per_elem(ref self) -> Int:
        if self.value == DTYPE_FLOAT32:
            return 4
        elif self.value == DTYPE_FLOAT16 or self.value == DTYPE_BFLOAT16:
            return 2
        elif self.value == DTYPE_FLOAT8E4M3 or self.value == DTYPE_FLOAT8E5M2:
            return 1
        elif self.value == DTYPE_INT32:
            return 4
        elif self.value == DTYPE_INT16:
            return 2
        elif self.value == DTYPE_INT8 or self.value == DTYPE_UINT8:
            return 1
        elif self.value == DTYPE_INT4 or self.value == DTYPE_UINT4:
            return 1
        elif self.value == DTYPE_BOOL:
            return 1
        elif self.value == DTYPE_INT2:
            return 1
        return 4

    fn is_float(ref self) -> Bool:
        return self.value >= DTYPE_FLOAT32 and self.value <= DTYPE_FLOAT8E5M2

    fn is_int(ref self) -> Bool:
        return self.value >= DTYPE_INT32 and self.value <= DTYPE_UINT4

    fn is_quantized(ref self) -> Bool:
        return self.value == DTYPE_INT4 or self.value == DTYPE_UINT4 or self.value == DTYPE_INT2

    fn __eq__(ref self, other: DType) -> Bool:
        return self.value == other.value

    fn __ne__(ref self, other: DType) -> Bool:
        return self.value != other.value

fn promote_dtype(a: DType, b: DType) -> DType:
    if a == b:
        return a
    if a == DType.float32() or b == DType.float32():
        return DType.float32()
    if a == DType.float16() or b == DType.float16():
        return DType.float32()
    if a == DType.bf16() or b == DType.bf16():
        return DType.float32()
    if a == DType.int32() or b == DType.int32():
        return DType.int32()
    if a == DType.int8() or b == DType.int8():
        return DType.int32()
    if a == DType.int4() or b == DType.int4():
        return DType.int8()
    return a