from enum import Enum, EnumElement
from tensor import DType, TensorView

alias QUANT_SCHEME_RTN = 0
alias QUANT_SCHEME_GPTQ = 1
alias QUANT_SCHEME_AWQ = 2
alias QUANT_SCHEME_SMOOTHQUANT = 3
alias QUANT_SCHEME_GGUF_Q4_K_M = 4
alias QUANT_SCHEME_NF4 = 5

struct QuantScheme(EnumElement):
    var value: Int

    @staticmethod
    fn rtn() -> QuantScheme:
        return QuantScheme(QUANT_SCHEME_RTN)

    @staticmethod
    fn gptq() -> QuantScheme:
        return QuantScheme(QUANT_SCHEME_GPTQ)

    @staticmethod
    fn awq() -> QuantScheme:
        return QuantScheme(QUANT_SCHEME_AWQ)

    @staticmethod
    fn smoothquant() -> QuantScheme:
        return QuantScheme(QUANT_SCHEME_SMOOTHQUANT)

    @staticmethod
    fn gguf_q4_k_m() -> QuantScheme:
        return QuantScheme(QUANT_SCHEME_GGUF_Q4_K_M)

    @staticmethod
    fn nf4() -> QuantScheme:
        return QuantScheme(QUANT_SCHEME_NF4)

struct QuantParams:
    var scheme: QuantScheme
    var group_size: Int
    var scales: TensorView[DType.float16]
    var zeros: TensorView[DType.float16]
    var bits: Int

    def __init__(out self, scheme: QuantScheme, group_size: Int, bits: Int):
        self.scheme = scheme
        self.group_size = group_size
        self.bits = bits
        self.scales = TensorView[DType.float16]()
        self.zeros = TensorView[DType.float16]()

    fn is_per_channel(ref self) -> Bool:
        return self.group_size == -1

    fn num_groups(ref self, total_elements: Int) -> Int:
        if self.is_per_channel():
            return total_elements
        return (total_elements + self.group_size - 1) // self.group_size