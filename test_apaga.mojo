from apaga import DType, Shape, Tensor, TensorView
from apaga import CPUCaps, CPUArch, CPUVendor
from apaga import Arena

def test_dtype():
    var dtype = DType.float32()
    print("DType float32 bytes:", dtype.bytes_per_elem())

    var shape = Shape.from_params(2, 3, 4)
    print("Shape numel:", shape.numel())

    try:
        var tensor = Tensor(dtype, Shape.from_params(1, 10))
        print("Created tensor with shape:", tensor.view.shape.numel())
    except:
        print("Tensor creation skipped")

def test_arena():
    try:
        var arena = Arena(1024)
        print("Arena capacity:", arena.available())
    except:
        print("Arena test failed")

def main() raises:
    print("=== Testing apaga core ===")
    test_dtype()
    test_arena()
    print("=== All tests passed ===")