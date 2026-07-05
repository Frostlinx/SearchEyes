from searcheyes.vlm_agent import (
    resolve_attn_implementation_choice,
    resolve_quantization_choice,
)


class _CudaProps:
    def __init__(self, total_memory_gb: float):
        self.total_memory = int(total_memory_gb * (1024**3))


class _Cuda:
    def __init__(self, total_memory_gb: float):
        self._props = _CudaProps(total_memory_gb)

    def get_device_properties(self, _: int):
        return self._props


class _MpsBackend:
    @staticmethod
    def is_available() -> bool:
        return False


class _Backends:
    mps = _MpsBackend()


class _TorchStub:
    def __init__(self, total_memory_gb: float):
        self.cuda = _Cuda(total_memory_gb)
        self.backends = _Backends()


def test_quantization_auto_prefers_4bit_for_8gb_cuda():
    torch_stub = _TorchStub(total_memory_gb=8.0)
    assert resolve_quantization_choice(torch_stub, "auto", "cuda") == "4bit"


def test_quantization_auto_disables_low_bit_on_cpu():
    torch_stub = _TorchStub(total_memory_gb=8.0)
    assert resolve_quantization_choice(torch_stub, "auto", "cpu") == "none"


def test_attn_auto_uses_sdpa_on_cuda():
    assert resolve_attn_implementation_choice("auto", "cuda") == "sdpa"


def test_attn_auto_uses_eager_on_cpu():
    assert resolve_attn_implementation_choice("auto", "cpu") == "eager"
