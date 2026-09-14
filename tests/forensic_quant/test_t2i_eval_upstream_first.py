import sys
import types

fake_torch = types.ModuleType("torch")
fake_torch_utils = types.ModuleType("torch.utils")
fake_torch_utils_data = types.ModuleType("torch.utils.data")
fake_torch_utils_data.DataLoader = object
fake_torch.utils = fake_torch_utils
fake_torch_utils.data = fake_torch_utils_data
sys.modules.setdefault("torch", fake_torch)
sys.modules.setdefault("torch.utils", fake_torch_utils)
sys.modules.setdefault("torch.utils.data", fake_torch_utils_data)
try:
    import PIL  # noqa: F401
    import PIL.Image  # noqa: F401
except ModuleNotFoundError:
    sys.modules.setdefault("PIL", types.SimpleNamespace(Image=object))
    sys.modules.setdefault("PIL.Image", types.SimpleNamespace(Image=object))

from src.forensic_quant.config import T2IConfig
from src.forensic_quant.t2i_eval import _load_pipeline, _make_decode_fn


class FakeLatents:
    def __init__(self):
        self.unsqueezed_dim = None

    def unsqueeze(self, dim):
        self.unsqueezed_dim = dim
        return ("unsqueezed", dim, self)


class DecodeOnly:
    def __init__(self):
        self.seen = None

    def decode(self, latents):
        self.seen = latents
        return latents


def test_make_decode_fn_uses_upstream_unsqueeze_convention_for_fp_decoder():
    latents = FakeLatents()
    decoder = DecodeOnly()
    decode = _make_decode_fn(decoder, config=None, quantized=False)

    result = decode(latents)

    assert result == ("unsqueezed", 0, latents)
    assert decoder.seen is latents
    assert latents.unsqueezed_dim == 0


def test_load_pipeline_does_not_force_torch_dtype(monkeypatch):
    calls = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls["model"] = model
            calls["kwargs"] = kwargs
            return cls()

        def to(self, device):
            calls["device"] = device
            return self

        def set_progress_bar_config(self, **kwargs):
            calls["progress"] = kwargs

    fake_diffusers = types.SimpleNamespace(StableDiffusionPipeline=FakePipeline)
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    monkeypatch.setenv("HF_TOKEN", "token-123")

    device = types.SimpleNamespace(type="cuda")
    config = types.SimpleNamespace(t2i=T2IConfig(diffusers_model="mirror/model"))
    pipe = _load_pipeline(config, device)

    assert isinstance(pipe, FakePipeline)
    assert calls["model"] == "mirror/model"
    assert calls["kwargs"] == {"token": "token-123", "local_files_only": False}
    assert calls["device"] is device

