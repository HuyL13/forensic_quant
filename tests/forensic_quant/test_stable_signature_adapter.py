import types

from src.forensic_quant.stable_signature_adapter import move_module_to_device, set_decoder_mode


class WeirdWrapper:
    def __init__(self):
        self.to_device = None
        self.eval_called = False
        self.post_quant_conv = Child()
        self.decoder = Child()

    def eval(self):
        self.eval_called = True
        return False

    def to(self, device):
        self.to_device = device
        return self


class Child:
    def __init__(self):
        self.mode = None

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"


def test_move_module_to_device_does_not_chain_eval_return_value():
    module = WeirdWrapper()

    result = move_module_to_device(module, "cuda")

    assert result is module
    assert module.eval_called is True
    assert module.to_device == "cuda"


def test_set_decoder_mode_only_touches_decoder_children():
    module = WeirdWrapper()
    module.train = lambda: (_ for _ in ()).throw(TypeError("disabled_train missing self"))

    set_decoder_mode(module, training=True)
    assert module.post_quant_conv.mode == "train"
    assert module.decoder.mode == "train"

    set_decoder_mode(module, training=False)
    assert module.post_quant_conv.mode == "eval"
    assert module.decoder.mode == "eval"
