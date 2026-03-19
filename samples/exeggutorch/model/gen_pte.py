import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.export import export, ExportedProgram
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights
from torchvision.models.squeezenet import SqueezeNet1_0_Weights

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower, to_edge, EdgeCompileConfig
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir.backend.utils import format_delegated_graph
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Precision helpers
# ---------------------------------------------------------------------------

DTYPE_MAP = {
    "fp32":       torch.float32,
    "fp16":       torch.float16,
    "bf16":       torch.bfloat16,
    "fp8_e4m3fn": torch.float8_e4m3fn,  # preferred for weights/activations
    "fp8_e5m2":   torch.float8_e5m2,    # wider dynamic range
}


def apply_precision(model, sample_inputs, precision):
    dtype = DTYPE_MAP[precision]
    if dtype != torch.float32:
        model = model.to(dtype)
        sample_inputs = tuple(
            t.to(dtype) if t.is_floating_point() else t for t in sample_inputs
        )
    return model, sample_inputs


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

_S = 28  # MNIST image side length


class TinyMLPMNIST(nn.Module):
    """Small MLP for MNIST digit classification (784 → 32 → 16 → 10)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(_S * _S, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def create_balanced_model():
    """Hand-crafted weights that recognise MNIST digits 0, 1, 4, and 7."""
    model = TinyMLPMNIST()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        w1 = model.fc1.weight

        # Feature 0: vertical line at column 14  (helps detect 1, 4, 7)
        w1[0, 14::_S] = 2.0

        # Feature 1: top horizontal band (rows 0-2)  → 7, 4
        w1[1, : 3 * _S] = 2.0

        # Feature 2: bottom horizontal band (rows 25-27)  → 1, 4
        w1[2, 25 * _S :] = 2.0

        # Feature 3: oval detector for 0
        w1[3, _S + 8 : _S + 20] = 2.0                      # top edge
        w1[3, 26 * _S + 8 : 26 * _S + 20] = 2.0            # bottom edge
        rows = torch.arange(4, 24)
        w1[3, rows * _S + 7] = 2.0                          # left edge
        w1[3, rows * _S + 20] = 2.0                         # right edge
        rows = torch.arange(10, 18)
        w1[3, rows * _S + 14] = -1.5                        # hollow centre

        # Feature 4: middle horizontal band (rows 13-14)  → 4
        w1[4, 13 * _S : 15 * _S] = 3.0

        # Second layer: combine features into 4 hidden units
        w2 = model.fc2.weight
        w2[0, [3, 0, 4]] = torch.tensor([ 5.0, -2.0, -3.0])
        w2[1, [0, 2, 1, 3]] = torch.tensor([ 3.0,  2.0, -1.0, -2.0])
        w2[2, [0, 1, 4, 3]] = torch.tensor([ 2.0,  1.0,  4.0, -2.0])
        w2[3, [1, 0, 2]] = torch.tensor([ 3.0,  1.0, -2.0])

        # Output layer: route hidden units to digits 0, 1, 4, 7
        w3 = model.fc3.weight
        w3[0, 0] = 5.0
        w3[1, 1] = 5.0
        w3[4, 2] = 5.0
        w3[7, 3] = 5.0

        # Suppress unrecognised digits
        model.fc3.bias[[2, 3, 5, 6, 8, 9]] = -3.0

    return model


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=8, dim_feedforward=128, seq_len=16):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        x = x + self.pos_embedding
        return self.encoder(x)


# ---------------------------------------------------------------------------
# Model registry  –  each entry: () -> (nn.Module, sample_inputs)
# ---------------------------------------------------------------------------

_224x224 = (torch.randn(1, 3, 224, 224),)

MODEL_REGISTRY = {
    "mobilenet":      lambda: (models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).eval(),         _224x224),
    "squeezenet":     lambda: (models.squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT).eval(),       _224x224),
    "alexnet":        lambda: (models.alexnet(weights=models.AlexNet_Weights.DEFAULT).eval(),            _224x224),
    "mobilenetv3small": lambda: (models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).eval(), _224x224),
    "lenet":          lambda: (LeNet().eval(),                  (torch.randn(1, 1, 28, 28),)),
    "transformer":    lambda: (SimpleTransformer().eval(),      (torch.randn(1, 16, 64),)),
    "mnist_mlp":      lambda: (create_balanced_model().eval(),  (torch.randn(1, 28, 28),)),
}


def build_model(name):
    return MODEL_REGISTRY[name]()


# ---------------------------------------------------------------------------
# Inference test
# ---------------------------------------------------------------------------

def _print_output(output):
    print(f"  output shape : {tuple(output.shape)}  dtype: {output.dtype}")
    torch.set_printoptions(sci_mode=False, precision=6)
    print(f"  output : {output}")
    torch.set_printoptions(sci_mode=None, precision=4)


def run_test(model, sample_inputs):
    shape_str = ", ".join(str(tuple(t.shape)) for t in sample_inputs)
    print(f"  input shape(s) : {shape_str}  dtype: {sample_inputs[0].dtype}")

    with torch.no_grad():
        print("  [random input]")
        _print_output(model(*sample_inputs))

        ones_inputs = tuple(torch.ones_like(t) for t in sample_inputs)
        print("  [all-ones input]")
        _print_output(model(*ones_inputs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pte", type=str, default="model.pte",
        help="Path to output the PTE file.",
    )
    parser.add_argument(
        "--model", type=str, default="mobilenet",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to export.",
    )
    parser.add_argument(
        "--precision", type=str, default="fp32",
        choices=list(DTYPE_MAP.keys()),
        help="Model weight/activation dtype (fp32, fp16, bf16, fp8_e4m3fn, fp8_e5m2).",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run a forward pass with random inputs and print results; skip export.",
    )
    parser.add_argument(
        "--lstats", action="store_true",
        help="Print XNNPack delegation summary and delegated graph after lowering.",
    )
    parser.add_argument(
        "--deprecated-lower", action="store_true",
        help="[TEMP] Use deprecated to_edge() + to_backend() instead of to_edge_transform_and_lower().",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible test inputs.",
    )
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"Model: {args.model}  Precision: {args.precision}")

    model, sample_inputs = build_model(args.model)
    model, sample_inputs = apply_precision(model, sample_inputs, args.precision)

    if args.test:
        run_test(model, sample_inputs)
        return

    print(f"Output: {args.pte}")
    exported_program: ExportedProgram = export(model, sample_inputs)

    if args.deprecated_lower:
        # TEMP: deprecated two-step flow for comparison/debugging
        edge = to_edge(exported_program, compile_config=EdgeCompileConfig(_check_ir_validity=False))
        edge = edge.to_backend(XnnpackPartitioner())
    else:
        edge = to_edge_transform_and_lower(
            exported_program,
            partitioner=[XnnpackPartitioner()],
        )

    if args.lstats:
        graph_module = edge.exported_program().graph_module
        delegation_info = get_delegation_info(graph_module)
        print("\n── Delegation Summary ──")
        print(delegation_info.get_summary())
        df = delegation_info.get_operator_delegation_dataframe()
        print(tabulate(df, headers="keys", tablefmt="fancy_grid"))
        print("\n── Delegated Graph ──")
        print(format_delegated_graph(graph_module))

    exec_prog = edge.to_executorch()
    with open(args.pte, "wb") as f:
        exec_prog.write_to_file(f)

    print(f"Wrote {args.pte}")


if __name__ == "__main__":
    main()
