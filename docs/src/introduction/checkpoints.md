# Checkpoints and workflows

A prepared model can be saved as a checkpoint and reloaded later, with its compression state intact. For example, you can use a checkpoint to:

- Keep the results of expensive compression steps, such as k-means clustering, calibration, or training, and continue from them later instead of redoing them.
- Prepare and calibrate or train a model on one machine, and finalize and export it on another.
- Finalize the same compressed model for more than one backend, for example to evaluate it with the [Torch backend](torch_backend.md) and then export it to [Core AI](integration_coreai.md).

This page covers how to checkpoint and reload a prepared model, and shows workflows that use checkpoints.

## Checkpointing and reloading prepared models

Every compressor's `prepare()` accepts a `state_dict` argument. Save the prepared model's `state_dict` (after calibration or training, if any), and later reload it into a fresh copy of the original model with the same compressor and config:

```python
import copy

import torch

from coreai_opt.quantization import Quantizer, QuantizerConfig

# INT8 weights and INT8 activations.
config = QuantizerConfig()

# 1. Prepare: insert fake quantization into a copy of the original model.
quantizer = Quantizer(copy.deepcopy(model), config)
prepared_model = quantizer.prepare(example_inputs)

# 2. Calibrate: observe activation ranges on representative data.
with quantizer.calibration_mode():
    for batch in calibration_dataloader:
        prepared_model(batch)

# 3. Save a checkpoint: the prepared model's state dict holds the compression
#    state, such as the calibrated quantization parameters.
torch.save(prepared_model.state_dict(), "prepared.pt")

# 4. Later, possibly on another machine: load the checkpoint.
state_dict = torch.load("prepared.pt")

# 5. Reload: prepare a fresh copy of the original model with the same config,
#    restoring the saved state instead of calibrating again.
quantizer = Quantizer(copy.deepcopy(model), config)
prepared_model = quantizer.prepare(example_inputs, state_dict=state_dict)
```

Reloading restores the saved compression state, such as palettization centroids, calibrated quantization parameters, and pruning masks, instead of redoing heavy computation such as k-means clustering. The reloaded model is in the same state as after a regular `prepare()` call, ready to evaluate, calibrate, train, or finalize for any backend.

When reloading, use the same config and model code as the run that produced the checkpoint. In graph mode, also use the same example inputs, dynamic shapes, and torch version. A mismatch raises an error rather than loading the wrong state.

## Common workflows

The workflows below show how to checkpoint and reload a prepared model in common scenarios.

### Evaluate with the Torch backend, then export to Core AI

Compress once, evaluate the compressed model quickly in PyTorch, then export the same compressed model to Core AI, without compressing it again:

```python
import copy

import torch

import coreai_opt as opt
from coreai_opt.palettization import KMeansPalettizer, KMeansPalettizerConfig

config = KMeansPalettizerConfig.presets.w4()

# 1. Compress once and save a checkpoint before finalizing.
palettizer = KMeansPalettizer(copy.deepcopy(model), config)
prepared_model = palettizer.prepare(example_inputs)
torch.save(prepared_model.state_dict(), "prepared.pt")

# 2. Evaluate with the Torch backend.
eval_model = palettizer.finalize(backend=opt.ExportBackend.Torch)
evaluate(eval_model.eval())

# 3. Reload the checkpoint, without re-running k-means, and finalize for Core AI.
state_dict = torch.load("prepared.pt")
palettizer = KMeansPalettizer(copy.deepcopy(model), config)
prepared_model = palettizer.prepare(example_inputs, state_dict=state_dict)
coreai_model = palettizer.finalize(backend=opt.ExportBackend.CoreAI)
```

Convert `coreai_model` as described in [Integration with Core AI](integration_coreai.md). The same flow works for every compressor; for `Quantizer`, calibrate or train before saving the checkpoint.

### Resume pruning with fine-tuning

[Pruning with fine-tuning](../pruning/overview.md) ramps sparsity up over many training steps with a sparsity schedule. Save a checkpoint during training, and resume later from where it stopped: the schedule continues from the saved step, so the resumed run continues exactly like an uninterrupted one.

```python
import copy

import torch

from coreai_opt.pruning import (
    MagnitudePruner,
    MagnitudePrunerConfig,
    ModuleMagnitudePrunerConfig,
    PruningSpec,
)
from coreai_opt.pruning.config import PolynomialDecaySchedule

num_epochs = 10

# Ramp sparsity up to 70% over num_epochs, one pruner.step() per epoch.
config = MagnitudePrunerConfig(
    global_config=ModuleMagnitudePrunerConfig(
        op_state_spec={"weight": PruningSpec(target_sparsity=0.7)},
        sparsity_schedule=PolynomialDecaySchedule(
            begin_step=0, total_iters=num_epochs, power=3.0
        ),
    ),
)

# 1. First session: prepare, then fine-tune for a few epochs while ramping up sparsity.
pruner = MagnitudePruner(copy.deepcopy(model), config)
prepared_model = pruner.prepare(example_inputs)
optimizer = torch.optim.SGD(prepared_model.parameters(), lr=1e-3)
for epoch in range(4):
    train_one_epoch(prepared_model, train_dataloader, optimizer)
    pruner.step()

# 2. Save a checkpoint mid-schedule: the model's state dict holds the pruning masks
#    and the schedule's step count. The optimizer state is saved alongside it.
torch.save(
    {"model": prepared_model.state_dict(), "optimizer": optimizer.state_dict()},
    "pruning.pt",
)

# 3. Later session: reload the model and the optimizer.
checkpoint = torch.load("pruning.pt")
pruner = MagnitudePruner(copy.deepcopy(model), config)
prepared_model = pruner.prepare(example_inputs, state_dict=checkpoint["model"])
optimizer = torch.optim.SGD(prepared_model.parameters(), lr=1e-3)
optimizer.load_state_dict(checkpoint["optimizer"])

# 4. Continue fine-tuning: the schedule resumes from the saved step.
for epoch in range(4, num_epochs):
    train_one_epoch(prepared_model, train_dataloader, optimizer)
    pruner.step()
```

### Checkpoint and reload a palettized and activation-quantized (P4A8) model

[Joint compression](../utils/joint_compression.md) applies two compressors in sequence: palettize the weights, finalize the palettizer, then quantize and calibrate the activations. Save a checkpoint after each stage, then rebuild the compressed model by reloading the stages in the same order, without re-running k-means or calibration:

```python
import copy

import torch

import coreai_opt as opt
from coreai_opt.palettization import KMeansPalettizer
from coreai_opt.quantization import Quantizer

# palett_config and quant_config as in Joint Compression.

# First run: compress, saving a checkpoint after each stage.
palettizer = KMeansPalettizer(copy.deepcopy(model), palett_config)
palettized = palettizer.prepare(example_inputs)
torch.save(palettized.state_dict(), "palettized.pt")  # before finalize(), which modifies the model
palettized_model = palettizer.finalize(backend=opt.ExportBackend.CoreAI)

quantizer = Quantizer(palettized_model, quant_config)
prepared_model = quantizer.prepare(example_inputs)
with quantizer.calibration_mode():
    for batch in calibration_dataloader:
        prepared_model(batch)
torch.save(prepared_model.state_dict(), "quantized.pt")

# Later: rebuild the P4A8 model from the checkpoints.
palett_state_dict = torch.load("palettized.pt")
palettizer = KMeansPalettizer(copy.deepcopy(model), palett_config)
palettizer.prepare(example_inputs, state_dict=palett_state_dict)
palettized_model = palettizer.finalize(backend=opt.ExportBackend.CoreAI)

quant_state_dict = torch.load("quantized.pt")
quantizer = Quantizer(palettized_model, quant_config)
prepared_model = quantizer.prepare(example_inputs, state_dict=quant_state_dict)

coreai_model = quantizer.finalize(backend=opt.ExportBackend.CoreAI)
```

Two checkpoints are needed because the quantizer's checkpoint is taken on the palettizer's finalized model: reloading it requires rebuilding that model first, by preparing and finalizing the palettizer. Without its own checkpoint, `palettizer.prepare()` would re-run k-means on the original weights, only for the result to be overwritten by the palettized weights stored in the quantizer's checkpoint. For the same reason, finalize the palettizer for the same backend in both runs. The reloaded `prepared_model` is calibrated and ready to finalize for Core AI.
