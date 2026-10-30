# Fast inference in PyTorch

A prepared model applies compression on the fly: on every forward pass, each compressed weight is reconstructed from its compression state (for example, the palettization LUT and indices, the quantization scales, or the pruning mask). That keeps the model ready for calibration and training, but it makes inference slow, especially for large models.

`finalize(backend=opt.ExportBackend.Torch)` computes each compressed weight once and folds it into a dense weight. Activation fake quantization is kept, so the finalized model produces the same outputs as the prepared model, while running much faster.

```python
import coreai_opt as opt

prepared_model = quantizer.prepare(example_inputs)
# ... calibrate or train if needed ...
eval_model = quantizer.finalize(backend=opt.ExportBackend.Torch)
eval_model.eval()
```

Use the Torch backend to:

- Evaluate the accuracy of a compressed model, or benchmark it, in PyTorch.
- Use the compressed weights with other PyTorch tooling. With weight-only compression (palettization, pruning, or weight-only quantization in eager mode), finalizing removes the compression parametrizations from the weights and writes the compressed values into them. The finalized model has the original model structure, with compressed weight values, so it works with any tooling that expects the original model structure.

Don't use it to:

- **Export a compressed model.** The finalized weights are dense and carry no compression information, so they can't be converted to a compressed Core AI or Core ML model. Keep a checkpoint of the prepared model and finalize it for `ExportBackend.CoreAI` or `ExportBackend.CoreML` instead. See [Evaluate with the Torch backend, then export to Core AI](checkpoints.md#evaluate-with-the-torch-backend-then-export-to-core-ai).
- **Fine-tune the compressed model.** The folded weights are plain parameters, so training moves them away from their compressed values. Fine-tune the prepared model instead, for example with `quantizer.training_mode()` (see [How to use coreai-opt](how_to_use_coreaiopt.md)), and finalize afterwards.
