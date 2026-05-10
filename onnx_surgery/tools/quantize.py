"""ONNX model quantization — INT8 (static/dynamic) and FP16 half-precision conversion.

Supports three quantization strategies:

1. **FP16 conversion** — Cast all float32 tensors to float16 (half-precision).
   Reduces model size by ~50% and can speed up inference on GPU hardware.

2. **Dynamic INT8 quantization** — Quantize weights to INT8 with
   dynamically-computed activation ranges. No calibration data required.

3. **Static INT8 quantization** — Quantize both weights and activations to INT8
   using representative calibration data. Highest compression and throughput.

Requires ``onnxruntime`` (optional dependency) for INT8 quantization backends.
FP16 conversion is pure-ONNX and works without onnxruntime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from onnx import TensorProto, ModelProto, numpy_helper, save as onnx_save

from ..core.model_loader import load_model as _load_model

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FP16 conversion (pure ONNX, no external deps)
# ---------------------------------------------------------------------------


def convert_to_fp16(model: ModelProto, min_positive: float = 1e-7) -> ModelProto:
    """Convert all float32 tensors in the model to float16 (half precision).

    Args:
        model: Input ONNX model (modified in-place, returned as new copy).
        min_positive: Minimum positive value to avoid underflow (default 1e-7).

    Returns:
        New ModelProto with float32 tensors cast to float16.

    Note:
        This does **not** set the overall model IR type to FP16 — individual
        tensor data types are converted.  Use :func:`convert_model_precision`
        for a higher-level wrapper that also updates value_info.
    """
    converted = model  # work in-place on the passed model

    # --- Convert initializers (weights / constants) ---
    for init in converted.graph.initializer:
        if init.data_type in (TensorProto.FLOAT,):
            _cast_initializer(init, TensorProto.FLOAT16, min_positive)

    # --- Convert node attributes that contain tensors ---
    for node in converted.graph.node:
        for attr in node.attribute:
            if attr.type == 7 and attr.t:  # TENSOR
                _cast_tensor_proto(attr.t, TensorProto.FLOAT16, min_positive)
            elif attr.type == 8:  # GRAPH
                _recurse_subgraph(attr.g, TensorProto.FLOAT16, min_positive)
            elif attr.type == 6:  # TENSORS (repeated)
                for t in attr.tensors:
                    _cast_tensor_proto(t, TensorProto.FLOAT16, min_positive)

    # --- Convert input / output value info (type hints) ---
    for v in list(converted.graph.input) + list(converted.graph.output):
        if (
            v.type.HasField("tensor_type")
            and v.type.tensor_type.elem_type == TensorProto.FLOAT
        ):
            v.type.tensor_type.elem_type = TensorProto.FLOAT16

    return converted


def convert_model_precision(
    model: ModelProto,
    target_dtype: int = TensorProto.FLOAT16,
    min_positive: float = 1e-7,
) -> ModelProto:
    """High-level wrapper: convert model precision (FP16 or FP32).

    Args:
        model: ONNX model to convert.
        target_dtype: Target data type (TensorProto.FLOAT16 or TensorProto.FLOAT).
                      Default FLOAT16.
        min_positive: Minimum positive value to avoid underflow (FP16 only).

    Returns:
        Converted ModelProto.
    """
    if target_dtype == TensorProto.FLOAT16:
        return convert_to_fp16(model, min_positive)
    elif target_dtype == TensorProto.FLOAT:
        # FP32 conversion (cast FP16 back) — reverse of above
        return _convert_from_fp16(model)
    else:
        raise ValueError(f"Unsupported target dtype: {target_dtype}")


def _convert_from_fp16(model: ModelProto, min_positive: float = 1e-7) -> ModelProto:
    """Convert float16 tensors back to float32."""
    for init in model.graph.initializer:
        if init.data_type in (TensorProto.FLOAT16,):
            _cast_initializer(init, TensorProto.FLOAT, min_positive)
    for node in model.graph.node:
        for attr in node.attribute:
            if attr.type == 7 and attr.t:
                _cast_tensor_proto(attr.t, TensorProto.FLOAT, min_positive)
            elif attr.type == 8:
                _recurse_subgraph(attr.g, TensorProto.FLOAT, min_positive)
            elif attr.type == 6:
                for t in attr.tensors:
                    _cast_tensor_proto(t, TensorProto.FLOAT, min_positive)
    for v in list(model.graph.input) + list(model.graph.output):
        if (
            v.type.HasField("tensor_type")
            and v.type.tensor_type.elem_type == TensorProto.FLOAT16
        ):
            v.type.tensor_type.elem_type = TensorProto.FLOAT
    return model


def _cast_initializer(
    init: TensorProto,
    target_dtype: int,
    min_positive: float,
) -> None:
    """Cast a single initializer (weight) tensor in-place."""
    if init.data_type == target_dtype:
        return
    np_array = numpy_helper.to_array(init)
    if target_dtype == TensorProto.FLOAT16:
        np_array = _to_fp16(np_array, min_positive)
    else:
        np_array = np_array.astype(np.float32)
    new_init = numpy_helper.from_array(np_array, name=init.name)
    init.data_type = new_init.data_type
    init.raw_data = new_init.raw_data
    init.dims[:] = new_init.dims


def _cast_tensor_proto(
    tp: TensorProto,
    target_dtype: int,
    min_positive: float,
) -> None:
    """Cast a TensorProto attribute in-place."""
    if tp.data_type == target_dtype:
        return
    np_array = numpy_helper.to_array(tp)
    if target_dtype == TensorProto.FLOAT16:
        np_array = _to_fp16(np_array, min_positive)
    else:
        np_array = np_array.astype(np.float32)
    new_tp = numpy_helper.from_array(np_array)
    tp.data_type = new_tp.data_type
    tp.raw_data = new_tp.raw_data
    tp.dims[:] = new_tp.dims


def _to_fp16(arr: np.ndarray, min_positive: float = 1e-7) -> np.ndarray:
    """Convert a float32 numpy array to float16, clamping subnormals."""
    return np.clip(arr.astype(np.float16), min_positive, None)


def _recurse_subgraph(graph_proto: Any, target_dtype: int, min_positive: float) -> None:
    """Recurse into nested subgraphs (e.g. If/Loop/Scan nodes)."""
    for init in graph_proto.initializer:
        if init.data_type == TensorProto.FLOAT and target_dtype == TensorProto.FLOAT16:
            _cast_initializer(init, target_dtype, min_positive)
        elif (
            init.data_type == TensorProto.FLOAT16 and target_dtype == TensorProto.FLOAT
        ):
            _cast_initializer(init, target_dtype, min_positive)
    for node in graph_proto.node:
        for attr in node.attribute:
            if attr.type == 7 and attr.t:
                _cast_tensor_proto(attr.t, target_dtype, min_positive)
            elif attr.type == 8:
                _recurse_subgraph(attr.g, target_dtype, min_positive)


# ---------------------------------------------------------------------------
# INT8 quantization (requires onnxruntime)
# ---------------------------------------------------------------------------


def quantize_int8(
    model: ModelProto,
    output_path: str | Path | None = None,
    quantization_mode: str = "default",
    per_channel: bool = False,
    weight_type: int | None = None,
    calibrate: bool = False,
    calibration_data: list[dict[str, np.ndarray]] | None = None,
) -> ModelProto:
    """Quantize an ONNX model to INT8 using ONNX Runtime.

    Args:
        model: Input ONNX ModelProto.
        output_path: If given, save the quantized model to this path.
        quantization_mode: ``"default"`` (dynamic for x86, static for ARM) or
            ``"dynamic"`` (weights only) or ``"static"`` (weights + activations).
        per_channel: Per-channel quantization for weights (default False).
        weight_type: Override weight type (e.g. ``TensorProto.INT8``).
        calibrate: Force static calibration even in default mode.
        calibration_data: List of dicts mapping input names to numpy arrays,
                          required for static quantization.

    Returns:
        Quantized ONNX ModelProto.

    Raises:
        ImportError: If ``onnxruntime`` is not installed.
        ValueError: If static quantization is requested without calibration data.
    """
    try:
        import onnxruntime.quantization as ort_quant
    except ImportError:
        raise ImportError(
            "INT8 quantization requires 'onnxruntime'. "
            "Install with: pip install onnxruntime or pip install onnx-model-surgery[quant]"
        )

    import tempfile

    model_path = (
        output_path or tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
    )

    # Save input model if we don't already have a path
    if output_path is None:
        onnx_save(model, model_path)

    extra_options = {
        "WeightSymmetric": True,
        "ActivationSymmetric": False,
        "PerChannel": per_channel,
    }

    if calibrate or quantization_mode == "static":
        if not calibration_data:
            raise ValueError(
                "Static INT8 quantization requires calibration_data. "
                "Pass at least one batch of representative inputs."
            )
        ort_quant.quantize_static(
            model_input=model_path,
            model_output=model_path,
            calibration_data_reader=_CalibrationDataReader(calibration_data),
            per_channel=per_channel,
            weight_type=weight_type or ort_quant.QuantType.QInt8,
            extra_options=extra_options,
        )
    elif quantization_mode == "dynamic":
        ort_quant.quantize_dynamic(
            model_input=model_path,
            model_output=model_path,
            per_channel=per_channel,
            weight_type=weight_type or ort_quant.QuantType.QInt8,
            extra_options=extra_options,
        )
    else:  # "default"
        # Let ONNX Runtime decide based on platform
        try:
            ort_quant.quantize_static(
                model_input=model_path,
                model_output=model_path,
                calibration_data_reader=_CalibrationDataReader(calibration_data or []),
                per_channel=per_channel,
                weight_type=weight_type or ort_quant.QuantType.QInt8,
                extra_options=extra_options,
            )
        except Exception:
            # Fall back to dynamic
            ort_quant.quantize_dynamic(
                model_input=model_path,
                model_output=model_path,
                per_channel=per_channel,
                weight_type=weight_type or ort_quant.QuantType.QInt8,
                extra_options=extra_options,
            )

    q_model = _load_model(model_path)
    return q_model


class _CalibrationDataReader:
    """Minimal calibration data reader for ONNX Runtime static quantization."""

    def __init__(self, data: list[dict[str, np.ndarray]]):
        self._data = data
        self._idx = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._idx >= len(self._data):
            return None
        batch = self._data[self._idx]
        self._idx += 1
        return batch

    def rewind(self) -> None:
        self._idx = 0


# ---------------------------------------------------------------------------
# Convenience: quantize a model file
# ---------------------------------------------------------------------------


def quantize_model_file(
    input_path: str | Path,
    output_path: str | Path,
    mode: str = "fp16",
    **kwargs: Any,
) -> str:
    """Load, quantize, and save an ONNX model in one call.

    Args:
        input_path: Path to source .onnx file.
        output_path: Path to save quantized .onnx file.
        mode: ``"fp16"``, ``"int8"``, ``"int8-dynamic"``, or ``"int8-static"``.
        **kwargs: Extra arguments passed to the specific quantizer.

    Returns:
        Path to the saved quantized model.

    Raises:
        ValueError: If mode is not recognised.
    """
    model = _load_model(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "fp16":
        q_model = convert_to_fp16(model)
        onnx_save(q_model, str(output_path))
    elif mode in ("int8", "int8-dynamic"):
        q_model = quantize_int8(model, quantization_mode="dynamic", **kwargs)
        onnx_save(q_model, str(output_path))
    elif mode == "int8-static":
        q_model = quantize_int8(
            model, calibration=True, quantization_mode="static", **kwargs
        )
        onnx_save(q_model, str(output_path))
    else:
        raise ValueError(
            f"Unknown quantization mode: {mode}. Use fp16, int8, int8-dynamic, or int8-static."
        )

    log.info("Quantized %s -> %s (mode=%s)", input_path, output_path, mode)
    return str(output_path)


__all__ = [
    "convert_to_fp16",
    "convert_model_precision",
    "quantize_int8",
    "quantize_model_file",
]
