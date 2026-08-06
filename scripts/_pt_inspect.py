"""Read PyTorch checkpoints without torch installed.

A `.pt` saved by `torch.save(obj, path)` (zip-format, default since 1.6)
is a zip containing:
  - `data.pkl`        — pickled Python object tree, with tensors stubbed
                        out via `_rebuild_tensor_v2(storage, …)`
  - `data/<n>`        — raw little-endian bytes for each tensor's storage,
                        keyed by integer id matching the pickle stubs

This module monkeypatches `pickle`'s find_class so that the stub functions
return a lightweight `TensorRecord(name, shape, dtype, offset, …)` instead
of a real torch.Tensor. The zip is kept open so `materialize()` can fetch
the bytes later as a numpy ndarray.

Tested against PyTorch 2.x checkpoints. Floats: f32 only (which is what
RamanPhysicsAI saves). Extend `_DTYPE_MAP` for other types.
"""
from __future__ import annotations

import io
import pickle
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# Map torch dtype string → numpy dtype.
_DTYPE_MAP: dict[str, np.dtype] = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "BoolStorage": np.dtype("?"),
    # Newer-style (torch >= 1.13)
    "torch.FloatTensor": np.dtype("<f4"),
}


@dataclass
class TensorRecord:
    """Lightweight stand-in for `torch.Tensor` extracted from a .pt zip."""
    storage_key: str  # e.g. "data/0"
    storage_dtype: np.dtype
    storage_numel: int  # total elements in the storage (may be larger than tensor)
    storage_offset: int  # element offset (NOT byte offset)
    size: tuple[int, ...]
    stride: tuple[int, ...]
    requires_grad: bool = False
    # Filled by inspect helpers
    _zip_ref: zipfile.ZipFile | None = field(default=None, repr=False)
    _archive_name: str = field(default="", repr=False)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.size

    @property
    def dtype(self) -> np.dtype:
        return self.storage_dtype

    def materialize(self) -> np.ndarray:
        """Read the raw bytes from the zip and shape into a contiguous numpy array."""
        if self._zip_ref is None:
            raise RuntimeError("TensorRecord has no zip reference; cannot materialize.")
        path = f"{self._archive_name}/{self.storage_key}"
        with self._zip_ref.open(path) as f:
            raw = f.read()
        arr = np.frombuffer(raw, dtype=self.storage_dtype, count=self.storage_numel)
        # Apply offset + shape. For typical state_dict tensors stride is row-major.
        if self.storage_offset:
            arr = arr[self.storage_offset:]
        # If stride is row-major, reshape works; we don't support exotic strided views.
        return arr.reshape(self.size).copy()


class _StubStorage:
    """Returned by find_class for the storage classes."""
    def __init__(self, dtype: np.dtype):
        self.dtype = dtype


class _CheckpointUnpickler(pickle.Unpickler):
    """Unpickler that catches torch-specific globals and substitutes stubs."""

    def __init__(self, file, *, zip_ref: zipfile.ZipFile, archive_name: str):
        super().__init__(file)
        self.zip_ref = zip_ref
        self.archive_name = archive_name
        self.records: list[TensorRecord] = []

    def find_class(self, module: str, name: str):
        # ---- torch.Storage subclasses (pickle protocol of FloatStorage etc.) ----
        if module == "torch" and name in (
            "FloatStorage", "DoubleStorage", "LongStorage",
            "IntStorage", "BoolStorage", "HalfStorage",
        ):
            return _StubStorage(_DTYPE_MAP.get(name, np.dtype("u1")))

        # ---- Tensor reconstruction functions ----
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            zip_ref = self.zip_ref
            archive_name = self.archive_name
            records = self.records

            def _rebuild_tensor_v2(storage, storage_offset, size, stride,
                                   requires_grad, backward_hooks, *_extra):
                # storage came from persistent_load below: it's our TensorRecord
                if not isinstance(storage, TensorRecord):
                    raise TypeError(f"Expected TensorRecord, got {type(storage)}")
                storage.size = tuple(size)
                storage.stride = tuple(stride)
                storage.storage_offset = int(storage_offset)
                storage.requires_grad = bool(requires_grad)
                storage._zip_ref = zip_ref
                storage._archive_name = archive_name
                records.append(storage)
                return storage
            return _rebuild_tensor_v2

        if module == "torch._utils" and name == "_rebuild_parameter":
            def _rebuild_parameter(data, requires_grad, backward_hooks):
                # `data` is already a TensorRecord
                if isinstance(data, TensorRecord):
                    data.requires_grad = bool(requires_grad)
                return data
            return _rebuild_parameter

        # ---- OrderedDict, dicts, basic collections ----
        if module == "collections":
            import collections
            return getattr(collections, name)

        # ---- numpy classes occasionally appear in configs ----
        if module.startswith("numpy"):
            import numpy as _np
            mod = _np
            for part in module.split(".")[1:]:
                mod = getattr(mod, part)
            return getattr(mod, name)

        # ---- Anything else: refuse with a clear error ----
        raise pickle.UnpicklingError(
            f"Refusing to load unsupported class {module}.{name}. "
            "Add a handler in _CheckpointUnpickler.find_class if you need it."
        )

    def persistent_load(self, pid):
        # PyTorch's persistent_id format for tensor storages:
        #   ('storage', stub_storage_class, key, location, numel)
        if not isinstance(pid, tuple) or len(pid) < 5:
            raise pickle.UnpicklingError(f"Unexpected persistent_id: {pid!r}")
        kind, stub_storage, key, location, numel = pid[:5]
        if kind != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent_id kind: {kind}")
        dtype = stub_storage.dtype if isinstance(stub_storage, _StubStorage) else _DTYPE_MAP.get(getattr(stub_storage, "__name__", "FloatStorage"), np.dtype("<f4"))
        # storage_key is the path "data/<n>" inside the zip (relative to archive root)
        rec = TensorRecord(
            storage_key=f"data/{key}",
            storage_dtype=dtype,
            storage_numel=int(numel),
            storage_offset=0,
            size=(),
            stride=(),
        )
        # Will be filled in by _rebuild_tensor_v2 above
        return rec


def load_checkpoint(path: str | Path) -> tuple[dict, zipfile.ZipFile, str]:
    """Open a `.pt` and return (top_level_object, open_zipfile, archive_name).

    Caller is responsible for closing the zipfile when done — it is needed
    to materialize tensor records lazily.
    """
    path = Path(path)
    z = zipfile.ZipFile(path, "r")
    # Find the archive root name (first segment of any member).
    names = z.namelist()
    if not names:
        raise ValueError(f"{path} is empty")
    archive_name = names[0].split("/", 1)[0]
    data_pkl = z.read(f"{archive_name}/data.pkl")
    up = _CheckpointUnpickler(io.BytesIO(data_pkl), zip_ref=z, archive_name=archive_name)
    obj = up.load()
    return obj, z, archive_name


def summarize_state_dict(sd: dict, max_keys: int = 30) -> None:
    """Print key → shape table for a state_dict's tensor records."""
    print(f"State-dict has {len(sd)} entries:")
    items = list(sd.items())
    for k, v in items[:max_keys]:
        if isinstance(v, TensorRecord):
            n = int(np.prod(v.size))
            print(f"  {k:<55s} {tuple(v.size)} {v.dtype} numel={n}")
        else:
            print(f"  {k:<55s} {type(v).__name__}")
    if len(items) > max_keys:
        print(f"  ... +{len(items)-max_keys} more")


__all__ = ["TensorRecord", "load_checkpoint", "summarize_state_dict"]
