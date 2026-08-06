"""Raman Physics-Informed AI MVP — learnable code modules.

Sub-packages:
    data       — datasets, dataloaders, preprocessing, splits
    models     — backbone, heads, reconstruction, uncertainty, baselines
    training   — losses, training loop, callbacks
    inference  — predict(), OOD scoring, report generation, visualization
    eval       — metrics, benchmark runner

The non-learned symbolic modules live in the top-level `engine/` package.
"""

__version__ = "0.1.0.dev0"
