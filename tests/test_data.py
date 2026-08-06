"""Tests for src.data.dataloader and src.data.splits.

Run with:
    pytest tests/test_data.py -v

Some tests require a raw CSV in ``data/raw/`` to exist; they're skipped
otherwise. The fixture auto-resolves the CSV path: it prefers
``AA_Data.csv`` (the post-revision primary, see PROJECT_REVISION_v2 §1.1)
but falls back to ``data.csv`` for backwards compatibility.

Tests requiring torch are skipped if torch isn't installed.

History
-------
* T03 (Groundwork): original split tests with API
  ``select_split / split_A_composition_ood / split_B_component_ood / ...``
* T07 (Phase A): split API replaced — see CHAT2_PHASE_A_HANDOVER §C
  decisions C6, and Issue #1 in §B. ``TestSplits`` rewritten to match
  the new ``split_A_vial_level / split_A_sample_level / save_split /
  load_split`` API. Scheme B and the ``select_split`` dispatcher are
  gone.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CFG = PROJECT_ROOT / "configs" / "default.yaml"
DATA_CFG = PROJECT_ROOT / "configs" / "data_config.yaml"


def _resolve_data_csv() -> Path | None:
    """Find the raw CSV — prefer AA_Data.csv, fall back to data.csv.

    Returns None if neither exists, so fixtures can ``pytest.skip``.
    """
    raw_dir = PROJECT_ROOT / "data" / "raw"
    for name in ("AA_Data.csv", "data.csv"):
        candidate = raw_dir / name
        if candidate.is_file():
            return candidate
    return None


DATA_CSV = _resolve_data_csv()


# ───────────────────────────────────────────────────────────────────────────
#  Fixtures
# ───────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def configs():
    with open(DEFAULT_CFG, encoding='utf-8') as f:
        defaults = yaml.safe_load(f)
    with open(DATA_CFG, encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    return defaults, data_cfg


@pytest.fixture(scope="module")
def table(configs):
    """Load the raw CSV once per test module."""
    if DATA_CSV is None:
        pytest.skip(
            f"No raw CSV found under {PROJECT_ROOT / 'data' / 'raw'}; "
            "expected AA_Data.csv or data.csv."
        )
    from src.data.dataloader import load_raw_csv
    defaults, _ = configs
    return load_raw_csv(
        csv_path=DATA_CSV,
        compound_full_names=defaults["compounds"]["full_names"],
        laser_wl_nm=defaults["wavenumber"]["laser_wavelength_nm"],
        expected_num_points=defaults["wavenumber"]["expected_num_points"],
    )


# ───────────────────────────────────────────────────────────────────────────
#  Helper-function tests (no CSV required)
# ───────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_wavelength_to_wavenumber_self_zero(self):
        from src.data.dataloader import wavelength_nm_to_wavenumber_cm_inv
        laser = 784.815734863281
        result = wavelength_nm_to_wavenumber_cm_inv(np.array([laser]), laser)
        assert abs(result[0]) < 1e-6

    def test_wavelength_to_wavenumber_known_values(self):
        """A wavelength of 800 nm with a 784.8157... nm laser should yield
        a positive Raman shift in the few-hundred cm⁻¹ range."""
        from src.data.dataloader import wavelength_nm_to_wavenumber_cm_inv
        wn = wavelength_nm_to_wavenumber_cm_inv(np.array([800.0]), 784.815734863281)
        # Expect roughly 240 cm⁻¹
        assert 200 < wn[0] < 300

    @pytest.mark.parametrize("vial,expected", [
        ("L-histidine",     True),
        ("D-glucosamine",   True),
        ("L-alanine",       True),
        ("L-asparagine",    True),
        ("L-aspartic-acid", True),
        ("L-glutamic-acid", True),
        ("Histidine",       True),
        ("DL-histidine",    True),
        ("a01",             False),
        ("a48",             False),
        ("a12",             False),
    ])
    def test_is_pure_vial(self, vial, expected):
        from src.data.dataloader import is_pure_vial
        compounds = ['Alanine', 'Asparagine', 'Aspartic Acid',
                     'Glutamic Acid', 'Histidine', 'Glucosamine']
        assert is_pure_vial(vial, compounds) == expected


# ───────────────────────────────────────────────────────────────────────────
#  CSV loading tests
# ───────────────────────────────────────────────────────────────────────────

class TestLoadRawCSV:
    def test_shape(self, table):
        assert table.spectra.shape == (table.num_samples, table.num_points)
        assert table.labels.shape == (table.num_samples, table.num_compounds)
        assert table.vial_ids.shape == (table.num_samples,)

    def test_dtypes(self, table):
        assert table.spectra.dtype == np.float32
        assert table.labels.dtype == np.float32

    def test_no_nans(self, table):
        assert not np.isnan(table.spectra).any()
        assert not np.isnan(table.labels).any()

    def test_label_simplex(self, table):
        sums = table.labels.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-3)

    def test_wavenumber_monotonic(self, table):
        diffs = np.diff(table.wavenumbers)
        assert (diffs > 0).all() or (diffs < 0).all(), \
            "Wavenumber axis must be strictly monotonic."

    def test_pure_count_at_least_six(self, table):
        n_pure = int(table.pure_mask().sum())
        assert n_pure >= 6, f"Expected at least 6 pure rows, got {n_pure}"

    def test_metadata_columns_ignored(self, tmp_path, configs):
        """AA_Data.csv adds 3 non-numeric metadata columns (file_name,
        Repitation, mix_method) vs. legacy data.csv. They must be
        silently dropped, not parsed as wavelengths.

        See PROJECT_REVISION_v2 §1.1 and the bug fixed in dataloader.py
        (`_is_numeric_header` filter).
        """
        import pandas as pd
        from src.data.dataloader import load_raw_csv

        df = pd.DataFrame({
            "file_name":  ["x"] * 4,                       # metadata text
            "Repitation": [1, 2, 3, 4],                    # metadata int
            "801.62":     np.random.rand(4).astype(np.float32),
            "802.0":      np.random.rand(4).astype(np.float32),
            "802.5":      np.random.rand(4).astype(np.float32),
            "vial #":     ["aa01"] * 4,
            "mix_method": ["Hand-Mixed"] * 4,              # metadata text
            "Alanine":       [0.4] * 4,
            "Asparagine":    [0.0] * 4,
            "Aspartic Acid": [0.0] * 4,
            "Glutamic Acid": [0.6] * 4,
            "Histidine":     [0.0] * 4,
            "Glucosamine":   [0.0] * 4,
        })
        csv = tmp_path / "aa_like.csv"
        df.to_csv(csv, index=False)

        defaults, _ = configs
        tbl = load_raw_csv(
            csv_path=csv,
            compound_full_names=defaults["compounds"]["full_names"],
            laser_wl_nm=defaults["wavenumber"]["laser_wavelength_nm"],
        )
        # Only the 3 numeric headers should be treated as spectra.
        assert tbl.spectra.shape == (4, 3)
        assert tbl.labels.shape == (4, 6)
        # Metadata cols must NOT leak into wavelengths.
        assert tbl.wavelengths_nm.tolist() == [801.62, 802.0, 802.5]


# ───────────────────────────────────────────────────────────────────────────
#  Dataset / DataLoader tests (require torch)
# ───────────────────────────────────────────────────────────────────────────

class TestDataset:
    def test_dataset_item_shape(self, table):
        torch = pytest.importorskip("torch")
        from src.data.dataloader import RamanDataset
        ds = RamanDataset(table, indices=np.arange(min(10, table.num_samples)))
        item = ds[0]
        assert isinstance(item, dict)
        assert set(item.keys()) == {"spectrum", "label", "vial_id", "index"}
        assert item["spectrum"].shape == (1, table.num_points)
        assert item["label"].shape == (table.num_compounds,)
        assert item["spectrum"].dtype == torch.float32
        assert item["label"].dtype == torch.float32
        assert isinstance(item["vial_id"], str)

    def test_dataloader_batch_shape(self, table):
        torch = pytest.importorskip("torch")
        from src.data.dataloader import RamanDataset, raman_collate
        from torch.utils.data import DataLoader

        ds = RamanDataset(table, indices=np.arange(min(64, table.num_samples)))
        loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=raman_collate)
        batch = next(iter(loader))
        assert batch["spectrum"].shape == (8, 1, table.num_points)
        assert batch["label"].shape == (8, table.num_compounds)
        assert len(batch["vial_id"]) == 8
        assert not torch.isnan(batch["spectrum"]).any()

    def test_empty_indices_raises(self, table):
        from src.data.dataloader import RamanDataset
        with pytest.raises(ValueError):
            RamanDataset(table, indices=np.array([], dtype=np.int64))

    def test_out_of_bounds_indices_raises(self, table):
        from src.data.dataloader import RamanDataset
        with pytest.raises(IndexError):
            RamanDataset(table, indices=np.array([0, table.num_samples + 5]))

    def test_build_dataloaders_returns_three(self, table):
        pytest.importorskip("torch")
        from src.data.dataloader import build_dataloaders, SplitIndices
        n = table.num_samples
        split = SplitIndices(
            train=np.arange(0, int(0.8 * n)),
            val=np.arange(int(0.8 * n), int(0.9 * n)),
            test=np.arange(int(0.9 * n), n),
            scheme="test_fixture",
        )
        loaders = build_dataloaders(table, split, batch_size=32, num_workers=0)
        assert set(loaders.keys()) == {"train", "val", "test"}


# ───────────────────────────────────────────────────────────────────────────
#  Split scheme tests — Phase A T07 API
# ───────────────────────────────────────────────────────────────────────────
#
# Migration note (T07 -> Phase A):
#   * `select_split(table, cfg)`          → REMOVED (no dispatcher)
#   * `split_A_composition_ood(table,…)`  → `split_A_vial_level(vial_ids, labels,…)`
#   * `split_A_prime_random(table,…)`     → `split_A_sample_level(n_rows, labels,…)`
#   * `split_B_component_ood(...)`        → REMOVED (49/54 vials contain every
#                                            compound; train pool collapses)
#   * `save_split_to_json / load_split_from_json` → `save_split / load_split`
#
# The new API takes plain arrays (vial_ids list + labels ndarray) instead
# of a `RawSpectraTable`, so each test bridges via `table.vial_ids` and
# `table.labels`.

class TestSplits:
    def test_scheme_A_disjoint_vials(self, table):
        """Scheme A: every vial must end up in exactly one of train/val/test."""
        from src.data.splits import split_A_vial_level
        sp = split_A_vial_level(
            vial_ids=list(table.vial_ids),
            labels=np.asarray(table.labels),
            seed=42,
            n_train_vials=42,
            n_val_vials=6,
            n_test_vials=6,
            include_pure_in_train=True,
        )
        v_tr = set(np.asarray(table.vial_ids)[sp.train])
        v_v  = set(np.asarray(table.vial_ids)[sp.val])
        v_te = set(np.asarray(table.vial_ids)[sp.test])
        assert (v_tr & v_v)  == set()
        assert (v_tr & v_te) == set()
        assert (v_v  & v_te) == set()
        # Sanity: every spectrum is accounted for.
        assert len(sp.train) + len(sp.val) + len(sp.test) == table.num_samples

    def test_scheme_A_pure_vials_in_train(self, table):
        """When include_pure_in_train=True (default), all 6 pure vials end up
        in the train split. This protects compound coverage and the
        reconstruction module's reference set.
        """
        from src.data.splits import split_A_vial_level, is_pure_vial
        sp = split_A_vial_level(
            vial_ids=list(table.vial_ids),
            labels=np.asarray(table.labels),
            seed=42,
            include_pure_in_train=True,
        )
        train_vials = set(np.asarray(table.vial_ids)[sp.train])
        pure_vials_in_train = {v for v in train_vials if is_pure_vial(v)}
        # All pure vials should be in train; none should leak to val/test.
        val_vials = set(np.asarray(table.vial_ids)[sp.val])
        test_vials = set(np.asarray(table.vial_ids)[sp.test])
        assert not any(is_pure_vial(v) for v in val_vials)
        assert not any(is_pure_vial(v) for v in test_vials)
        assert len(pure_vials_in_train) == 6

    def test_scheme_A_prime_disjoint_indices(self, table):
        """Scheme A' (sample-level): all 4378 indices partition into train/val/test."""
        from src.data.splits import split_A_sample_level
        sp = split_A_sample_level(
            n_rows=table.num_samples,
            labels=np.asarray(table.labels),
            seed=42,
            train_frac=0.6, val_frac=0.2, test_frac=0.2,
        )
        s_tr = set(sp.train)
        s_v  = set(sp.val)
        s_te = set(sp.test)
        assert (s_tr & s_v)  == set()
        assert (s_tr & s_te) == set()
        assert (s_v  & s_te) == set()
        assert len(sp.train) + len(sp.val) + len(sp.test) == table.num_samples

    def test_scheme_A_prime_invalid_fractions(self, table):
        """Fractions that don't sum to 1 must raise."""
        from src.data.splits import split_A_sample_level
        with pytest.raises(ValueError):
            split_A_sample_level(
                n_rows=table.num_samples,
                labels=np.asarray(table.labels),
                seed=42,
                train_frac=0.5, val_frac=0.3, test_frac=0.3,
            )

    def test_json_round_trip(self, table, tmp_path):
        """save_split → load_split must yield identical lists + scheme name."""
        from src.data.splits import (
            split_A_vial_level, save_split, load_split,
        )
        sp = split_A_vial_level(
            vial_ids=list(table.vial_ids),
            labels=np.asarray(table.labels),
            seed=42,
        )
        fpath = tmp_path / "split.json"
        save_split(sp, fpath)
        sp2 = load_split(fpath)
        assert list(sp.train) == list(sp2.train)
        assert list(sp.val)   == list(sp2.val)
        assert list(sp.test)  == list(sp2.test)
        assert sp.scheme == sp2.scheme
        assert sp.seed == sp2.seed


# ───────────────────────────────────────────────────────────────────────────
#  Preprocessing tests (T04)
# ───────────────────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_remove_cosmic_rays_preserves_shape(self):
        from src.data.preprocess import remove_cosmic_rays
        s = np.random.RandomState(0).randn(1024).astype(np.float32) * 100 + 5000
        out = remove_cosmic_rays(s)
        assert out.shape == s.shape
        assert out.dtype == s.dtype

    def test_remove_cosmic_rays_reduces_spike(self):
        """Inject a clear cosmic spike; the algorithm must reduce it."""
        from src.data.preprocess import remove_cosmic_rays
        rng = np.random.RandomState(0)
        s = (rng.randn(1024) * 50 + 5000).astype(np.float32)
        s_spiked = s.copy()
        s_spiked[500] = s[500] * 100  # huge cosmic
        s_clean = remove_cosmic_rays(s_spiked, threshold=5.0)
        assert abs(s_clean[500] - s[500]) < abs(s_spiked[500] - s[500])

    def test_savgol_window_must_be_odd(self):
        from src.data.preprocess import savitzky_golay
        s = np.random.RandomState(0).randn(1024).astype(np.float32)
        with pytest.raises(ValueError):
            savitzky_golay(s, window=10, polyorder=3)

    def test_savgol_polyorder_lt_window(self):
        from src.data.preprocess import savitzky_golay
        s = np.random.RandomState(0).randn(1024).astype(np.float32)
        with pytest.raises(ValueError):
            savitzky_golay(s, window=5, polyorder=5)

    def test_snv_invariants(self):
        from src.data.preprocess import snv_normalize
        rng = np.random.RandomState(0)
        s = (rng.randn(1024) * 1000 + 5000).astype(np.float32)
        s_n = snv_normalize(s)
        assert abs(s_n.mean()) < 1e-3
        assert abs(s_n.std() - 1.0) < 1e-3

    def test_snv_flat_input_safe(self):
        """Constant spectrum should not divide-by-zero, must return mean-centred."""
        from src.data.preprocess import snv_normalize
        s = np.full(1024, 1234.0, dtype=np.float32)
        out = snv_normalize(s)
        assert np.isfinite(out).all()

    def test_pipeline_deterministic(self, table):
        from src.data.preprocess import preprocess_pipeline
        s = table.spectra[0]
        a = preprocess_pipeline(s)
        b = preprocess_pipeline(s)
        assert np.array_equal(a, b)

    def test_pipeline_output_invariants(self, table):
        from src.data.preprocess import preprocess_pipeline
        s = preprocess_pipeline(table.spectra[0])
        assert s.shape == table.spectra[0].shape
        assert abs(s.mean()) < 1e-3
        assert abs(s.std() - 1.0) < 1e-3
        assert not np.isnan(s).any()

    def test_is_preprocessed_flag(self, table):
        from src.data.preprocess import preprocess_pipeline
        already = preprocess_pipeline(table.spectra[0])
        passthrough = preprocess_pipeline(already, is_preprocessed=True)
        assert np.array_equal(already, passthrough)

    def test_preprocess_batch_shape(self, table):
        from src.data.preprocess import preprocess_batch
        sub = table.spectra[:8]
        out = preprocess_batch(sub)
        assert out.shape == sub.shape
        assert out.dtype == sub.dtype
        # Each row should be SNV-normalized
        assert np.allclose(out.mean(axis=1), 0, atol=1e-3)
        assert np.allclose(out.std(axis=1), 1, atol=1e-3)

    def test_make_preprocess_fn_returns_none_when_disabled(self):
        from src.data.preprocess import make_preprocess_fn
        cfg = {"preprocessing": {"apply_on_the_fly": False}}
        assert make_preprocess_fn(cfg) is None

    def test_make_preprocess_fn_returns_callable_when_enabled(self):
        from src.data.preprocess import make_preprocess_fn
        cfg = {"preprocessing": {"apply_on_the_fly": True,
                                  "cosmic_threshold": 5.0,
                                  "asls_lam": 1e5, "asls_p": 0.01, "asls_max_iter": 10,
                                  "savgol_window": 11, "savgol_polyorder": 3,
                                  "snv_eps": 1e-8}}
        fn = make_preprocess_fn(cfg)
        assert callable(fn)
        s = np.random.RandomState(0).randn(1024).astype(np.float32) * 100 + 5000
        out = fn(s)
        assert out.shape == s.shape