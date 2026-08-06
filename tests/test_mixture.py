import sys; sys.path.insert(0, '.')
import numpy as np, torch
from engine.peak_extractor import PeakExtractor
from engine.symbolic_mapper import BondMapper
from engine.novelty_locator import NoveltyLocator

wn = np.load('data/processed/wavenumbers.npy')
X = torch.load('data/processed/spectra_full.pt', weights_only=True).numpy()
Y = torch.load('data/processed/labels.pt', weights_only=True).numpy()

# Pick a 50-50 Histidine-Glucosamine mixture (or as close as possible)
target = np.array([0., 0., 0., 0., 0.5, 0.5])
dists = np.linalg.norm(Y - target, axis=1)
i = int(np.argmin(dists))
print(f'Using mixture row {i}, labels {Y[i].round(3)} (target was 50-50 His-Glc)')

ext = PeakExtractor(wn)
mapper = BondMapper.from_json('engine/bond_mapping.json')

peaks = ext.extract_full(X[i])
print(f'Extracted {len(peaks)} peaks')
ann = mapper.annotate_peaks(peaks)
d = mapper.disambiguate_compound(ann)
print(f"Likely compounds: {d['likely_compounds']}")
votes_str = ", ".join([f"{k}: {v:.1f}" for k, v in d['votes'].items() if v > 0])
print(f"Votes: {{{votes_str}}}")