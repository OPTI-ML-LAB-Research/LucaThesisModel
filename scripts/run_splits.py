import numpy as np, torch
from src.data.splits import split_A_vial_level, split_A_sample_level, save_split

labels   = torch.load('data/processed/labels.pt').numpy()
vial_ids = np.load('data/processed/vial_ids.npy', allow_pickle=True).tolist()

sa  = split_A_vial_level(vial_ids, labels, seed=42)
sap = split_A_sample_level(len(vial_ids), labels, seed=42)

save_split(sa,  'data/splits/split_A_composition_ood.json')
save_split(sap, 'data/splits/split_A_prime_sample_level.json')
print(f"A : train={len(sa.train)}  val={len(sa.val)}  test={len(sa.test)}")
print(f"A': train={len(sap.train)}  val={len(sap.val)}  test={len(sap.test)}")