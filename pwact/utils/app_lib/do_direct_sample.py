"""
DIRECT sampling template script run inside PWact iteration working directories.

Uses MatPL's built-in NEP-DIRECT — no maml/m3gnet dependency.

Reads `direct.json` (if present) for configuration:
    {"nep_model": "../../00.train/000/model_record/nep5.txt",
     "threshold": 0.04, "k": 1, "pool": "mean"}
"""
import json
import os
import sys

import numpy as np
from ase.io import read

# Make MatPL importable
_matpl_root = os.environ.get("MATPL_ROOT")
if _matpl_root is None:
    _matpl_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../MatPL"))
    if os.path.exists(_matpl_root):
        os.environ["MATPL_ROOT"] = _matpl_root
if _matpl_root not in sys.path:
    sys.path.insert(0, _matpl_root)

from src.sampling.direct import DIRECTSampler, BirchClustering, SelectKFromClusters
from src.sampling.nep_describer import NEPStructure

# --------------- Read config ---------------
cfg = {
    "nep_model": None,
    "threshold": 0.04,
    "k": 1,
    "pool": "mean",
    "cell_angle_min": 40.0,
    "cell_angle_max": 140.0,
}
if os.path.exists("direct.json"):
    with open("direct.json") as f:
        cfg.update(json.load(f))

if cfg["nep_model"] is None:
    raise ValueError("nep_model must be specified in direct.json (path to nep5.txt or nep_model.ckpt)")

write_file = "select.xyz"
if os.path.exists(write_file):
    os.remove(write_file)

# --------------- Load structures ---------------
filenames = ["candidate.xyz"]

def load_ase_MD_traj(filenames):
    structs, trajs, lens = [], [], []
    for fn in filenames:
        traj = read(fn, index=":")
        structs += list(traj)
        trajs.append(traj)
        lens.append(len(traj))
    return structs, trajs, lens

structures, trajs, lens = load_ase_MD_traj(filenames)
n_image = len(structures)
print(f"Loaded {n_image} candidate structures")

# --------------- NEP encoder + DIRECT pipeline ---------------
encoder = NEPStructure(cfg["nep_model"], pool=cfg["pool"])
print(f"Using NEP descriptor (dim={encoder.dim}) from {cfg['nep_model']}")

sampler = DIRECTSampler(
    structure_encoder=encoder,
    clustering=BirchClustering(n=None, threshold_init=cfg["threshold"]),
    select_k_from_clusters=SelectKFromClusters(k=cfg["k"]),
)
selection = sampler.fit_transform(structures)
n, m = selection["PCAfeatures"].shape

# --------------- Plots ---------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

explained_variance = sampler.pca.pca.explained_variance_ratio_
selection["PCAfeatures_unweighted"] = selection["PCAfeatures"] / explained_variance[:m]

plt.plot(range(1, explained_variance.shape[0] + 1), explained_variance * 100, "o-")
plt.xlabel(r"i$^{\mathrm{th}}$ PC", size=20)
plt.ylabel("Explained variance", size=20)
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig("PCA_variance.png", dpi=360)
plt.close()

all_features = selection["PCAfeatures_unweighted"]
selected_indexes = selection["selected_indexes"]

fig, ax = plt.subplots(figsize=(5, 5))
plt.plot(all_features[:, 0], all_features[:, 1], "*", alpha=0.5,
         label=f"All {len(all_features):,} structures")
plt.plot(all_features[selected_indexes, 0], all_features[selected_indexes, 1], "*",
         alpha=0.5, label=f"DIRECT sampled {len(selected_indexes):,}")
plt.legend(frameon=False, fontsize=14, loc="upper left",
           bbox_to_anchor=(-0.02, 1.02), reverse=True)
plt.ylabel("PC 2", size=20)
plt.xlabel("PC 1", size=20)
plt.tight_layout()
plt.savefig("PCA_direct.png", dpi=360)
plt.close()

def feature_coverage_score(col, sel_idx, n_bins=100):
    bins = np.linspace(col.min(), col.max(), n_bins)
    n_all = np.count_nonzero(np.histogram(col, bins=bins)[0])
    n_sel = np.count_nonzero(np.histogram(col[sel_idx], bins=bins)[0])
    return n_sel / max(n_all, 1)

scores = [feature_coverage_score(all_features[:, i], selected_indexes)
          for i in range(all_features.shape[1])]
x = np.arange(len(scores))

plt.figure(figsize=(15, 4))
plt.bar(x, scores, width=0.3,
        label=f"DIRECT, $\\overline{{\\mathrm{{Coverage\\ score}}}}$ = {np.mean(scores):.3f}")
plt.xticks(x, [f"PC {n+1}" for n in range(len(x))], size=16)
plt.yticks(np.linspace(0, 1.0, 6), size=16)
plt.ylabel("Coverage score", size=20)
plt.legend(shadow=True, loc="lower right", fontsize=16)
plt.tight_layout()
plt.savefig("Cov_score.png", dpi=360)
plt.close()

# --------------- Filter & write ---------------
def get2index(num, list_lens):
    for idx, i in enumerate(list_lens):
        if num >= i:
            num -= i
        else:
            break
    return idx, num

indices = selection["selected_indexes"]
select_idx = []
for index in indices:
    idx, num = get2index(index, lens)
    atoms = trajs[idx][num]
    angles = atoms.cell.cellpar()[-3:]
    if angles.max() > cfg["cell_angle_max"] or angles.min() < cfg["cell_angle_min"]:
        continue
    atoms.set_scaled_positions(atoms.get_scaled_positions())
    atoms.write(write_file, format="extxyz", append=True)
    select_idx.append(idx)

np.savetxt("select_idx.dat", np.array(indices), fmt="%8d")
print(f"DIRECT done: wrote {len(select_idx)} structures from {len(indices)} clusters")
