from maml.sampling.direct import DIRECTSampler, BirchClustering, SelectKFromClusters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from ase.io import read
import subprocess, os, sys

write_file = "select.xyz"
if os.path.exists(write_file):
    os.remove(write_file)
filenames = ["candidate.xyz"]
k = 1
threshold = .04
def load_ase_MD_traj(filenames: list):
    """
    Load .traj to pymatgen structures
    """
    structs = []
    trajs = []
    lens = []
    for filename in filenames:
        traj = read(filename,index=":")
        structs += [i for i in traj]
        trajs.append(traj)
        lens.append(len(traj))
    return structs, trajs, lens

structures, trajs, lens = load_ase_MD_traj(filenames)
n_image = len(structures)

DIRECT_sampler = DIRECTSampler(
    clustering=BirchClustering(n=None, threshold_init=threshold), select_k_from_clusters=SelectKFromClusters(k=k)
)

DIRECT_selection = DIRECT_sampler.fit_transform(structures)
n, m = DIRECT_selection["PCAfeatures"].shape

explained_variance = DIRECT_sampler.pca.pca.explained_variance_ratio_
DIRECT_selection["PCAfeatures_unweighted"] = DIRECT_selection["PCAfeatures"] / explained_variance[:m]

plt.plot(
    range(1, explained_variance.shape[0]+1),
    explained_variance * 100,
    "o-",
)
plt.xlabel("i$^{\mathrm{th}}$ PC", size=20)
plt.ylabel("Explained variance", size=20)
ax = plt.gca()
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.tight_layout()
plt.savefig("PCA_variance.png",dpi=360)
plt.close()

def plot_PCAfeature_coverage(all_features, selected_indexes, method="DIRECT"):
    fig, ax = plt.subplots(figsize=(5, 5))
    selected_features = all_features[selected_indexes]
    plt.plot(all_features[:, 0], all_features[:, 1], "*", alpha=0.5, label=f"All {len(all_features):,} structures")
    plt.plot(
        selected_features[:, 0],
        selected_features[:, 1],
        "*",
        alpha=0.5,
        label=f"{method} sampled {len(selected_features):,}",
    )
    legend = plt.legend(frameon=False, fontsize=14, loc="upper left", bbox_to_anchor=(-0.02, 1.02), reverse=True)
    #for lh in legend.legendHandles:
    #    lh.set_alpha(1)
    plt.ylabel("PC 2", size=20)
    plt.xlabel("PC 1", size=20)

all_features = DIRECT_selection["PCAfeatures_unweighted"]
selected_indexes = DIRECT_selection["selected_indexes"]
plot_PCAfeature_coverage(all_features, selected_indexes)
plt.tight_layout()
plt.savefig("PCA_direct.png",dpi=360)
plt.close()

#manual_selection_index = np.arange(0, n_image, int(n_image/n))
#plot_PCAfeature_coverage(all_features, manual_selection_index, "Manually")
#plt.tight_layout()
#plt.savefig("PCA_manually.png",dpi=360)
#plt.close()

def calculate_feature_coverage_score(all_features, selected_indexes, n_bins=100):
    selected_features = all_features[selected_indexes]
    n_all = np.count_nonzero(
        np.histogram(all_features, bins=np.linspace(min(all_features), max(all_features), n_bins))[0]
    )
    n_select = np.count_nonzero(
        np.histogram(selected_features, bins=np.linspace(min(all_features), max(all_features), n_bins))[0]
    )
    return n_select / n_all

def calculate_all_FCS(all_features, selected_indexes, b_bins=100):
    select_scores = [
        calculate_feature_coverage_score(all_features[:, i], selected_indexes, n_bins=b_bins)
        for i in range(all_features.shape[1])
    ]
    return select_scores

all_features = DIRECT_selection["PCAfeatures_unweighted"]
scores_DIRECT = calculate_all_FCS(all_features, DIRECT_selection["selected_indexes"], b_bins=100)
#scores_MS = calculate_all_FCS(all_features, manual_selection_index, b_bins=100)
x = np.arange(len(scores_DIRECT))
x_ticks = [f"PC {n+1}" for n in range(len(x))]

plt.figure(figsize=(15, 4))
plt.bar(
    x,
    scores_DIRECT,
    width=0.3,
    label=f"DIRECT, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_DIRECT):.3f}",
)
#plt.bar(
#    x + 0.3, scores_MS, width=0.3, label=f"Manual, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_MS):.3f}"
#)
plt.xticks(x, x_ticks, size=16)
plt.yticks(np.linspace(0, 1.0, 6), size=16)
plt.ylabel("Coverage score", size=20)
plt.legend(shadow=True, loc="lower right", fontsize=16)
plt.tight_layout()
plt.savefig("Cov_score.png",dpi=360)
plt.close()

def get2index(num: int, list_lens: list):
    for idx, i in enumerate(list_lens):
        if num >= i:
            num -= i
        else:
            break
    return idx, num

indices = DIRECT_selection["selected_indexes"]
select_idx = []
for ii,index in enumerate(indices):
    idx, num = get2index(index, lens)
    atoms = trajs[idx][num]
    angles = atoms.cell.cellpar()[-3:]
    if angles.max() > 140 or angles.min() < 40:
        continue
    else:
        atoms.set_scaled_positions(atoms.get_scaled_positions())
        atoms.write(write_file,format="extxyz",append=True)
        select_idx.append(idx)
np.savetxt("select_idx.dat",np.array(indices),fmt="%8d")
