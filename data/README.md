# Dataset Files

The hyperspectral data files are not included in this repository.

Download the public HyMars benchmark from Science Data Bank:

https://www.scidb.cn/en/detail?dataSetId=4ff0774d45464f239a73f37796f7a786

For details about the dataset, please refer to:

Bobo Xi, Yun Zhang, Jiaojiao Li, et al. HyMars: Mars Hyperspectral Image Classification Benchmark Datasets[DS/OL]. V2. Science Data Bank, 2025[2025-01-20]. https://doi.org/10.57760/sciencedb.19732.

Then place the MATLAB files in this directory with the following names and keys:

| Dataset | Data file | Data key | Label file | Label key | Classes |
| --- | --- | --- | --- | --- | --- |
| HC | `holden.mat` | `holden` | `holden_gt.mat` | `holden_gt` | 6 |
| NF | `NiliFossae.mat` | `NiliFossae` | `NiliFossae_gt.mat` | `NiliFossae_gt` | 9 |
| UP | `Utopia.mat` | `Utopia` | `Utopia_gt.mat` | `Utopia_gt` | 9 |

The `.gitignore` file excludes `.mat`, `.npy`, and `.npz` data files by default.
