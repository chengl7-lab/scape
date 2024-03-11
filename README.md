## SCAPE-APA: a package for estimating alternative polyadenylation events from scRNA-seq data

### Installation

Environment setup.
```
conda config --append channels bioconda 
conda config --append channels conda-forge 
conda create -n scape_env python=3.9
conda activate scape_env
```
Install scape-apa via [pypi](https://pypi.org/project/scape-apa).
```
pip install scape-apa
```
### Commands

| Command | Description |
| --- | --- |
| scape gen_utr_annotation | Generate UTR annotation. |
| scape prepare_input | Prepare data per UTR. |
| scape infer_pa | Parameters inference. |
| scape merge_pa | Merge PA within junction per gene or UTR. |
| scape cal_exp_pa_len | Calculate the expected length of PA. |
| scape ex_pa_cnt_mat | Extract read count matrix. |


### Usage
Get help information of `scape` or `scape commands`.
```
scape --help
scape gen_utr_annotation --help
```

