## SCAPE-APA: a package for estimating alternative polyadenylation events from scRNA-seq data

### Installation

Environment setup
```
conda config --append channels bioconda 
conda config --append channels conda-forge 
conda create -n scape_env python=3.9
conda activate scape_env
```
Install scape-apa via [pypi](https://pypi.org/project/scape-apa)
```
pip install scape-apa
```
### Usage
```
scape --help
```
Commands:
  cal_exp_pa_len      INPUT: - output_dir: path to output_dir folder (to...
  ex_pa_cnt_mat       INPUT: - output_dir: path to output_dir folder
  gen_utr_annotation  Define possible 3' UTR regions.
  infer_pa            INPUT: - pkl_input_file: file path (pickle)...
  merge_pa            INPUT: - output_dir: directory to + pkl_output...
  prepare_input       INPUT:cb_df - utr_file: path to utr_file df -...

```
scape gen_utr_annotation --help
```



