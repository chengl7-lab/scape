## SCAPE-APA: a package for estimating alternative polyadenylation events from scRNA-seq data

### Installation

#### Environment setup
```
conda config --append channels bioconda 
conda config --append channels conda-forge 
conda create -n scape_env python=3.9
conda activate scape_env
```
#### PyPI installation
Install scape-apa via [pypi](https://pypi.org/project/scape-apa).
```
pip install scape-apa
```
#### Installation from our GitHub repository
```
git clone https://github.com/chengl7-lab/scape.git
cd scape
pip install .
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

Get help information of `scape` or `scape commands`.
```
scape --help
scape gen_utr_annotation --help
```

### Usage

#### gen_utr_annotation

| Input Argument |  Type | Required | Default | Description |
| --- | --- |  --- |  --- |  --- | 
| --gff_file           | TEXT | Yes | NA  |  The gff3 or gff3.gz file including annotation of gene body. |
| --output_dir         | TEXT | Yes | NA  | Directory to save dataframe of selected UTR. |
| --res_file_name      | TEXT | Yes | NA  | File Name of dataframe of the UTR annotation. The suffix `.csv` is automatically generated. |
| --gff_merge_strategy | TEXT | No  | merge | Method for processing overlapping regions. It follows `merge_strategy` in package gffutils. |

OUTPUT: An csv file including information of annotated 3UTR which is stored at `{output_dir}/{res_file_name}.csv`.

#### prepare_input

| Input Argument |  Type | Required | Default | Description |
| --- | --- |  --- |  --- |  --- | 
| --utr_file  | TEXT    |Yes | NA |  UTR annotation file (dataframe, resulted from gen_utr_annotation).|
| --cb_file   | TEXT    |Yes | NA | File of tsv.gz including all validated barcodes (by CellRanger). This file has one column of cell barcode which must be consistent with value of CB tag in bam_file file. |
| --bam_file  | TEXT    |Yes | NA | Bam file that is used for searching reads over annotated UTR.|
| --output_dir| TEXT    |Yes | NA | Directory to save pickle files of selected reads over annotated UTR. |
| --chunksize |INTERGER | No |1000 | Number of UTR regions included in each small pickle file, which contains the preprocessed input file for APA analysis. |

OUTPUT: Pickle files that include tuples (gene info, dataframe of parameter). 



