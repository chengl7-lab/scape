conda config --append channels bioconda 
conda config --append channels conda-forge 
conda create -n test_conda python=3.9
conda activate test_conda

conda install pip
conda install -c bioconda bedtools
pip install taichi
pip install .

scape --help
scape gen_utr_annotation --help

python -m scape --help
python -m scape gen_utr_annotation --help



