import click
from .input_processor import prepare_input
from .junction_handler import merge_pa
from .apa_core import infer_pa
from .utils import gen_utr_annotation, cal_exp_pa_len, ex_pa_cnt_mat

@click.group()
def cli():
    """
    An analysis framework for estimating alternative polyadenylation events from single cell RNA-seq data.
    Current version: 1.0.0
    """
    pass

def display_paper_info():
    print()
    print("This software is affiliated with the following paper:")
    print("SCAPE-APA: a package for estimating alternative polyadenylation events from scRNA-seq data")
    print(f"Guangzhao Cheng\N{SUPERSCRIPT ONE}, Tien Le\N{SUPERSCRIPT ONE}, Ran Zhou, Lu Cheng\N{SUPERSCRIPT PLUS SIGN}")
    print("BioRxiv")
    print("2024")
    print("https://doi.org/10.1101/2024.03.12.584547")
    print()


cli.add_command(gen_utr_annotation)
cli.add_command(prepare_input)
cli.add_command(infer_pa)
cli.add_command(merge_pa)
cli.add_command(cal_exp_pa_len)
cli.add_command(ex_pa_cnt_mat)
