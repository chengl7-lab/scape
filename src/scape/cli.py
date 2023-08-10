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
    print("Title: Your Paper Title")
    print("Authors: First Author, Second Author, Third Author")
    print("Journal: Journal Name")
    print("Year: 2023")
    print("DOI: https://doi.org/your-paper-doi")


cli.add_command(gen_utr_annotation)
cli.add_command(prepare_input)
cli.add_command(infer_pa)
cli.add_command(merge_pa)
cli.add_command(cal_exp_pa_len)
cli.add_command(ex_pa_cnt_mat)
