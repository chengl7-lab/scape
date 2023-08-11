import pybedtools
import gffutils
from .apa_core import Parameters, exp_pa_len, cal_exp_pa_len_by_cluster

import numpy as np
import pandas as pd
import pickle
from timeit import default_timer as timer
import os
import csv
import gzip
import click

"""
Author: Tien Le
Date: 22.06.2023
"""

""""------------------start----------------------------"""
# prepare.py
"""
Changes:
- Tien: 22/04/2023. Replace function pybedtools_integration.to_bedtool() by pybedtools.BedTool("\n".join(list_bed_line), from_string=True)
It return empty BedTools when applying sort() or merge() on BedTools object returned by pybedtools_integration.to_bedtool()
- Tien: 29/06/2022
- Tien: 11/08/2022. Extend UTR to downstream by 300bp
- Tien: 28/10/2022. Extend UTR to upstream by 300bp
"""
@click.command(name="gen_utr_annotation")
@click.option(
    '--gff_file',
    type=str,
    help='The gff3 (gz or not) file for preparing utr_file and intron region.',
    required=True
    )
@click.option(
    '--output_dir',
    type=str,
    help='Output directory to store all SCAPE results.',
    required=True
    )
@click.option(
    '--res_file_name',
    type=str,
    default='genes',
    help='The name for the utr annotation file without suffix. A .csv suffix is automatically appended.',
    required=True
    )
@click.option(
    '--gff_merge_strategy',
    type=str,
    default='merge',
    help='Merge strategy which is used when creating database from gff3 by gffutils.'
    )
def gen_utr_annotation(gff_file: str, output_dir: str, res_file_name: str, gff_merge_strategy: str):
    """
    Define possible 3' UTR regions.

    INPUT:
    - gff_file: directory to annotation GFF3 file. Should be .gff or .gff.gz. GFF3 file must be sorted by chromosome name and start position
        Explanation of GFF3: https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md
        Explanation of biotype used in GFF3: https://www.gencodegenes.org/pages/biotypes.html
                                             https://www.ensembl.org/info/genome/genebuild/biotypes.html
    - output_dir: path to store output_dir file
    - res_file_name: res_file_name name of output_dir file
    - gff_merge_strategy: merge strategy used in gffutils. Should be one of:
        "error" - by default
            Raise error
        "warning":
            Log a warning, which indicates that all future instances of the
            same ID will be ignored
        "merge":
            Combine old and new attributes -- but only if everything else
            matches; otherwise error. This can be slow, but is thorough.
            + features must have same (chrom, source, start, stop, score, strand, phase) to be merged
            + attributes of all features with the same primary key will be merged
            + primary key is ID by default
        "create_unique":
            Autoincrement based on the ID, always creating a new ID.
        "replace":
            Replaces existing database feature with `f`.

    OUTPUT:
    - Dataframe includes columns ["chrom", "start", "end", "strand", "gene_id", "utr_id"]. DF will be saved as csv file to directory in which script is running
    """
    # if not all([gff_file,output_dir,res_file_name]):
    #     cli(['gen_utr_annotation', '--help'])
    #     sys.exit(1)
        
        
    ## define name for output_dir file
    path_to_gff = os.path.splitext(os.path.abspath(gff_file))[0]

    annot_db_f = '{}.{}.db'.format(path_to_gff, gff_merge_strategy)

    ## if preparation was ran once for the same gff file, there is no need to rerun but load available output_dir instead
    if os.path.exists(annot_db_f):
        # if exsit, just read it
        print(f"Read existing database at {annot_db_f}")
        annot_db = gffutils.FeatureDB(annot_db_f)
    else:
        # else creating new database from gff3
        print(f"No available database at {annot_db_f}. Creation of new database from GFF3 will takes several hours.")
        start_t = timer()
        annot_db = gffutils.create_db(gff_file
                                      , annot_db_f
                                      , merge_strategy=gff_merge_strategy
                                      #                         , disable_infer_transcripts=True ## use when input is a gff
                                      #                         , disable_infer_genes=True ## use when input is a gff
                                      )
        end_t = timer()
        print(f"Creating database using gffutils from gff3 finished in {(end_t - start_t) / 60} min.")
    ## create an initial empty dataframe object
    final_utr_lst = []

    ## define list of gene type and RNA type
    gene_rna_dict = {"gene": ("mRNA", "transcript", "lnc_RNA", "ncRNA")
                     , "ncRNA_gene": ("miRNA", "lnc_RNA", "snRNA", "ncRNA", "snoRNA", "scRNA", "rRNA", "tRNA")
                     }
    
    ## search for utr pers gene type
    for g in gene_rna_dict:
        start_t = timer()
        final_utr_lst += search_utr(g, gene_rna_dict[g], annot_db, output_dir)
        end_t = timer()
        print(f'Finish {g} in {(end_t - start_t) / 60} minutes.')
    
    ## convert list of utr of all genes to dataframe
    final_utr_df = pd.DataFrame(final_utr_lst, columns=["chrom", "start", "end", "strand", "gene_id", "gene_name"])
    
    ## Indexing utr_file region for each gene
    final_utr_df["utr_id"] = final_utr_df.groupby(["gene_id"])["start"].rank(method="first", ascending=True).astype(
        "int64")
    print(f"Start writing to csv", os.path.join(output_dir, res_file_name + ".csv"))
    final_utr_df.to_csv(os.path.join(output_dir, res_file_name + '.csv'), index=False)
    ## end of gen_utr_annotation function

    
    
    
    
    
    
def search_utr(gene_type: str, rna_types: tuple, annot_db, output_dir: str):
    """
    Define 3' UTR for certain gene type and RNA types
    INPUT:
    - gene_type: gene, ncRNA_gene
    - rna_types: mRNA, transcript, lnc_RNA, ncRNA,...
    
    PROCESS:
    per one gene -> per one transcript -> find 3UTR or last exon
    
    OUTPUT:
    utr_lst: list of utr_file regions, [chrom, start, end, strand, gene_id, gene_name]
    """
    gene_iter = annot_db.features_of_type(featuretype=gene_type)
    utr_lst = []
    ## For each gene
    for gene in gene_iter:
        ## ignore any feature that is need to be experimentially confirmed "TEC"
        try:
            gene_biotype = gene.attributes["biotype"]
            if "TEC" in gene_biotype:
                continue
        except:
            pass

        gene_utr_lst = []

        ## list of children of gene that potentially have APA events
        RNA_child_iter = annot_db.children(id=gene
                                           , level=1
                                           , limit=(gene.seqid, gene.start, gene.end)
                                           #                                                , strand=gene.strand
                                           , featuretype=rna_types
                                           , completely_within=True
                                           ## child feature must be totally within region of gene
                                           )

        ## For each transcript
        ## For example: RNA_child can be one mRNA that belongs to gene
        for RNA_child in RNA_child_iter:
            ## ignore any feature that is need to be experimentially confirmed
            try:
                RNA_biotype = RNA_child.attributes["biotype"]
                if "TEC" in RNA_biotype:
                    continue
            except:
                pass

            ## list of 3' UTR of a mRNA
            UTR3_child_iter = annot_db.children(id=RNA_child
                                                , level=1
                                                , limit=(RNA_child.seqid, RNA_child.start, RNA_child.end)
                                                #                                                     , strand=RNA_child.strand
                                                , featuretype=("three_prime_UTR", "three_prime_utr", "3'-UTR")
                                                , completely_within=True
                                                ## child feature must be totally within region of rna
                                                )

            ## If exists any annotated 3' UTR for current mRNA, then continue the next transcript
            UTR3_child_lst = [utr for utr in UTR3_child_iter]
            if len(UTR3_child_lst) > 0:
                gene_utr_lst += UTR3_child_lst
                continue
            
            ## If no annotated 3UTR found ==> choose the last exon (toward 3'end)
            
            ## Take the last exon if we are processing "+"
            ## sort exons by their start position (on the left hand site)
            ## in descending order
            if gene.strand == "+":
                order_by="start"
                reverse=True
            ## Take the first exon if we are processing "-"
            ## sort exons by their end position (on the right hand site)
            ## in ascending order
            elif gene.strand == "-":
                order_by="end"
                reverse=False
            else:
                raise Exception(f"Cannot define strand of gene {gene.name}")
            
            exon_child_iter = annot_db.children(id=RNA_child
                                                , limit=(RNA_child.seqid, RNA_child.start, RNA_child.end)
                                                #                                                             , strand=RNA_child.strand
                                                , featuretype="exon"
                                                , level=1
                                                , order_by=order_by
                                                , reverse=reverse
                                                )
            ## if rna has no exon, then ignore
            try:
                utr_exon = next(exon_child_iter)
                gene_utr_lst.append(utr_exon)
            except:
                pass
        ## End of: for RNA_child in RNA_child_iter
        
        ## Only continue when there exists utr_file region for this gene
        ## Example: gene ENSMUSG00000094562 in mus musculus includes only V_gene_segment but we don't process such gene type here. So there is no UTR for ENSMUSG00000094562.
        if len(gene_utr_lst) == 0:
            continue
#     if len(gene_utr_lst) > 0:
        ## extend each utr_file region to both downstream and upstream by 300bp
        for utr in gene_utr_lst:
            utr.end += 300
            utr.start -= 300
        bed_lines = []
        for gene_utr in gene_utr_lst:
            bed_lines.append("\t".join([str(gene_utr.seqid)
                                           , str(max(gene_utr.start, 0))
                                           , str(max(gene_utr.end, 0))
                                           , ".", "0", str(gene_utr.strand)]))
        gene_utr_bed = pybedtools.BedTool("\n".join(bed_lines), from_string=True)
        try:
            os.remove(os.path.join(output_dir, "check.bed"))
        except OSError:
            pass
        
        gene_utr_bed.saveas(os.path.join(output_dir, "check.bed"))

        ## Processing normal chromosome. For example: chr1 or 1
        
        if not gene.seqid.upper().startswith(("MT", "CHRM", "MITO")):
            ## merge any utr_file that are within 500bp to each other into one utr_file
            gene_utr_bed = gene_utr_bed.sort().merge(d=500, s=True)

            ## Processing mitochondrial chromosome
        elif gene.seqid.upper().startswith(("MT", "CHRM", "MITO")):
            ## merge all utrs into one utr_file
            gene_utr_bed = gene_utr_bed.sort().merge(s=True)
        else:
            raise Exception(f"Cannot define if {gene.id} and {gene.seqid} is mitochondrial chromosome or not.")

        gene_utr_df = gene_utr_bed.to_dataframe()
        ## current strand is stored in column "name" due to function to_dataframe()
        gene_utr_df.rename(columns={'name': 'strand'}, inplace=True)
        
        ## Get gene_id from attributes
        try:
            gene_id = gene.attributes["gene_id"]
        except:
            print(gene.attributes)
            raise Exception("No attritbute gene_id.")

        ## Get gene name from attributes
        try:
            gene_name = gene.attributes["Name"]
        except:
            gene_name = ""
            print("No attribute Name for gene.")

        gene_utr_df["gene_id"] = ";".join([str(i) for i in gene_id])
        gene_utr_df["gene_name"] = ";".join([str(i) for i in gene_name])
        
        utr_lst += gene_utr_df[["chrom", "start", "end", "strand", "gene_id", "gene_name"]].values.tolist()
    ## End of: for gene in gene_iter
    
    return utr_lst
    ## End function search_utr
""""------------------end----------------------------"""

""""------------------start----------------------------"""

# apa_exp_pa_len.py
"""
CHANGES:
- Tien: 12/08/2022
"""
@click.command(name="cal_exp_pa_len")
@click.option(
    '--output_dir',
    type=str,
    help='Directory which was used in previous steps to save output by prepare_input and infer_pa',
    required=True
    )
@click.option(
    '--cell_cluster_file',
    type=str,
    default="None",
    help='An csv file containing two columns in order: cell barcode (CB) and respective group (cell_cluster_file). Its name will be included in the file name of final result.'
    )
@click.option(
    '--res_pkl_file',
    type=str,
    default="None",
    help='Name of res pickle file that contains PASs for calculating expected PA length. Its name will be included in the file name of final result.'
    )
#def apaexppalen(pkl_output_dir: str, pkl_input_dir: str, cell_cluster_file: str, out_file: str):
def cal_exp_pa_len(output_dir: str, cell_cluster_file: str, res_pkl_file: str):
    """
    INPUT:
    - output_dir: path to output_dir folder (to result files (class Parameters))
    - res_pkl: 
    - cell_cluster_file: path to dataframe of cell_cluster_file-cell barcode including 2 columns: cell barcode index and respective group

    OUTPUT:
    - exp_pa_len.csv : dataframe with 2 columns
    """
    if not os.path.exists(os.path.join(output_dir, "pkl_output")):
        raise Exception("Please use the same directory that stores res pickle files by infer_pa")
    if not os.path.exists(os.path.join(output_dir, "pkl_input")):
        raise Exception("Please use the same directory that stores res pickle files by prepare_input")
    
    final_res = os.path.join(output_dir, res_pkl_file)
    if not (os.path.exists(final_res)):
        raise Exception("Must run apajunction before apaexppalen")

    if not (os.path.exists(os.path.join(output_dir, "barcode_index.csv"))):
        raise Exception("Please use the same output directory as in prepare_input and infer_pa")
    cb_index = os.path.join(output_dir, "barcode_index.csv")
    cb_dict = pd.read_csv(cb_index, index_col="index").T.to_dict("list")

    dict_res = dict()  ## gene_id : [[absolute_alpha_arr], [label_arr], K, leftmost, rightmost, strand, cb_id]
    exp_len_lst = []  ## [gene_id, norm_length, K] or [gene_id, cell_cluster_file, norm_length, K] if cell_cluster_file!="None"

    if cell_cluster_file == "None":
        cluster_df = None
        output_path = os.path.join(output_dir, "all_cell." + res_pkl_file.replace(".pkl", ".pa.len.csv").replace("res.", ""))
    else:
        if not (os.path.exists(cell_cluster_file)):
            raise Exception("Given cell_cluster_file file does not exists")
        cluster_df = pd.read_csv(cell_cluster_file, index_col="index")  ## CB is index and group is column
        prefix = os.path.splitext(os.path.basename(cell_cluster_file))[0]
        output_path = os.path.join(output_dir, prefix +"."+ res_pkl_file.replace(".pkl", ".pa.len.csv").replace("res.", ""))

    start_t = timer()
    ## Load output_dir from scape and calculate expected pa length
    ## if in apajunction (handling junction step), we merge all utr_file as one for each gene, this expected length will be calculated w.r.t. this combined utr_file
    ## if in apajunction (handling junction step), we treat each utr_file of each gene separately, this expected length will be calculated w.r.t. this separate utr_file
    ## Calculate expected length for each gene
    with open(final_res, "rb") as final_res_fh:
        while True:
            try:
                para = pickle.load(final_res_fh)
                chrom, gene_id, utr_id, st_en, strand = para.gene_info_str.split(":")
                ## calculate expected len in general
                if cell_cluster_file == "None":
                    exp_len = exp_pa_len(para, para.label_arr)  ## exp_pa_len(apamix_res, label_arr)
                    exp_len_lst.append([gene_id + ":" + utr_id, exp_len, para.K])

                ## calculate expected len grouped by sample
                else:

                    #             # ---------------------------------
                    #             cluster_dict = cluster_df.iloc[:, 0].to_dict()
                    #             for cb_i in dict_res[gene][6]:
                    #                 exp_len_lst.append([cb_i, gene, cluster_dict[cb_i]])
                    #             #----------------------------------

                    cluster_dict = cluster_df.iloc[:, 0].to_dict()
                    ## array of cb_id that in a certain group of cells
                    partition = np.array([cluster_dict[cb] for cb in para.cb_id_arr.tolist()])
                    ## cal_exp_pa_len_by_cluster(apamix_res, partition, label_arr)
                    uni_clusters, avg_len_arr = cal_exp_pa_len_by_cluster(para, partition)
                    for idx in range(len(uni_clusters)):
                        exp_len_lst.append([gene_id + ":" + utr_id, uni_clusters[idx], avg_len_arr[idx], para.K])
            except EOFError:
                break

    end_t = timer()
    print(f"Done calculating expected pa length each gene in {(end_t - start_t) / 60} min.")

    if cell_cluster_file == "None":
        final_df = pd.DataFrame(exp_len_lst, columns=["gene_id", "exp_length", "num_pa"])
        final_df.to_csv(output_path
                        , header=True
                        , index=False
                        )
    else:
        final_par_df = pd.DataFrame(exp_len_lst, columns=["gene_id", "cell_cluster", "exp_length", "num_pa"])
        #         final_par_df = pd.DataFrame(exp_len_lst, columns=["cb_index", "gene_id", "cell_cluster_file"])
        final_par_df.to_csv(output_path
                            , header=True
                            , index=False
                            )
    end_t = timer()
    print(f"Done in {(end_t - start_t) / 60} min. ")


""""------------------end----------------------------"""

""""------------------start----------------------------"""
# apa_count_matrix.py
"""
CHANGES: 
- Tien: 05/04/2023
"""
@click.command(name="ex_pa_cnt_mat")
@click.option(
    '--output_dir',
    type=str,
    help='Directory which was used in previous steps to save output by prepare_input and infer_pa.',
    required=True
    )
@click.option(
    '--res_pkl_file',
    type=str,
    default="None",
    help='Name of res pickle file that contains PASs for calculating expected PA length. Its name will be included in the file name of final result.'
    )
def ex_pa_cnt_mat(output_dir: str, res_pkl_file: str):
    """
    INPUT:
    - output_dir: path to output_dir folder

    OUTPUT:
    - count matrix: where index are pa site information and columns are cell barcode. Being saved as a tuple (file_name, df)
    """

    # if not all([pkl_input_dir, pkl_output_dir]):
    #     cli(['apacount', '--help'])
    #     sys.exit(1)
    res_pkl = os.path.join(output_dir, res_pkl_file)
    if not (os.path.exists(output_dir)):
        raise Exception("Given output_dir folder does not exists.")
    if not (os.path.exists(res_pkl)):
        raise Exception(f"Invalid file {res_pkl}. Given res_pkl_file is not in output_dir.")

    outpath = os.path.join(output_dir, res_pkl_file.replace(".pkl", ".cnt.tsv.gz"))

    ## output_dir file
    cb_index = os.path.join(output_dir, "barcode_index.csv")
    cb_df = pd.read_csv(cb_index, index_col="index")

    start_t = timer()
    cb_lst = cb_df["CB"].tolist()
    count_list = []  ## each element is a dataframe of each gene

    ## Convert Parameters object to dataframe of count
    with open(res_pkl, "rb") as final_res_fh:
        while True:
            try:
                para = pickle.load(final_res_fh)
                ## gene info: 7:ENSG00000105793:9:90384891-90391455:+
                gene_info = para.gene_info_str.split(sep=":")
                st, en = gene_info[3].split(sep="-")
                ## only consider pa with label less than K (pa=K is uniform component)
                tmp_cnt_df = pd.concat([pd.Series(para.label_arr[para.label_arr < para.K]),
                                        pd.Series(cb_df.loc[para.cb_id_arr[para.label_arr < para.K], "CB"].tolist())
                                        ], axis=1, ignore_index=True)

                tmp_cnt_df.columns = ["label", "cb_file"]
                tmp_cnt_df["cnt"] = 1
                ## count number of read
                tmp_cnt_df = tmp_cnt_df.pivot_table(index=['label'], columns='cb_file', values='cnt').reset_index()
                ## add columns for missing barcode
                tmp_missing_cb_df = pd.DataFrame(np.zeros((tmp_cnt_df.shape[0]
                                                          , len([col for col in cb_lst if col not in tmp_cnt_df.columns]))
#                                                          , dtype=float
                                                        )
                                                , columns=[col for col in cb_lst if col not in tmp_cnt_df.columns]
                                                )

                tmp_cnt_df = pd.concat([tmp_cnt_df, tmp_missing_cb_df], axis=1)
                ## define full information of pa
                if gene_info[4] == "+":
                    tmp_cnt_df["pa_info"] = str(gene_info[0]) + \
                                            ":" + pd.Series(para.alpha_arr[tmp_cnt_df["label"]] + int(st)).astype(str) + \
                                            ":" + pd.Series(para.beta_arr[tmp_cnt_df["label"]]).astype(str) + \
                                            ":" + str(gene_info[4]) + \
                                            ":" + pd.Series(tmp_cnt_df["label"] + 1).astype(str) + \
                                            ":" + str(gene_info[1]) + \
                                            ":" + str(gene_info[2])
                else:
                    tmp_cnt_df["pa_info"] = str(gene_info[0]) + \
                                            ":" + pd.Series(int(en) - para.alpha_arr[tmp_cnt_df["label"]] + 1).astype(
                        str) + \
                                            ":" + pd.Series(para.beta_arr[tmp_cnt_df["label"]]).astype(str) + \
                                            ":" + str(gene_info[4]) + \
                                            ":" + pd.Series(tmp_cnt_df["label"] + 1).astype(str) + \
                                            ":" + str(gene_info[1]) + \
                                            ":" + str(gene_info[2])
                count_list.append(tmp_cnt_df[["pa_info"] + cb_lst])
            except EOFError:
                print(f"Finish counting for each gene")
                break
    end_t = timer()
    print(f"Finish {res_pkl} in {(end_t - start_t) / 60} min.")

    ## remove file of counts if exists
    try:
        os.remove(outpath)
    except OSError:
        pass

    ## Write counts to all_matrix_cnt.tsv.gz
    start_t = timer()
    with gzip.open(outpath, 'wt') as outcsv:
        # configure writer to write standard csv file
        writer = csv.writer(outcsv, delimiter=',', quoting=csv.QUOTE_ALL, lineterminator='\n')
        writer.writerow(["pa_info"] + cb_lst)
        for ele in count_list:
            ele.fillna(value=0, inplace=True)
            tmp_list = ele[["pa_info"] + cb_lst].values.tolist()
            for row_ in tmp_list:
                writer.writerow(row_)
                end_t1 = timer()
    end_t = timer()
    print(f"Finish writing all in {(end_t - start_t) / 60} min.")
""""------------------end----------------------------"""