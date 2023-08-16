from collections import Counter, defaultdict

import numpy as np
import pysam
import pandas as pd
import pickle
from timeit import default_timer as timer
import os
import click
from glob import glob
#from cli.apa_junction import flag_junction_read, proc_junction_pos_read, proc_junction_neg_read
from .junction_handler import flag_junction_read, proc_junction_pos_read, proc_junction_neg_read

"""
Author: Tien Le
Date: 22.06.2023
"""



"""
CHANGES:
- Tien: 11/08/2022 Add code for processing bulk data
- Tien: 28/10/2022 Add function for detecting junction reads (flag_junction_read()) for single cell RNA-seq data only
                    Add unique read_id to each dataframe of paramater per gene-utr_file region. 
                        Different gene-utr_file region can have the same read_id but they means different reads.
- Tien: 09/08/2023 Change chunksize to 100 in prepare_input. When 1000, the inference using apa_core with no GPU took more than 2 days to finish.
NOTE:
- For single-cell RNA-deq data from 10X, Cell Ranger ver >3.1 must be used to have tag "pa"
    Read 1 includes cell barcode and UMI only
    Read 2 includes sequence fragment
- For bulk RNA-seq data, STAR alginment can be used with additional two parameters:
    --bamRemoveDuplicatesType UniqueIdentical ## to have duplicated reads marked as SAM flag
    --outSAMattributes NH HI AS nM GX GN XS ## to have gene name and gene ID information in BAM file
    Need to have query name in BAM file
"""


@click.command(name="prepare_input")
@click.option(
    '--utr_file',
    type=str,
    help='UTR annotation file (dataframe, resulted of gen_utr_annotation) ',
    required=True
    )
@click.option(
    '--cb_file',
    type=str,
    default="None",
    help='Path to barcode.tsv.gz which includes all desired barcodes (output of CellRanger). This file must be consistent with CB tag in bam_file file',
    required=True
    )
@click.option(
    '--bam_file',
    type=str,
    help='Path to bam_file file (output of CellRanger).',
    required=True
    )
@click.option(
    '--output_dir',
    type=str,
    help='Output directory.',
    required=True
    )
@click.option(
    '--chunksize',
    type=int,
    default=100,
    help='Number of UTR regions included in each sub-file, which contains preprocessed input file for APA analysis.'
    )
# @click.option(
#     '--seqtype',
#     type=click.Choice(["single-cell", "bulk"]),
#     default="single-cell",
#     help ='To know if input data is from single-cell RNA sequencing, or bulk RNA sequencing. Values must be [single-cell, bulk]'
#     )
#def apainput(utr_file: str, cb_file: str, bam_file: str, output_dir: str, chunksize: int, seqtype:str):
def prepare_input(utr_file: str, cb_file: str, bam_file: str, output_dir: str, chunksize: int, seqtype = "single-cell"):
    """
    INPUT:cb_df
    - utr_file: path to utr_file df
    - cb_file: path to CB file
    - bam_file: path to bam_file file
    - output_dir: path to output_dir folder
    - chunksize: number of object per output_dir file
    - sortBAMbyName: bam_file file need to be sorted by query name for faster performance of searching for mate when processing bulk data
    
    OUTPUT: Pickle files in which include tuple (gene info, dataframe of parameter).
            Example of file name: Chr1.100.1.input.pkl
            Dataframe of parameter includes columns: "x", "l", "r", "pa", "cb_id", "read_id", "junction"
    """
    # if not all([utr_file,cb_file,bam_file,output_dir]):
    #     cli(['prepare_input', '--help'])
    #     sys.exit(1)
        
    print(f"Processing ", bam_file)
    
    ## check if input files exist
    if not (os.path.exists(bam_file)):
        raise Exception("Given Bam file does not exists")
    if (not (os.path.exists(cb_file))) and (cb_file != "None"):
        raise Exception("Given barcode file does not exists")
    if not (os.path.exists(utr_file)):
        raise Exception("Given UTR file does not exists")
    
    ## create output directory if not exists
    os.makedirs(os.path.join(output_dir, "pkl_input"), exist_ok=True)
        
    ## generate prefix of output files
    outfile = os.path.join(output_dir, "pkl_input", os.path.basename(bam_file)[:-4])
    
    ## create log file for this chromosome
    log_filename = outfile+".log"
    if os.path.exists(log_filename):
        os.remove(log_filename)
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    ## delete all input.pkl files (that are corresponding to one considering bam file) in directory
    for filename in glob(os.path.join(output_dir
                                      , "pkl_input"
                                      , f"{os.path.basename(bam_file)[:-4]}*input.pkl"
                                     )
                        ):
        os.remove(filename)
    # Write log messages
    logging.info("All pickle files respective to "+os.path.basename(bam_file)[:-4]+" in pkl_input are deleted.")

    start_t1 = timer()
    ## Read file of barcode if exists, otherwise create the new one
    if cb_file == "None":
        cb_df = pd.DataFrame([1], columns=["index"], index=["1"])
        cb_df.index.name = "CB"
        cb_df.to_csv(os.path.join(output_dir, "barcode_index.csv"))
    else:
        if not os.path.exists(os.path.join(output_dir, "barcode_index.csv")):
            cb_df = cell_barcode_index(cb_file, output_dir)
            print(f"Create index file for barcode ", os.path.join(output_dir, "barcode_index.csv"))
        else:
            cb_df = pd.read_csv(os.path.join(output_dir, "barcode_index.csv"), index_col="CB")
            print(f"Read existing index file for barcode", os.path.join(output_dir, "barcode_index.csv"))


    ## read list of chromosome name from bam file
    chr_name_lst = bam_chr_name(bam_file)
    if chr_name_lst is None:
        raise Exception("Bam file does not have header information")


    ## Process each UTR region
    utr_df = pd.read_csv(utr_file)
    n_obj = 1 ## to count number of elements in current pickle file. If >chunksize, then create a new pickle file.
    cnt = 0 ## to count number of total generated pickle files.

    ## Process each UTR
    # Write log messages
    logging.info(outfile + "." + str(chunksize) + ".tmp." + str(1) + ".input.pkl is in processed")
    for idx in range(len(utr_df)):
        
        left_site = utr_df.loc[idx, "start"]
        right_site = utr_df.loc[idx, "end"]
        chr_name_ori = str(utr_df.loc[idx, "chrom"])

        ## generate chromosome name that is consistent with chromosome name used in bam file => use when querying reads in given region by pysam
        chr_name=com_chr_name(chr_name_lst, chr_name_ori)
        if chr_name is None:
            continue

        ## extract information of strand, utr, gene
        strand = utr_df.loc[idx, "strand"]
        utr_id = utr_df.loc[idx, "utr_id"]
        gene_id = utr_df.loc[idx, "gene_id"]

        # Process reads within selected UTR region
        if seqtype == "single-cell":
            out_tuple = proc_10x_bam_file(str(chr_name), int(left_site), int(right_site), strand
                                          , gene_id, int(utr_id)
                                          , bam_file, cb_df)
        else:
            raise Exception(f'Invalid value of seqtype')

        # Write tuple to output_dir/pkl_input/
        if cnt <= chunksize:
            outfile_ = outfile + "." + str(chunksize) + ".tmp." + str(n_obj) + ".input.pkl"
            
            # append to file
            with open(outfile_, 'a+b') as f:
                if (not out_tuple[1].empty) and (out_tuple[1].shape[0] > 100):
                    ## include region with at least 100 reads
                    pickle.dump(out_tuple, f)
                    cnt += 1
#                     print(f"Save ", outfile_)
        else:
            # Write log messages
            logging.info(outfile + "." + str(chunksize) + ".tmp." + str(n_obj) + ".input.pkl is successfully processed")
            # start to write to the new file
            cnt = 0
            n_obj += 1
            outfile_ = outfile + "." + str(chunksize) + ".tmp." + str(n_obj) + ".input.pkl"
            # Write log messages
            logging.info(outfile + "." + str(chunksize) + ".tmp." + str(n_obj) + ".input.pkl is in processed")

            # append to file
            with open(outfile_, 'a+b') as f:
                if (not out_tuple[1].empty) and (out_tuple[1].shape[0] > 100): 
                    ## include region with at least 100 reads
                    pickle.dump(out_tuple, f)
                    cnt += 1
#                     print(f"Save ", outfile_)

    ## End processing each UTR region
    end_t1 = timer()
    print(f'Done {cnt} files in {(end_t1 - start_t1)/60} minutes.')
    
    ## rename file: replacing "tmp" by the total number of files for this chromosome
    for idx in range(1, n_obj+1):
        os.rename(outfile + "." + str(chunksize) + ".tmp." + str(idx) + ".input.pkl"
                  , outfile + "." + str(chunksize) +"."+ str(n_obj) +"."+ str(idx) + ".input.pkl")
    
    # Write log messages
    logging.info("FINISHED")
    logging.info("There are "+str(n_obj)+" pickle input files for this BAM file "+bam_file)
    logging.info("Phrase tmp in name of pickle files is replaced by "+str(n_obj))
    


    


## ===========================================================================================================
## CORE FUNCTIONS


## ===========================================================================================================
## ============================================= SINGLE-CELL RNA-SEQ =========================================
## ============================================ CASE: SINGLE-END READ ========================================
## ========================== CASE: Read 2 is on the same strand of transcript ===============================


## process one UTR region (utr_id) respective to 1 gene_id
def proc_10x_bam_file(chr_name: str, left_site: int, right_site: int, strand: str
                      , gene_id:str, utr_id:int
                      , bamfile: str, cb_df):
    """
    Receive information of one UTR region and search for reads that are within considered region. 
    This script is applicable to standard output from 10X, where R1 includes CB and UMI, R2 includes molecular fragment.
    
    INPUT: 
    - chr_name: chromosome name
    - left_site: leftmost coordinate of UTR region (relative position on "+" strand)
    - right_site: rightmost coordinate of UTR region (relative position on "+" strand)
    - strand: strand of UTR region. Should be "+" or "-"
    - gene_id: gene_id (ex: ENSMUSG00000000827) which is store as gene_id in GTF
    - utr_id: index of UTR region
    - bamfile: directory to bam_file file
    - CBfile: directory to cell barcode file. Can be .tsv, .tsv.gz, .csv, .csv.gz
    
    OUTPUT:
    tuple: (gene_info, dataframe of selected reads)
    gene_info: chr:gene_ID:UTR_ID:st-ed:strand
    dataframe of selected reads: has columns ["x", "l", "r", "pa", "read_id", "cb_id"]
    """
#     cb_df = cell_barcode_index(CBfile)
#     print(chr_name, left_site, right_site, strand, gene_id, utr_id)
    if strand == "+":
        return proc_10x_bam_file_pos(chr_name, left_site, right_site, strand
                                      , gene_id, utr_id
                                      , bamfile, cb_df)
    elif strand == "-":
        return proc_10x_bam_file_neg(chr_name, left_site, right_site, strand
                                      , gene_id, utr_id
                                      , bamfile, cb_df)
    else:
        raise Exception(f'unknown strand={strand}, strand must be either + or -.')


## create look-up table for barcode and its index
def cell_barcode_index(CBfile: str, output:str):
    """
    Indexing cell barcode
    
    INPUT:
    - CBfile: directory to cell barcode file. Can be .tsv, .tsv.gz, .csv, .csv.gz
    
    OUTPUT:
    - Dataframe: index is cell barcode and one column "index"
    
    """
    cb_df = pd.read_csv(CBfile, names=["CB"])
    cb_df.reset_index(inplace=True)
    cb_df.set_index("CB", inplace=True)
    
    ## This will save a csv file to directory in which script is running
    cb_df.to_csv(os.path.join(output, "barcode_index.csv"))
    ## to read csv: pd.read_csv("barcode_index.csv", index_col="CB")
    return cb_df


## comparison chromosome name in utr.csv and bam file
def bam_chr_name(bamfile):
    with open(bamfile, 'rb') as bfh:
        py_bamfile = pysam.AlignmentFile(bfh)
        for line in py_bamfile.fetch():
            return(list(line.header.references))
    return None
def com_chr_name(chr_lst, chr_name):
    if chr_name in chr_lst:
        return chr_name
    elif (chr_name.startswith("chr")) & (chr_name[3:] in chr_lst):
        return chr_name[3:]
    elif (("chr"+str(chr_name)) in chr_lst):
        return "chr"+str(chr_name)
    else:
        return None
    

        
## processing strand "+" from single cell rna sequencing data
def proc_10x_bam_file_pos(chr_name: str, left_site: int, right_site: int, strand: str
                          , gene_id:str, utr_id:int
                          , bamfile: str, cb_df):
    """
    Receive information of one UTR region and search for reads that are within considered region.
    This script is applicable to standard output from 10X, where R1 includes CB and UMI, R2 includes molecular fragment.
    
    INPUT: 
    - chr_name: chromosome name
    - left_site: leftmost coordinate of UTR region (relative position on "+" strand)
    - right_site: rightmost coordinate of UTR region (relative position on "+" strand)
    - strand: strand of UTR region. Should be "+" or "-"
    - gene_id: gene_id (ex: ENSMUSG00000000827) which is store as gene_id in GTF
    - utr_id: index of UTR region
    - bamfile: directory to bam_file file
    - cb_df: look-up table (dataframe) of cell barcode
    
    OUTPUT:
    tuple: (gene_info, dataframe of selected reads)
    """
    out_tbl = []
    assert strand == "+"
    with open(bamfile, 'rb') as bfh:
        py_bamfile = pysam.AlignmentFile(bfh)
        for line in py_bamfile.fetch(chr_name, left_site, right_site):
            
            ## read_qc() == True means that read passes quality control checks
            if not read_qc(line):
                continue
                
            ## ignore if read maps to reverse strand
            elif line.is_reverse:
                continue
                
            ## process read 2 on "+" strand
            else:
#                 ref_st = line.reference_start - left_site # x
                ref_st = line.reference_start + line.query_alignment_start - left_site # x

                # start position (x) should be within [left_site, right_site], ignore if not satisfied
                if ref_st < 0:
                    continue

                r2_utr_len = line.query_alignment_length  # l
                
                ## ignore if part of read is out_file of considered region
#                 if (line.reference_start +r2_utr_len) >= right_site:
                if (line.reference_start + line.query_alignment_start +r2_utr_len) >= right_site:
                    continue

                cell_barcode, umi = read_CB_UMI(line)
                ## look for index of cell barcode
                cell_barcode_idx = read_CB_check(cell_barcode, cb_df)
                
                ## ignore if CB is not validated by cel ranger
                if cell_barcode_idx is None:
                    continue
                    
                ## check if read contain poly A tail
                ## tag pa is given by Cell Ranger
                if line.has_tag('pa'): 
                    pa_site = ref_st + r2_utr_len - 1
                else:
                    pa_site = np.nan
                
                ## set length of read 1 to null because in standard 10X, R1 contains CB and UMI only
                r = np.nan
                
                ## flag for junction read
                junc = flag_junction_read(line)
                ## derive absolute position of the 3'-end of the two segments from junction read
                if junc == 1:
                    ## segment 1 is always on 5'-end w.r.t. "+" strand
                    seg1_en, seg2_en = proc_junction_pos_read(line)
                else:
                    seg1_en, seg2_en = np.nan, np.nan
                
                ## append array of current read to output list
                out_tbl.append([ref_st # x
                                , r2_utr_len # l
                                , r # r
                                , pa_site # pa
                                , utr_id
                                , cell_barcode
                                , cell_barcode_idx
                                , umi
                                , strand
                                , junc
                                , seg1_en, seg2_en
                               ])
        
        gene_info, out_df = list_read_to_dataframe(out_tbl, chr_name, gene_id, utr_id, left_site, right_site, strand)
    return (gene_info, out_df)



## processing strand "-" from single cell rna sequencing data
def proc_10x_bam_file_neg(chr_name: str, left_site: int, right_site: int, strand: str
                          , gene_id:str, utr_id:int
                          , bamfile: str, cb_df):
    """
    Receive information of one UTR region and search for reads that are within considered region. 
    This script is applicable to standard output from 10X, where R1 includes CB and UMI, R2 includes molecular fragment.
    
    INPUT: 
    - chr_name: chromosome name
    - left_site: leftmost coordinate of UTR region (relative position on "+" strand)
    - right_site: rightmost coordinate of UTR region (relative position on "+" strand)
    - strand: strand of UTR region. Should be "+" or "-"
    - gene_id: gene_id (ex: ENSMUSG00000000827) which is store as gene_id in GTF
    - utr_id: index of UTR region
    - bamfile: directory to bam_file file
    - cb_df: look-up table (dataframe) of cell barcode
    
    OUTPUT:
    tuple: (gene_info, dataframe of selected reads)
    """
    out_tbl = []
    assert strand == "-"
    with open(bamfile, 'rb') as bfh:
        py_bamfile = pysam.AlignmentFile(bfh)
        for line in py_bamfile.fetch(chr_name, left_site, right_site):
            
#             ## read_qc() == True means that read passes quality control checks
            if not read_qc(line):
                continue
                 
            ## ignore if read maps to forward strand  
            elif line.is_forward:
                continue
                
            ## process read 2 on "-" strand
            else:
#                 cnt+=1 # debug
#                 ref_st = right_site - (line.reference_start + line.query_alignment_length - 1) # x
                ref_st = right_site - (line.reference_start + line.query_alignment_start + line.query_alignment_length - 1) # x

                # start position (x) should be within [left_site, right_site], not processing if not satisfied
                if ref_st <= 0:
                    continue

                r2_utr_len = line.query_alignment_length  # l
                
                ## ignore if part of read is out_file of considered region
#                 if line.reference_start < left_site:
                if line.reference_start + line.query_alignment_start < left_site:
                    continue
            
                cell_barcode, umi = read_CB_UMI(line)
                ## look for index of cell barcode
                cell_barcode_idx = read_CB_check(cell_barcode, cb_df)
                
                ## ignore if CB is not validated by cel ranger
                if cell_barcode_idx is None:
                    continue
                    
                ## check if read contain poly A tail 
                ## tag pa is given by Cell Ranger
                if line.has_tag('pa'): 
#                     pa_site = right_site - line.reference_start
                    pa_site = right_site - (line.reference_start + line.query_alignment_start)
                else:
                    pa_site = np.nan
                    
                ## set length of read 1 to null because in standard 10X, R1 contains CB and UMI only
                r = np.nan
                
                ## flag for junction read
                if "N" in line.cigarstring:
                    junc = flag_junction_read(line)
                else:
                    junc=0
                ## derive absolute position of the 3'-end of the two segments from junction read
                if junc == 1:
                    ## segment 1 is always on 5'-end w.r.t. "+" strand
                    seg1_en, seg2_en = proc_junction_neg_read(line)
                else:
                    seg1_en, seg2_en = np.nan, np.nan
                
                ## append array of current read to output list
                out_tbl.append([ref_st # x
                                , r2_utr_len # l
                                , r # r
                                , pa_site # pa
                                , utr_id
                                , cell_barcode
                                , cell_barcode_idx
                                , umi
                                , strand
                                , junc
                                , seg1_en, seg2_en
                               ])
        
        gene_info, out_df = list_read_to_dataframe(out_tbl, chr_name, gene_id, utr_id, left_site, right_site, strand)
    return (gene_info, out_df)


## Quality control per read
def read_qc(line):
    ## ignore reads with low mapping quality
    if line.mapping_quality != 255:
        return False

    ## ignore (PCR or optical) duplicated reads
    ## works only if alignment tool adds to sam flags.
    if line.is_duplicate: 
        ## Cell Ranger marks 1 read as main read, and others as duplicated (sam flag = 1024 or 1040)
        return False

    ## ignore reads that cannot map to any annotated gene
    ## this tag GX is given by Cell Ranger
    if not line.has_tag("GX"):
        return False

    else:
        ## ignore reads that map to more than 1 gene
        if ";" in line.get_tag("GX"):
            return False

    ## ignore read 1
    if line.is_read1:
        return False
    
    ## ignore if read does not have valid cell barcode and UMI
    ## these tags are given by Cell Ranger
    if not (line.has_tag("CB") and line.has_tag("UB")):
        return False
        
    return True


## Derive cell barcode (CB) and UMI (UB) from read
def read_CB_UMI(line):
    cell_barcode = line.get_tag("CB")
    # just to check UMI is unique, could be removed after check
    umi = line.get_tag("UB")  
    return cell_barcode, umi


## Check if cell barcode of read is validated by cell ranger and there exists corresponding index of CB
def read_CB_check(cell_barcode, cb_df):
    ## ignore read whose cell barcode is not in list of provided barcode
    if cell_barcode not in cb_df.index:
        return None
    ## look for index of cell barcode
    cell_barcode_idx = cb_df.loc[cell_barcode, "index"]
    return cell_barcode_idx


## convert list of reads that corresponding to UTR to dataframe after processing all satisfied reads
def list_read_to_dataframe(out_tbl, chr_name, gene_id, utr_id, left_site, right_site, strand):
    out_df = pd.DataFrame(out_tbl
                          , columns = ["x"
                                       , "l"
                                       , "r"
                                       , "pa"
                                       , "utr_id"
                                       , "cb_file"
                                       , "cb_id"
                                       , "umi"
                                       , "strand"
                                       , "junction"
                                       , "seg1_en", "seg2_en"
                                      ]
                         )

    ## only keep first read whose umi is duplicated (not due to PCR/optical) in each cell
    out_df.drop_duplicates(subset=["umi", "cb_file"], keep="first"
                           , inplace=True, ignore_index=True
                          )
    out_df["read_id"] = out_df.index
#     out_df["seg1_en"] = out_df["seg1_en"].astype('Int64')
#     out_df["seg2_en"] = out_df["seg2_en"].astype('Int64')
#     out_df["pa"] = out_df["pa"].astype('Int64')
#     out_df["r"] = out_df["r"].astype('Int64')
    gene_info = str(chr_name)+":"+str(gene_id)+":"+str(utr_id)+":"+str(left_site)+"-"+str(right_site)+":"+str(strand)
    return gene_info, out_df[["x", "l", "r", "pa", "cb_id", "read_id", "junction", "seg1_en", "seg2_en"]]

                 
