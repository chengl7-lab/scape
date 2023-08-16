import numpy as np
import pandas as pd
import pickle
from timeit import default_timer as timer
import os
import click
from .apa_core import Parameters

"""
Author: Tien Le
Date: 22.06.2023
"""


"""
- Created: 28/10/2022
- 10/08/2023: only process merge_pa when all input pickle files were successfully processed in infer_pa and each of them is respective to one res pickle file.
    
NOTED:
- Order of reads in the final dataframe must be fixed and the same as order resulted from apa_core -> solved by adding readID to class Parameters
- No uniform component after merging
"""

## ==============================================================
## Call function to handle pa site within junction site
## ==============================================================
@click.command(name="merge_pa")
@click.option(
    '--output_dir',
    type=str,
    help='Directory which was used in previous steps to save output by `prepare_input` and `infer_pa`',
    required=True
    )
@click.option(
    '--utr_merge',
    type=bool,
    default=True,
    help='By default, True if want to process all pa site of one gene at together. False if want to process each utr_file separately.'
    )
#def apajunction(pkl_input_dir:str, pkl_output_dir:str, utr_merge=True):
def merge_pa(output_dir:str, utr_merge=True):
    """
    INPUT:
        - output_dir: directory to 
            + pkl_output which is including files ".res.pkl" from apacore: each object is a parameter class
            + pkl_input which is includingfiles ".input.pkl" from apainput: each object is a tuple
                                    (pa_info, dataframe of input parameter (x,l,r,pa,cb_id,read_id,junction,seg1_en,seg2_en))
                                    
    """
    if not os.path.exists(os.path.join(output_dir, "pkl_output")):
        raise Exception("Please use the same directory that stores res pickle files by infer_pa")
    if not os.path.exists(os.path.join(output_dir, "pkl_input")):
        raise Exception("Please use the same directory that stores res pickle files by prepare_input")
    

    model_in_files = [i for i in os.listdir(os.path.join(output_dir, "pkl_input")) if ".input.pkl" in i]
    ## only consider output files that are respective to an input file
    model_out_files = [i for i in os.listdir(os.path.join(output_dir, "pkl_output")) 
                       if (".res.pkl" in i) and (i[:-8]+".input.pkl" in model_in_files)]
    if len(model_in_files) != len(model_out_files):
        raise Exception("Number of *.res.pkl is different from number of *.input.pkl. Please make sure that all input files are successfully used for infering PAS.")
    if utr_merge:
        outfile = os.path.join(output_dir, "res.gene.pkl")
    else:
        outfile = os.path.join(output_dir, "res.utr.pkl")
        
    out_dict = {} #key:gene or gene_utr, value: {gene_utr:res}
    for model_out_file in model_out_files:
        outp = os.path.join(output_dir, "pkl_output", model_out_file)
        with open(outp, 'rb') as out_fh:
            while True:
                try:
                    para = pickle.load(out_fh)
                    gene_info = para.gene_info_str.split(sep=":")
                    
                    ## if merge all utr_file as one, then have gene_ID as key
                    if utr_merge:
                        out_key = gene_info[1]
                    ## treat each utr_file separately, then have gene_utr as key
                    else:
                        out_key = ":".join(para.gene_info_str.split(sep=":")[1:3])
                        
                    if out_key not in out_dict.keys():
                        out_dict[out_key] = {}
                    out_dict[out_key][para.gene_info_str] = para
                except EOFError:
                    break
    print("Done read model output_dir")
    
    in_dict = {} #key:gene or gene_utr, value: {gene_utr:res}
    for model_in_file in model_in_files:
        inp = os.path.join(output_dir, "pkl_input", model_in_file)
        with open(inp, 'rb') as in_fh:
            while True:
                try:
                    gene_info_str, df = pickle.load(in_fh)
                    
                    ## if merge all utr_file as one, then have gene_ID as key
                    if utr_merge:
                        out_key = gene_info_str.split(sep=":")[1]
                    ## treat each utr_file separately, then have geneID_utrID as key
                    else:
                        out_key = ":".join(gene_info_str.split(sep=":")[1:3])
                        
                    if out_key not in in_dict.keys():
                        in_dict[out_key] = {}
                    in_dict[out_key][gene_info_str] = df
                except EOFError:
                    break
    print("Done read model input")

                    
    pa_dict = {}
    pa_info_dict = {}
#     cnt=1
    st=timer()
    try:
        os.remove(outfile)
    except OSError:
        pass
#     junc_lst=[]
#     change_lst=[]
    junction_pct_thres=0.4
    total_read_pct_thres=0.05
    with open(outfile, "wb") as outfile_h:
        ## If utr_merge, process all utrs (respective to all Parameters) for that gene
        ## else, process one utr (respective to one Parameters)
        for gene in list(out_dict.keys()):
            gene_info = list(out_dict[gene].keys())[0]
            if gene_info[-1:] == "+":
                res, junc, change = proc_junction_pos_pa(in_dict[gene], out_dict[gene], gene, junction_pct_thres, total_read_pct_thres)
            else:
                res, junc, change = proc_junction_neg_pa(in_dict[gene], out_dict[gene], gene, junction_pct_thres, total_read_pct_thres)
            pickle.dump(res, outfile_h)
#             cnt+=1
#             junc_lst.append([gene, junc])
#             change_lst.append([gene, change])
    en=timer()
    print(en-st)
#     junc_df = pd.DataFrame(junc_lst, columns=["gene_id", "junc"])
#     junc_df.to_csv(outdir+"/gene_junc_flag.csv", index=False)
#     change_df = pd.DataFrame(change_lst, columns=["gene_id", "change"])
#     change_df.to_csv(outdir+"/gene_change_by_junc_flag.csv", index=False)

    


## ========= Pre-processing =========
## detect junction reads via CIGAR information
def flag_junction_read(read):
    """
    Input: read from BAM file which is read by pysam
    Output: 1 if there exists only 1 junction site in considering read, 0 otherwise
    """
    cigar_tuples = read.cigartuples 
    ## example: [(1,10), (3,100)]: the first 10 bases are insertion, the next 100 bases are intron
    
    cigar_arr = np.array([[cigar, base] for (cigar, base) in cigar_tuples])
    
    ## Number of intronic region with more than 30bp
    num_intron_30bp = np.sum((cigar_arr[:, 0]==3) & (cigar_arr[:, 1]>30))
    
    ## if exists only one intronic region with more than 30bp
    if num_intron_30bp==1:
        ## flag as junction read
        return 1
    else:
        ## flag as normal read
        return 0



## ========= Pre-processing =========
## Derive the 3'-end position of two segments from one junction read on "+" strand
def proc_junction_pos_read(read):
    """
    Input: junction read from BAM file based on flag_junction_read
    Output: absolute end position of two segments of junction read
    """
    cigar_tuples = read.cigartuples
    ## only keeps cigar: 0 (alignment match = sequence match or mismatch)
    ##                   , 2 (deletion)
    ##                   , 3 (intron (skipped region from reference))
    ##                   , 7 (sequence match)
    ##                   , 8 (mismatch)
    cigar_arr = np.array([[cigar, base] 
                          for (cigar, base) in cigar_tuples 
                          if cigar in [0,2,3,7,8]])
    ## number of intronic regions within read
    num_intron = np.sum(cigar_arr[:,0]==3)
    ## stop when no intronic region found
    if num_intron==0:
        return np.nan, np.nan
    ## get ther highest number of base within intronic region
    intron_base_max = np.max(cigar_arr[cigar_arr[:,0]==3, 1])
    ## get the first index where cigar==3 and its number of base is the highest value
    intron_base_max_idx = np.where((cigar_arr[:,0]==3) 
                                   & (cigar_arr[:,1]==intron_base_max))[0][0]
    
    ref_st = read.reference_start
    ref_en = read.reference_end
    ## segment 1 is on the left of junction site. Its end is equal to summation of reference start and alignment/deletion bases
    seg1_base = np.sum(cigar_arr[:intron_base_max_idx, 1])
    seg1_en = ref_st + seg1_base
    ## segment 2 is on the right of junction site. Its end is equal to reference end of read
    seg2_en = ref_en
    return seg1_en, seg2_en
    
    


## ========= Pre-processing =========
## Derive the 3'-end position of two segments from one junction read on "-" strand
def proc_junction_neg_read(read):
    """
    Input: junction read from BAM file based on flag_junction_read
    Output: absolute end position of two segments of junction read
    """
    cigar_tuples = read.cigartuples
    ## only keeps cigar: 0 (alignment match = sequence match or mismatch)
    ##                   , 2 (deletion)
    ##                   , 3 (intron (skipped region from reference))
    ##                   , 7 (sequence match)
    ##                   , 8 (mismatch)
    cigar_arr = np.array([[cigar, base] 
                          for (cigar, base) in cigar_tuples 
                          if cigar in [0,2,3,7,8]])
    ## number of intronic regions within read
    num_intron = np.sum(cigar_arr[:,0]==3)
    ## stop when no intronic region found
    if num_intron==0:
        return np.nan, np.nan
    ## get ther highest number of base within intronic region
    intron_base_max = np.max(cigar_arr[cigar_arr[:,0]==3, 1])
    ## get the first index where cigar==3 and its number of base is the highest value
    intron_base_max_idx = np.where((cigar_arr[:,0]==3) 
                                   & (cigar_arr[:,1]==intron_base_max))[0][0]
    
    ref_st = read.reference_start
    ref_en = read.reference_end
    ## segment 1 is on the left of junction site. Its end is equal to reference start of read
    seg1_en = ref_st
    ## segment 2 is on the right of junction site. Its end is equal to reference end substracted from alignment/deletion bases
    seg2_base = np.sum(cigar_arr[:intron_base_max_idx+1, 1])
    seg2_en = ref_st + seg2_base
    return seg1_en, seg2_en
    
    



## =====================================================================================================================
## =====================================================================================================================


## ========= Post-processing =========
## Merge pa site on "-" strand which is between junction site with the "correct" pa site next to the last end position
def proc_junction_neg_pa(para_dict, res_dict, gene_key, junction_pct_thres, total_read_pct_thres):
    """
    Input: Information of only one gene or one utr_file, merge information from all pa site of it.
        - para_dict: dataframe["cb_id", "read_id", "junction", "seg1_en", "seg2_en"] resulted from apa_input_prepare
            {gene_utr_info : dataframe}
        - res_dict resulted from apa_core
            {gene_utr_info : res}. res is a Parameters obejct
            + alpha_arr = [relative_position_pa1, relative_position_pa2] 
                            in which relative position of pa 1 is always the smallest 
                            or it means that pa 1 is closer to the "start" (w.r.t. strand information) of 5'-end of utr_file region which is "x"
            + "-" strand -> pa position = right of utr_file - alpha + 1
            + "+" strand -> pa position = left of utr_file + alpha
    Output: 
        - class Parameters. NOTED: there will be no information in cb_id_arr/readID_arr/label_arr about reads that are assigned to uniform component
    """
#     junction_pct_thres=0.4
#     total_read_pct_thres=0.05
    
    ## Extract attributes from Parameters object to numpy arrays.
    ## Later, they will be used to create new Parameters object with updated PA information.
    pa_beta_arr, pa_loc_arr, pa_label_arr, pa_cb_arr, pa_read_arr, pa_seg_en1_arr, pa_seg_en2_arr, junc_read_arr, total_read_arr, utr_st_arr, utr_en_arr, K, weight_pa_arr, pa_unique_label_arr, chrom, gene = ex_attr_Parameters_to_arr(para_dict, res_dict, "-")
    
    pa_dict = {}
    
    ## Processing PA sites within the same junction sites
    
    ## Step 1: select pa with more than 40% junction reads
    pa_to_merge_arr, pa_junc_pct, junc = ex_pa_w_high_junc(junc_read_arr
                                                           , total_read_arr
                                                           , junction_pct_thres
                                                           , pa_unique_label_arr)
    ## debug
    close_to_en1 = 0 ## debug
    close_to_en2 = 0 ## debug
    close_no = 0 ## debug
    print(pa_to_merge_arr)
    
    ## Step 2: merging pa list
    ## Strand "-", considering the closest PA site on the left
    while len(pa_to_merge_arr)>0:
        ori_len = len(pa_to_merge_arr)
        first_pa = pa_to_merge_arr[0]
        if pa_junc_pct[first_pa] <= junction_pct_thres:
            ## exclude pa with junction site from to-be-merged list if its %junction read is lower than threshold
            pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
            continue
        median_en1 = np.median(pa_seg_en1_arr[pa_seg_en1_arr[:,1]==first_pa, 0])
        median_en2 = np.median(pa_seg_en2_arr[pa_seg_en2_arr[:,1]==first_pa, 0])
        ## if two ends are closest to the same pa -> exclude that pa
        if find_closest_pa(median_en1, pa_loc_arr, pa_unique_label_arr, "-") == find_closest_pa(median_en2, pa_loc_arr, pa_unique_label_arr, "-"):
            pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
            continue
        ## if first pa in to-be-merged list is before its end of segment 1
        if first_pa == find_closest_pa(median_en1, pa_loc_arr, pa_unique_label_arr, "-"): ## strand "-"
            close_to_en1 +=1 ## debug
            ## select pa that is the closet to 
            pa = find_closest_pa(median_en2, pa_loc_arr, pa_unique_label_arr, "-") ## strand "-"
            ## if pa=None, then continue
            if pa==None:
                ## exclude pa with no other close pa
                pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
                continue
            if weight_pa_arr[first_pa] > weight_pa_arr[pa]:
                ## merge pa into first_pa
                pa_dict, pa_keep, pa_replace = update_pa_dict(first_pa, pa, pa_dict)
                ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
                pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct, junc_read_arr, total_read_arr, pa_keep, pa_replace)
            else:
                ## merge first_pa into pa
                pa_dict, pa_keep, pa_replace = update_pa_dict(pa, first_pa, pa_dict)
                ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
                pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct, junc_read_arr, total_read_arr, pa_keep, pa_replace)
                ## exclude first_pa from to-be-merged array
                pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
                
        elif first_pa == find_closest_pa(median_en2, pa_loc_arr, pa_unique_label_arr, "-"): ## strand "-"
            close_to_en2 +=1 ## debug
            pa = find_closest_pa(median_en1, pa_loc_arr, pa_unique_label_arr, "-") ## strand "-"
            ## if pa=None, then continue
            if pa==None:
                ## exclude pa with no other close pa
                pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
                continue
            if weight_pa_arr[first_pa] > weight_pa_arr[pa]:
                ## merge pa into first_pa
                pa_dict, pa_keep, pa_replace = update_pa_dict(first_pa, pa, pa_dict)
                ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
                pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct
                                                                               , junc_read_arr
                                                                               , total_read_arr
                                                                               , pa_keep
                                                                               , pa_replace)
            else:
                ## merge first_pa into pa
                pa_dict, pa_keep, pa_replace = update_pa_dict(pa, first_pa, pa_dict)
                ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
                pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct
                                                                               , junc_read_arr
                                                                               , total_read_arr
                                                                               , pa_keep
                                                                               , pa_replace)
                pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
        else:
            close_no +=1 ## debug
            pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
        
        if len(pa_to_merge_arr) == ori_len:
            pa_to_merge_arr = pa_to_merge_arr[1:]

    ## generate one new Parameters object with updated PA information on strand "-"
    para, change = gen_updated_Parameters(pa_unique_label_arr, pa_dict, total_read_arr
                                           , total_read_pct_thres, pa_beta_arr, pa_loc_arr
                                           , pa_label_arr, pa_cb_arr, pa_read_arr, utr_st_arr
                                           , utr_en_arr, chrom, gene, gene_key, "-")
    return para, junc, change



## ========= Post-processing =========
## Merge pa site on "+" strand which is between junction site with the "correct" pa site next to the last end position
def proc_junction_pos_pa(para_dict, res_dict, gene_key, junction_pct_thres, total_read_pct_thres):
    """
    Input: Information of only one gene or one utr_file, merge information from all pa site of it.
        - para_dict: dataframe["cb_id", "read_id", "junction", "seg1_en", "seg2_en"] resulted from apa_input_prepare
            {gene_utr_info : dataframe}
        - res_dict resulted from apa_core
            {gene_utr_info : res}. res is a Parameters obejct
            + alpha_arr = [relative_position_pa1, relative_position_pa2] 
                            in which relative position of pa 1 is always the smallest 
                            or it means that pa 1 is closer to the "start" (w.r.t. strand information) of 5'-end of utr_file region which is "x"
            + "-" strand -> pa position = right of utr_file - alpha + 1
            + "+" strand -> pa position = left of utr_file + alpha
    Output: 
        - class Parameters. NOTED: there will be no information in cb_id_arr/readID_arr/label_arr about reads that are assigned to uniform component
    """
#     junction_pct_thres=0.4
#     total_read_pct_thres=0.05
    
    ## Extract attributes from Parameters object to numpy arrays.
    ## Later, they will be used to create new Parameters object with updated PA information.
    pa_beta_arr, pa_loc_arr, pa_label_arr, pa_cb_arr, pa_read_arr, pa_seg_en1_arr, pa_seg_en2_arr, junc_read_arr, total_read_arr, utr_st_arr, utr_en_arr, K, weight_pa_arr, pa_unique_label_arr, chrom, gene = ex_attr_Parameters_to_arr(para_dict, res_dict, "+")
    
    pa_dict = {}
    
    ## Processing PA sites within the same junction sites
    
    ## Step 1: select pa with more than 40% junction reads
    pa_to_merge_arr, pa_junc_pct, junc = ex_pa_w_high_junc(junc_read_arr
                                                           , total_read_arr
                                                           , junction_pct_thres
                                                           , pa_unique_label_arr)
    ## debug
    close_to_en1 = 0 ## debug
    close_to_en2 = 0 ## debug
    close_no = 0 ## debug
    print(pa_to_merge_arr)
    
    ## Step 2: merging pa list
    ## Strand "+", considering the closest PA site on the left
    while len(pa_to_merge_arr)>0:
        ori_len = len(pa_to_merge_arr)
        first_pa = pa_to_merge_arr[0]
        if pa_junc_pct[first_pa] <= junction_pct_thres:
            ## exclude pa with junction site from to-be-merged list if its %junction read is lower than threshold
            pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
            continue
        median_en1 = np.median(pa_seg_en1_arr[pa_seg_en1_arr[:,1]==first_pa, 0])
        median_en2 = np.median(pa_seg_en2_arr[pa_seg_en2_arr[:,1]==first_pa, 0])
        ## if two ends are closest to the same pa -> exclude that pa
        if find_closest_pa(median_en1, pa_loc_arr, pa_unique_label_arr, "+") == find_closest_pa(median_en2, pa_loc_arr, pa_unique_label_arr, "+"):
            pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
            continue
        ## if first pa in to-be-merged list is before its end of segment 2
        if first_pa == find_closest_pa(median_en2, pa_loc_arr, pa_unique_label_arr, "+"): ## strand "+"
            close_to_en2 +=1 ## debug
            ## select pa that is the closet to 
            pa = find_closest_pa(median_en1, pa_loc_arr, pa_unique_label_arr, "+") ## strand "+"
            ## if pa=None, then continue
            if pa==None:
                ## exclude pa with no other close pa
                pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
                continue
            if weight_pa_arr[first_pa] > weight_pa_arr[pa]:
                ## merge pa into first_pa
                pa_dict, pa_keep, pa_replace = update_pa_dict(first_pa, pa, pa_dict)
                ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
                pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct, junc_read_arr, total_read_arr, pa_keep, pa_replace)
                pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
#                 ori_len += 1 ## keep first pa for this round
            else:
                ## merge first_pa into pa
                pa_dict, pa_keep, pa_replace = update_pa_dict(pa, first_pa, pa_dict)
                ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
                pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct, junc_read_arr, total_read_arr, pa_keep, pa_replace)
                ## exclude first_pa from to-be-merged array
                pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
                
        elif first_pa == find_closest_pa(median_en1, pa_loc_arr, pa_unique_label_arr, "+"): ## strand "+"
            close_to_en1 +=1 ## debug
            pa = find_closest_pa(median_en2, pa_loc_arr, pa_unique_label_arr, "+") ## strand "+"
            ## if pa=None, then continue
            if pa==None:
                ## exclude pa with no other close pa
                pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
                continue
            if weight_pa_arr[first_pa] > weight_pa_arr[pa]:
                ## merge pa into first_pa
                pa_dict, pa_keep, pa_replace = update_pa_dict(first_pa, pa, pa_dict)
                ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
                pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct, junc_read_arr, total_read_arr, pa_keep, pa_replace)
                pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
            else:
                ## merge first_pa into pa
                pa_dict, pa_keep, pa_replace = update_pa_dict(pa, first_pa, pa_dict)
                ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
                pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct, junc_read_arr, total_read_arr, pa_keep, pa_replace)
                pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
        else:
            close_no +=1 ## debug
            pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
        
        if len(pa_to_merge_arr) == ori_len:
            pa_to_merge_arr = pa_to_merge_arr[1:]

    ## generate one new Parameters object with updated PA information on strand "+"
    para, change = gen_updated_Parameters(pa_unique_label_arr, pa_dict, total_read_arr
                                           , total_read_pct_thres, pa_beta_arr, pa_loc_arr
                                           , pa_label_arr, pa_cb_arr, pa_read_arr, utr_st_arr
                                           , utr_en_arr, chrom, gene, gene_key, "+")
    return para, junc, change


    

## ======================================================
## Common function for post processing junction site

## extract values from all Parameters objects to one numpy array per attribute
def ex_attr_Parameters_to_arr(para_dict, res_dict, strand):
    ## component of class Parameters
    pa_beta_arr = np.array([], dtype=int)
    pa_loc_arr = np.array([], dtype=int)
    pa_label_arr = np.array([], dtype=int)
    pa_cb_arr = np.array([], dtype=int)
    pa_read_arr = np.array([], dtype=int)
    pa_seg_en1_arr = np.empty((0,2), int)
    pa_seg_en2_arr = np.empty((0,2), int)
    
    junc_read_arr = np.array([], dtype=int)
    total_read_arr = np.array([], dtype=int)
    utr_st_arr = np.array([], dtype=int)
    utr_en_arr = np.array([], dtype=int)
    K=0
    ## having for loop to concatenate information from all utr_file into one array per output parameter
    for utr in np.sort([u for u in res_dict.keys()]):
        res = res_dict[utr]
        res_info = res.gene_info_str.split(sep=":") ## 4:ENSG00000174780:2:56475376-56477472:+
        chrom = res_info[0]
        gene = res_info[1]
        strand = res_info[4]
        utr_st = int(res_info[3].split(sep="-")[0])
        utr_en = int(res_info[3].split(sep="-")[1])
        ## array of utr_file start
        utr_st_arr = np.append(utr_st_arr, [utr_st])
        ## array of utr_file end
        utr_en_arr = np.append(utr_en_arr, [utr_en])
        ## array of beta, in order of pa id from pa_label_arr
        pa_beta_arr = np.append(pa_beta_arr, res.beta_arr)
        ## array of absolute location of pa, in order of pa id from pa_label_arr
        ## pa_loc_arr depends on strand information
        if strand == "+":
            ## "+" strand -> pa position = left of utr_file + alpha
            pa_loc_arr = np.append(pa_loc_arr, (utr_st + res.alpha_arr))
        else:
            ## "-" strand -> pa position = right of utr_file - alpha + 1
            pa_loc_arr = np.append(pa_loc_arr, (utr_en - res.alpha_arr + 1))
            
        ## array of pa label, each value is respective to one read. 
        ## ex: utr1=[pa0,pa0,pa1], utr2=[pa0, pa1] -> append utr2 to utr1 : [pa0(utr1), pa0(utr1), pa1(utr1), pa2(utr2), pa3(utr2)]
        pa_label_arr = np.append(pa_label_arr, (K+res.label_arr[res.label_arr < int(res.K)]))
        ## array of CB_ID, resepctive to order of pa_label_arr
        pa_cb_arr = np.append(pa_cb_arr, (res.cb_id_arr[res.label_arr < int(res.K)]))
        ## array of read_ID, resepctive to order of pa_label_arr
        pa_read_arr = np.append(pa_read_arr, (res.readID_arr[res.label_arr < int(res.K)]))
        ## array of total read count per pa, in order of pa id from pa_label_arr
        label, counts = np.unique(res.label_arr[res.label_arr < int(res.K)]
                                  , return_counts=True)
        pa_read_count=np.zeros(res.K)
        np.put(pa_read_count, label, counts)
        pa_read_count[pa_read_count==0]=1
        total_read_arr = np.append(total_read_arr, pa_read_count)
        
        input_df = para_dict[utr].set_index("read_id")
        ## only keep reads respective to calculated pa
        input_df = input_df.loc[res.readID_arr[res.label_arr < int(res.K)],] 
        ## assign pa id to each read from input dataframe
        input_df.loc[res.readID_arr[res.label_arr < int(res.K)], "label"] = K+res.label_arr[res.label_arr < int(res.K)] 
        ## arrar of position of seg1 and seg2, each value is respective to one read.
        pa_seg_en1_arr = np.append(pa_seg_en1_arr
                                   , input_df.loc[input_df["junction"]==1, ["seg1_en", "label"]].to_numpy()
                                   , axis=0
                                  ) ## 2D arrray where column1=seg1_en
        pa_seg_en2_arr = np.append(pa_seg_en2_arr
                                   , input_df.loc[input_df["junction"]==1, ["seg2_en", "label"]].to_numpy()
                                   , axis=0
                                  ) ## 2D arrray where column1=seg2_en
        ## array of junction read count per pa, in order of pa id from pa_label_arr
        junc_arr = input_df.loc[res.readID_arr[res.label_arr < int(res.K)], "junction"].to_numpy()
        label, counts = np.unique((res.label_arr[res.label_arr < int(res.K)])[junc_arr==1]
                                  , return_counts=True)
        junc_read=np.zeros(res.K)
        np.put(junc_read, label, counts)
        junc_read_arr = np.append(junc_read_arr, junc_read)
        ## factor to re-label pa_id
        K = K + int(res.K)
        
    
    ## Sort everything by pa_loc. The smallest pa location has pa id = 0
    ## pa_unique_label_arr depends on strand information
    if strand == "+":
        ## pa id that are sorted by location in ascending order. strand "+"
        pa_unique_label_arr = np.argsort(pa_loc_arr) 
    else:
        ## pa id that are sorted by location in decending order. strand "-"
        pa_unique_label_arr = np.argsort(-pa_loc_arr) 
    
    pa_beta_arr = pa_beta_arr[pa_unique_label_arr]
    pa_loc_arr = pa_loc_arr[pa_unique_label_arr]
    junc_read_arr = junc_read_arr[pa_unique_label_arr]
    total_read_arr = total_read_arr[pa_unique_label_arr]
    tmp_dict = {pa_unique_label_arr[new]:new for new in range(len(pa_unique_label_arr))}
    pa_label_arr = pd.Series(pa_label_arr).replace(tmp_dict).to_numpy()
    pa_seg_en1_arr[:,1] = pd.Series(pa_seg_en1_arr[:,1]).replace(tmp_dict).to_numpy()
    pa_seg_en2_arr[:,1] = pd.Series(pa_seg_en2_arr[:,1]).replace(tmp_dict).to_numpy()
    pa_unique_label_arr = np.array([i for i in range(len(pa_loc_arr))])
    
    weight_pa_arr = total_read_arr
    
    return pa_beta_arr, pa_loc_arr, pa_label_arr, pa_cb_arr, pa_read_arr, pa_seg_en1_arr, pa_seg_en2_arr, junc_read_arr, total_read_arr, utr_st_arr, utr_en_arr, K, weight_pa_arr, pa_unique_label_arr, chrom, gene


## select pa with more than 40% junction reads
def ex_pa_w_high_junc(junc_read_arr, total_read_arr, junction_pct_thres, pa_unique_label_arr):
    pa_junc_pct = junc_read_arr / total_read_arr ## sorted in ascending order by pa position
    pa_to_merge_arr = pa_unique_label_arr[pa_junc_pct > junction_pct_thres] ## sorted in ascending order by pa position
    ## debug
    if len(pa_to_merge_arr) > 0:
        junc=1
    else:
        junc=0
    return pa_to_merge_arr, pa_junc_pct, junc


## find_closest_pa depends on strand information
def find_closest_pa(en_position, pa_position_arr, pa_arr, strand): 
    """
    - pa_position_arr : the absolute position of pa. List can be un-ordered
    - pa_arr : list of pa id which are ordered by its absolute position. Ex: pa id = 0 means it is the first pa at the 5'-end
    """
    ## closet position is counted from "en_position" toward 3'-end
    ## return ID of the closet pa
    if (strand=="+") & (len(pa_arr[np.sort(pa_position_arr) >= en_position]) >0):
        return pa_arr[np.sort(pa_position_arr) >= en_position][0]
    elif (strand=="-") & (len(pa_arr[(-np.sort(-pa_position_arr)) <= en_position]) >0):
        return pa_arr[(-np.sort(-pa_position_arr)) <= en_position][0]
    else:
        ## there is no closest pa to the 3'-end, return None
        return None

## iteratively merge pa_replace to pa_keep
def update_pa_dict(pa_keep, pa_replace, pa_dict):
    ## if pa_replace is already replaced by another pa 
    while np.sum(np.array([pa for pa in pa_dict.keys()]) == pa_replace)>0:
        pa_replace=pa_dict[pa_replace]
    if pa_replace==pa_keep:
        return pa_dict, pa_keep, pa_replace
    ## merge pa_replace into pa_keep
    pa_dict[pa_replace] = pa_keep
    return pa_dict, pa_keep, pa_replace


## re-calculate total reads per PA, proportion of junction reads per PA
def recal_pa_junc_pct(pa_junc_pct, junc_arr, total_arr, pa_keep, pa_replace):
    ## set replaced pa to 0 for percentage of junction read
    ## recalculate total read and total junction read for pa_keep. Set both for pa_replace to zero
    if pa_keep == pa_replace:
        return pa_junc_pct, junc_arr, total_arr
    total_arr[pa_keep] = np.sum(total_arr[[pa_keep, pa_replace]])
    junc_arr[pa_keep] = np.sum(junc_arr[[pa_keep, pa_replace]])
    total_arr[pa_replace] = 0
    junc_arr[pa_replace] = 0
    pa_junc_pct[pa_keep] = junc_arr[pa_keep] / total_arr[pa_keep]
    pa_junc_pct[pa_replace] = 0
    return pa_junc_pct, junc_arr, total_arr


## generate the updated Parameter object
def gen_updated_Parameters(pa_unique_label_arr, pa_dict, total_read_arr
                           , total_read_pct_thres, pa_beta_arr, pa_loc_arr
                           , pa_label_arr, pa_cb_arr, pa_read_arr, utr_st_arr
                           , utr_en_arr, chrom, gene, gene_key, strand):
    ## re-label pa id: merge pa with junction reads. Ex: [0,1,2,3,4] ->[1,2,3,4]
    while np.sum(np.isin(pa_unique_label_arr, [pa for pa in pa_dict.keys()])) > 0:
        pa_unique_label_arr = pd.Series(pa_unique_label_arr).replace(pa_dict).to_numpy()
        pa_label_arr = pd.Series(pa_label_arr).replace(pa_dict).to_numpy()
    
    
    ## exclude pa with pct lower than 5%. Ex: [1,2,3,4] -> [1,3,4]
    pa_remain_arr = np.array(list(set(pa_unique_label_arr)))
    pa_total_pct = total_read_arr[pa_remain_arr]/np.sum(total_read_arr[pa_remain_arr])
    pa_remove_arr = pa_remain_arr[pa_total_pct <= total_read_pct_thres] ## list pa with low percentage
    pa_remain_arr = pa_remain_arr[pa_total_pct > total_read_pct_thres] ## list pa with high percentage
    uni_comp = len(pa_remain_arr)
    
    ## attributes of final class Parameters
    pa_beta_arr = pa_beta_arr[pa_remain_arr]
    pa_loc_arr = pa_loc_arr[pa_remain_arr]
    ## re-label pa id so that they are consecutive numbers and starting from 0. Ex: [1,2,3,4] -> [0,1,2,3]
    tmp_dict = {pa_remain_arr[new]:new for new in range(len(pa_remain_arr))}
    for pa in pa_remove_arr:
        tmp_dict[pa] = uni_comp
    pa_label_arr = pd.Series(pa_label_arr).replace(tmp_dict).to_numpy()
    ws_arr = total_read_arr[pa_remain_arr] / np.sum(total_read_arr[pa_remain_arr])
    utr_st = np.min(utr_st_arr)
    utr_en = np.max(utr_en_arr)
    
    para = Parameters(title="Final Result"
                      , alpha_arr=(utr_en - pa_loc_arr + 1) if strand == "-" else (pa_loc_arr-utr_st)
                      , beta_arr=pa_beta_arr
                      , ws=ws_arr
                      , K=len(pa_remain_arr)
                      , L=0
                      , cb_id_arr=pa_cb_arr[pa_label_arr < uni_comp] ## only keep reads with respective to any pa site
                      , readID_arr=pa_read_arr[pa_label_arr < uni_comp] ## only keep reads with respective to any pa site
                     )
    para.label_arr=pa_label_arr[pa_label_arr < uni_comp] ## only keep reads with respective to any pa site
    if len(gene_key.split(sep=":"))==1:
        para.gene_info_str=str(chrom)+":"+gene+":"+str(1)+":"+str(utr_st)+"-"+str(utr_en)+":"+strand ## 4:ENSG00000174780:1:56475376-56477472:+
    else:
        para.gene_info_str=str(chrom)+":"+gene_key+":"+str(utr_st)+"-"+str(utr_en)+":"+strand ## 4:ENSG00000174780:2:56475376-56477472:+
    tmp, counts=np.unique(para.label_arr, return_counts=True)
    #     return para
    ## debug
    if len(pa_dict)>0:
        print(gene_key, pa_dict)
    if len(pa_dict)>0:
        change=1
    else:
        change=0
    return para, change








































# ### OLD

# ## ========= Post-processing =========
# ## Merge pa site on "+" strand which is between junction site with the "correct" pa site next to the last end position
# def proc_junction_pos_pa(para_dict, res_dict, gene_key, junction_pct_thres, total_read_pct_thres):
#     """
#     Input: Information of only one gene or one utr_file, merge information from all pa site of it.
#         - para_dict: dataframe["cb_id", "read_id", "junction", "seg1_en", "seg2_en"] resulted from apa_input_prepare
#             {gene_utr_info : dataframe}
#         - res_dict resulted from apa_core
#             {gene_utr_info : res}. res is a Parameters obejct
#             + alpha_arr = [relative_position_pa1, relative_position_pa2] 
#                             in which relative position of pa 1 is always the smallest 
#                             or it means that pa 1 is closer to the "start" (w.r.t. strand information) of 5'-end of utr_file region which is "x"
#             + "-" strand -> pa position = right of utr_file - alpha + 1
#             + "+" strand -> pa position = left of utr_file + alpha
#     Output: 
#         - class Parameters. NOTED: there will be no information in cb_id_arr/readID_arr/label_arr about reads that are assigned to uniform component
#     """
#     junction_pct_thres=0.4
#     total_read_pct_thres=0.05
    
#     ## component of class Parameters
#     pa_beta_arr = np.array([], dtype=int)
#     pa_loc_arr = np.array([], dtype=int)
#     pa_label_arr = np.array([], dtype=int)
#     pa_cb_arr = np.array([], dtype=int)
#     pa_read_arr = np.array([], dtype=int)
#     pa_seg_en1_arr = np.empty((0,2), int)
#     pa_seg_en2_arr = np.empty((0,2), int)
    
#     junc_read_arr = np.array([], dtype=int)
#     total_read_arr = np.array([], dtype=int)
#     utr_st_arr = np.array([], dtype=int)
#     utr_en_arr = np.array([], dtype=int)
#     K=0
#     ## having for loop to concatenate information from all utr_file into one array per output parameter
#     for utr in np.sort([u for u in res_dict.keys()]):
#         res = res_dict[utr]
# #         print(res) ## debug
#         res_info = res.gene_info_str.split(sep=":") ## 4:ENSG00000174780:2:56475376-56477472:+
#         chrom = res_info[0]
#         gene = res_info[1]
#         strand = res_info[4]
#         utr_st = int(res_info[3].split(sep="-")[0])
#         utr_en = int(res_info[3].split(sep="-")[1])
#         ## array of utr_file start
#         utr_st_arr = np.append(utr_st_arr, [utr_st])
#         ## array of utr_file end
#         utr_en_arr = np.append(utr_en_arr, [utr_en])
#         ## array of beta, in order of pa id from pa_label_arr
#         pa_beta_arr = np.append(pa_beta_arr, res.beta_arr)
#         ## array of absolute location of pa, in order of pa id from pa_label_arr
#         ## strand "+"
#         pa_loc_arr = np.append(pa_loc_arr, (utr_st + res.alpha_arr)) ## strand "+"
#         ## array of pa label, each value is respective to one read. 
#         ## ex: utr1=[pa0,pa0,pa1], utr2=[pa0, pa1] -> append utr2 to utr1 : [pa0(utr1), pa0(utr1), pa1(utr1), pa2(utr2), pa3(utr2)]
#         pa_label_arr = np.append(pa_label_arr, (K+res.label_arr[res.label_arr < int(res.K)]))
#         ## array of CB_ID, resepctive to order of pa_label_arr
#         pa_cb_arr = np.append(pa_cb_arr, (res.cb_id_arr[res.label_arr < int(res.K)]))
#         ## array of read_ID, resepctive to order of pa_label_arr
#         pa_read_arr = np.append(pa_read_arr, (res.readID_arr[res.label_arr < int(res.K)]))
#         ## array of total read count per pa, in order of pa id from pa_label_arr
#         label, counts = np.unique(res.label_arr[res.label_arr < int(res.K)]
#                                   , return_counts=True)
#         pa_read_count=np.zeros(res.K)
#         np.put(pa_read_count, label, counts)
#         pa_read_count[pa_read_count==0]=1
#         total_read_arr = np.append(total_read_arr, pa_read_count)

        
#         input_df = para_dict[utr].set_index("read_id")
#         input_df = input_df.loc[res.readID_arr[res.label_arr < int(res.K)],] ## only keep reads respective to calculated pa
#         input_df.loc[res.readID_arr[res.label_arr < int(res.K)], "label"] = K+res.label_arr[res.label_arr < int(res.K)] ## assign pa id to each read from input dataframe
#         ## arrar of position of seg1 and seg2, each value is respective to one read.
#         pa_seg_en1_arr = np.append(pa_seg_en1_arr
#                                    , input_df.loc[input_df["junction"]==1, ["seg1_en", "label"]].to_numpy()
#                                    , axis=0
#                                   ) ## 2D arrray where column1=seg1_en
#         pa_seg_en2_arr = np.append(pa_seg_en2_arr
#                                    , input_df.loc[input_df["junction"]==1, ["seg2_en", "label"]].to_numpy()
#                                    , axis=0
#                                   ) ## 2D arrray where column1=seg2_en
#         ## array of junction read count per pa, in order of pa id from pa_label_arr
#         junc_arr = input_df.loc[res.readID_arr[res.label_arr < int(res.K)], "junction"].to_numpy()
#         label, counts = np.unique((res.label_arr[res.label_arr < int(res.K)])[junc_arr==1]
#                                   , return_counts=True)
#         junc_read=np.zeros(res.K)
#         np.put(junc_read, label, counts)
#         junc_read_arr = np.append(junc_read_arr, junc_read)
#         ## factor to re-label pa_id
#         K = K + int(res.K)
        
# #     print("before: K="+str(K)) ## debug
#     ## Sort everything by pa_loc. The smallest pa location has pa id = 0
#     pa_unique_label_arr = np.argsort(pa_loc_arr) ## pa id that are sorted by location in ascending order. strand "+"
#     pa_beta_arr = pa_beta_arr[pa_unique_label_arr]
#     pa_loc_arr = pa_loc_arr[pa_unique_label_arr]
#     junc_read_arr = junc_read_arr[pa_unique_label_arr]
#     total_read_arr = total_read_arr[pa_unique_label_arr]
#     tmp_dict = {pa_unique_label_arr[new]:new for new in range(len(pa_unique_label_arr))}
#     pa_label_arr = pd.Series(pa_label_arr).replace(tmp_dict).to_numpy()
#     pa_seg_en1_arr[:,1] = pd.Series(pa_seg_en1_arr[:,1]).replace(tmp_dict).to_numpy()
#     pa_seg_en2_arr[:,1] = pd.Series(pa_seg_en2_arr[:,1]).replace(tmp_dict).to_numpy()
#     pa_unique_label_arr = np.array([i for i in range(len(pa_loc_arr))]) 
    
    
#     weight_pa_arr = total_read_arr
#     pa_dict = {}
    
#     def find_closest_pa(en_position, pa_position_arr, pa_arr): ## strand "+"
#         """
#         - pa_position_arr : the absolute position of pa. List can be un-ordered
#         - pa_list : list of pa id which are ordered by its absolute position. Ex: pa id = 0 means it is the first pa at the 5'-end
#         """
#         ## closet position is counted from "en_position" toward 3'-end
#         ## return ID of the closet pa
#         if len(pa_arr[np.sort(pa_position_arr) >= en_position]) >0:
#             return pa_arr[np.sort(pa_position_arr) >= en_position][0]
#         else:
#             ## there is no closest pa to the 3'-end, return None
#             return None
    
#     def update_pa_dict(pa_keep, pa_replace, pa_dict):
#         ## if pa_replace is already replaced by another pa 
#         while np.sum(np.array([pa for pa in pa_dict.keys()]) == pa_replace)>0:
#             pa_replace=pa_dict[pa_replace]
#         if pa_replace==pa_keep:
#             return pa_dict, pa_keep, pa_replace
#         ## merge pa_replace into pa_keep
#         pa_dict[pa_replace] = pa_keep
#         return pa_dict, pa_keep, pa_replace
    
#     def recal_pa_junc_pct(pa_junc_pct, junc_arr, total_arr, pa_keep, pa_replace):
#         ## set replaced pa to 0 for percentage of junction read
#         ## recalculate total read and total junction read for pa_keep. Set both for pa_replace to zero
#         if pa_keep == pa_replace:
#             return pa_junc_pct, junc_arr, total_arr
#         total_arr[pa_keep] = np.sum(total_arr[[pa_keep, pa_replace]])
#         junc_arr[pa_keep] = np.sum(junc_arr[[pa_keep, pa_replace]])
#         total_arr[pa_replace] = 0
#         junc_arr[pa_replace] = 0
#         pa_junc_pct[pa_keep] = junc_arr[pa_keep] / total_arr[pa_keep]
#         pa_junc_pct[pa_replace] = 0
#         return pa_junc_pct, junc_arr, total_arr
    
#     ## Step 1: select pa with more than 40% junction reads
#     pa_junc_pct = junc_read_arr / total_read_arr ## sorted in ascending order by pa position
#     pa_to_merge_arr = pa_unique_label_arr[pa_junc_pct > junction_pct_thres] ## sorted in ascending order by pa position
#     ## debug
#     if len(pa_to_merge_arr) > 0:
#         junc=1
#     else:
#         junc=0
#     ## debug
#     close_to_en1 = 0 ## debug
#     close_to_en2 = 0 ## debug
#     close_no = 0 ## debug
    
#     ## Step 2: merging pa list
#     while len(pa_to_merge_arr)>0:
#         ori_len = len(pa_to_merge_arr)
#         first_pa = pa_to_merge_arr[0]
#         if pa_junc_pct[first_pa] <= junction_pct_thres:
#             ## exclude pa with junction site from to-be-merged list if its %junction read is lower than threshold
#             pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
#             continue
#         median_en1 = np.median(pa_seg_en1_arr[pa_seg_en1_arr[:,1]==first_pa, 0])
#         median_en2 = np.median(pa_seg_en2_arr[pa_seg_en2_arr[:,1]==first_pa, 0])
#         ## if two ends are closest to the same pa -> exclude that pa
#         if find_closest_pa(median_en1, pa_loc_arr, pa_unique_label_arr) == find_closest_pa(median_en2, pa_loc_arr, pa_unique_label_arr):
#             pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
#             continue
#         ## if first pa in to-be-merged list is before its end of segment 2
#         if first_pa == find_closest_pa(median_en2, pa_loc_arr, pa_unique_label_arr): ## strand "+"
#             close_to_en2 +=1 ## debug
#             ## select pa that is the closet to 
#             pa = find_closest_pa(median_en1, pa_loc_arr, pa_unique_label_arr) ## strand "+"
#             ## if pa=None, then continue
#             if pa==None:
#                 ## exclude pa with no other close pa
#                 pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
#                 continue
#             if weight_pa_arr[first_pa] > weight_pa_arr[pa]:
#                 ## merge pa into first_pa
#                 pa_dict, pa_keep, pa_replace = update_pa_dict(first_pa, pa, pa_dict)
#                 ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
#                 pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct, junc_read_arr, total_read_arr, pa_keep, pa_replace)
#                 pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
# #                 ori_len += 1 ## keep first pa for this round
#             else:
#                 ## merge first_pa into pa
#                 pa_dict, pa_keep, pa_replace = update_pa_dict(pa, first_pa, pa_dict)
#                 ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
#                 pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct, junc_read_arr, total_read_arr, pa_keep, pa_replace)
#                 ## exclude first_pa from to-be-merged array
#                 pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
                
#         elif first_pa == find_closest_pa(median_en1, pa_loc_arr, pa_unique_label_arr): ## strand "+"
#             close_to_en1 +=1 ## debug
#             pa = find_closest_pa(median_en2, pa_loc_arr, pa_unique_label_arr) ## strand "+"
#             ## if pa=None, then continue
#             if pa==None:
#                 ## exclude pa with no other close pa
#                 pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
#                 continue
#             if weight_pa_arr[first_pa] > weight_pa_arr[pa]:
#                 ## merge pa into first_pa
#                 pa_dict, pa_keep, pa_replace = update_pa_dict(first_pa, pa, pa_dict)
#                 ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
#                 pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct, junc_read_arr, total_read_arr, pa_keep, pa_replace)
#                 pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
#             else:
#                 ## merge first_pa into pa
#                 pa_dict, pa_keep, pa_replace = update_pa_dict(pa, first_pa, pa_dict)
#                 ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
#                 pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct, junc_read_arr, total_read_arr, pa_keep, pa_replace)
#                 pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
#         else:
#             close_no +=1 ## debug
#             pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
        
#         if len(pa_to_merge_arr) == ori_len:
#             pa_to_merge_arr = pa_to_merge_arr[1:]
#     ## re-label pa id
#     while np.sum(np.isin(pa_unique_label_arr, [pa for pa in pa_dict.keys()])) > 0:
#         pa_unique_label_arr = pd.Series(pa_unique_label_arr).replace(pa_dict).to_numpy()
#         pa_label_arr = pd.Series(pa_label_arr).replace(pa_dict).to_numpy()
        
    
#     ## exclude pa with pct lower than 5%. Ex: [1,2,3,4] -> [1,3,4]
#     pa_remain_arr = np.array(list(set(pa_unique_label_arr)))
#     pa_total_pct = total_read_arr[pa_remain_arr]/np.sum(total_read_arr[pa_remain_arr])
# #     print(pa_total_pct) ## debug
#     pa_remove_arr = pa_remain_arr[pa_total_pct <= total_read_pct_thres] ## list pa with low percentage
#     pa_remain_arr = pa_remain_arr[pa_total_pct > total_read_pct_thres] ## list pa with high percentage
#     uni_comp = len(pa_remain_arr)
    
#     ## attributes of final class Parameters
#     pa_beta_arr = pa_beta_arr[pa_remain_arr]
#     pa_loc_arr = pa_loc_arr[pa_remain_arr]
#     ## re-label pa id so that they are consecutive numbers and starting from 0. Ex: [1,2,3,4] -> [0,1,2,3]
#     tmp_dict = {pa_remain_arr[new]:new for new in range(len(pa_remain_arr))}
#     for pa in pa_remove_arr:
#         tmp_dict[pa] = uni_comp
#     pa_label_arr = pd.Series(pa_label_arr).replace(tmp_dict).to_numpy()
#     ws_arr = total_read_arr[pa_remain_arr] / np.sum(total_read_arr[pa_remain_arr])
#     utr_st = np.min(utr_st_arr)
#     utr_en = np.max(utr_en_arr)

#     para = Parameters(title="Final Result"
#                       , alpha_arr=pa_loc_arr-utr_st ## strand "+"
#                       , beta_arr=pa_beta_arr
#                       , ws=ws_arr
#                       , K=len(pa_remain_arr)
#                       , L=0
#                       , cb_id_arr=pa_cb_arr[pa_label_arr < uni_comp] ## only keep reads with respective to any pa site
#                       , readID_arr=pa_read_arr[pa_label_arr < uni_comp] ## only keep reads with respective to any pa site
#                      )
#     para.label_arr=pa_label_arr[pa_label_arr < uni_comp] ## only keep reads with respective to any pa site
#     if len(gene_key.split(sep=":"))==1:
#         para.gene_info_str=str(chrom)+":"+gene+":"+str(1)+":"+str(utr_st)+"-"+str(utr_en)+":"+strand ## 4:ENSG00000174780:1:56475376-56477472:+
#     else:
#         para.gene_info_str=str(chrom)+":"+gene_key+":"+str(utr_st)+"-"+str(utr_en)+":"+strand ## 4:ENSG00000174780:2:56475376-56477472:+
#     tmp, counts=np.unique(para.label_arr, return_counts=True)
# #     return para
#     ## debug
#     if len(pa_dict)>0:
#         print(gene_key, pa_dict)
#     if len(pa_dict)>0:
#         change=1
#     else:
#         change=0
#     return para, junc, change
#     ## debug























# ## ========= Post-processing =========
# ## Merge pa site on "-" strand which is between junction site with the "correct" pa site next to the last end position
# def proc_junction_neg_pa(para_dict, res_dict, gene_key, junction_pct_thres, total_read_pct_thres):
#     """
#     Input: Information of only one gene or one utr_file, merge information from all pa site of it.
#         - para_dict: dataframe["cb_id", "read_id", "junction", "seg1_en", "seg2_en"] resulted from apa_input_prepare
#             {gene_utr_info : dataframe}
#         - res_dict resulted from apa_core
#             {gene_utr_info : res}. res is a Parameters obejct
#             + alpha_arr = [relative_position_pa1, relative_position_pa2] 
#                             in which relative position of pa 1 is always the smallest 
#                             or it means that pa 1 is closer to the "start" (w.r.t. strand information) of 5'-end of utr_file region which is "x"
#             + "-" strand -> pa position = right of utr_file - alpha + 1
#             + "+" strand -> pa position = left of utr_file + alpha
#     Output: 
#         - class Parameters. NOTED: there will be no information in cb_id_arr/readID_arr/label_arr about reads that are assigned to uniform component
#     """
#     junction_pct_thres=0.4
#     total_read_pct_thres=0.05
    
#     ## component of class Parameters
#     pa_beta_arr = np.array([], dtype=int)
#     pa_loc_arr = np.array([], dtype=int)
#     pa_label_arr = np.array([], dtype=int)
#     pa_cb_arr = np.array([], dtype=int)
#     pa_read_arr = np.array([], dtype=int)
#     pa_seg_en1_arr = np.empty((0,2), int)
#     pa_seg_en2_arr = np.empty((0,2), int)
    
#     junc_read_arr = np.array([], dtype=int)
#     total_read_arr = np.array([], dtype=int)
#     utr_st_arr = np.array([], dtype=int)
#     utr_en_arr = np.array([], dtype=int)
#     K=0
#     ## having for loop to concatenate information from all utr_file into one array per output parameter
#     for utr in np.sort([u for u in res_dict.keys()]):
#         res = res_dict[utr]
#         res_info = res.gene_info_str.split(sep=":") ## 4:ENSG00000174780:2:56475376-56477472:+
#         chrom = res_info[0]
#         gene = res_info[1]
#         strand = res_info[4]
#         utr_st = int(res_info[3].split(sep="-")[0])
#         utr_en = int(res_info[3].split(sep="-")[1])
#         ## array of utr_file start
#         utr_st_arr = np.append(utr_st_arr, [utr_st])
#         ## array of utr_file end
#         utr_en_arr = np.append(utr_en_arr, [utr_en])
#         ## array of beta, in order of pa id from pa_label_arr
#         pa_beta_arr = np.append(pa_beta_arr, res.beta_arr)
#         ## array of absolute location of pa, in order of pa id from pa_label_arr
#         ## strand "-"
#         pa_loc_arr = np.append(pa_loc_arr, (utr_en - res.alpha_arr + 1))
#         ## array of pa label, each value is respective to one read. 
#         ## ex: utr1=[pa0,pa0,pa1], utr2=[pa0, pa1] -> append utr2 to utr1 : [pa0(utr1), pa0(utr1), pa1(utr1), pa2(utr2), pa3(utr2)]
#         pa_label_arr = np.append(pa_label_arr, (K+res.label_arr[res.label_arr < int(res.K)]))
#         ## array of CB_ID, resepctive to order of pa_label_arr
#         pa_cb_arr = np.append(pa_cb_arr, (res.cb_id_arr[res.label_arr < int(res.K)]))
#         ## array of read_ID, resepctive to order of pa_label_arr
#         pa_read_arr = np.append(pa_read_arr, (res.readID_arr[res.label_arr < int(res.K)]))
#         ## array of total read count per pa, in order of pa id from pa_label_arr
#         label, counts = np.unique(res.label_arr[res.label_arr < int(res.K)]
#                                   , return_counts=True)
#         pa_read_count=np.zeros(res.K)
#         np.put(pa_read_count, label, counts)
#         pa_read_count[pa_read_count==0]=1
#         total_read_arr = np.append(total_read_arr, pa_read_count)
        
#         input_df = para_dict[utr].set_index("read_id")
#         input_df = input_df.loc[res.readID_arr[res.label_arr < int(res.K)],] ## only keep reads respective to calculated pa
#         input_df.loc[res.readID_arr[res.label_arr < int(res.K)], "label"] = K+res.label_arr[res.label_arr < int(res.K)] ## assign pa id to each read from input dataframe
#         ## arrar of position of seg1 and seg2, each value is respective to one read.
#         pa_seg_en1_arr = np.append(pa_seg_en1_arr
#                                    , input_df.loc[input_df["junction"]==1, ["seg1_en", "label"]].to_numpy()
#                                    , axis=0
#                                   ) ## 2D arrray where column1=seg1_en
#         pa_seg_en2_arr = np.append(pa_seg_en2_arr
#                                    , input_df.loc[input_df["junction"]==1, ["seg2_en", "label"]].to_numpy()
#                                    , axis=0
#                                   ) ## 2D arrray where column1=seg2_en
#         ## array of junction read count per pa, in order of pa id from pa_label_arr
#         junc_arr = input_df.loc[res.readID_arr[res.label_arr < int(res.K)], "junction"].to_numpy()
#         label, counts = np.unique((res.label_arr[res.label_arr < int(res.K)])[junc_arr==1]
#                                   , return_counts=True)
#         junc_read=np.zeros(res.K)
#         np.put(junc_read, label, counts)
#         junc_read_arr = np.append(junc_read_arr, junc_read)
#         ## factor to re-label pa_id
#         K = K + int(res.K)
        
    
#     ## Sort everything by pa_loc. The smallest pa location has pa id = 0
#     pa_unique_label_arr = np.argsort(-pa_loc_arr) ## pa id that are sorted by location in decending order. strand "-"
#     pa_beta_arr = pa_beta_arr[pa_unique_label_arr]
#     pa_loc_arr = pa_loc_arr[pa_unique_label_arr]
#     junc_read_arr = junc_read_arr[pa_unique_label_arr]
#     total_read_arr = total_read_arr[pa_unique_label_arr]
#     tmp_dict = {pa_unique_label_arr[new]:new for new in range(len(pa_unique_label_arr))}
#     pa_label_arr = pd.Series(pa_label_arr).replace(tmp_dict).to_numpy()
#     pa_seg_en1_arr[:,1] = pd.Series(pa_seg_en1_arr[:,1]).replace(tmp_dict).to_numpy()
#     pa_seg_en2_arr[:,1] = pd.Series(pa_seg_en2_arr[:,1]).replace(tmp_dict).to_numpy()
#     pa_unique_label_arr = np.array([i for i in range(len(pa_loc_arr))])
    
    
#     weight_pa_arr = total_read_arr
#     pa_dict = {}
    
#     def find_closest_pa(en_position, pa_position_arr, pa_arr): ## strand "-"
#         """
#         - pa_position_arr : the absolute position of pa. List can be un-ordered
#         - pa_arr : list of pa id which are ordered by its absolute position. Ex: pa id = 0 means it is the first pa at the 5'-end
#         """
#         ## closet position is counted from "en_position" toward 3'-end
#         ## return ID of the closet pa
#         if len(pa_arr[(-np.sort(-pa_position_arr)) <= en_position]) >0:
#             return pa_arr[(-np.sort(-pa_position_arr)) <= en_position][0]
#         else:
#             ## there is no closest pa to the 3'-end, return None
#             return None
    
#     def update_pa_dict(pa_keep, pa_replace, pa_dict):
#         ## if pa_replace is already replaced by another pa 
#         while np.sum(np.array([pa for pa in pa_dict.keys()]) == pa_replace)>0:
#             pa_replace=pa_dict[pa_replace]
#         if pa_replace==pa_keep:
#             return pa_dict, pa_keep, pa_replace
#         ## merge pa_replace into pa_keep
#         pa_dict[pa_replace] = pa_keep
#         return pa_dict, pa_keep, pa_replace
    
#     def recal_pa_junc_pct(pa_junc_pct, junc_arr, total_arr, pa_keep, pa_replace):
#         ## set replaced pa to 0 for percentage of junction read
#         ## recalculate total read and total junction read for pa_keep. Set both for pa_replace to zero
#         if pa_keep == pa_replace:
#             return pa_junc_pct, junc_arr, total_arr
#         total_arr[pa_keep] = np.sum(total_arr[[pa_keep, pa_replace]])
#         junc_arr[pa_keep] = np.sum(junc_arr[[pa_keep, pa_replace]])
#         total_arr[pa_replace] = 0
#         junc_arr[pa_replace] = 0
#         pa_junc_pct[pa_keep] = junc_arr[pa_keep] / total_arr[pa_keep]
#         pa_junc_pct[pa_replace] = 0
#         return pa_junc_pct, junc_arr, total_arr
    
#     ## Step 1: select pa with more than 40% junction reads
#     pa_junc_pct = junc_read_arr / total_read_arr ## sorted in ascending order by pa position
#     pa_to_merge_arr = pa_unique_label_arr[pa_junc_pct > junction_pct_thres] ## sorted in ascending order by pa position
#     ## debug
#     if len(pa_to_merge_arr) > 0:
#         junc=1
#     else:
#         junc=0
#     ## debug
#     close_to_en1 = 0 ## debug
#     close_to_en2 = 0 ## debug
#     close_no = 0 ## debug
#     ## Step 2: merging pa list
#     while len(pa_to_merge_arr)>0:
#         ori_len = len(pa_to_merge_arr)
#         first_pa = pa_to_merge_arr[0]
#         if pa_junc_pct[first_pa] <= 0.4:
#             ## exclude pa with junction site from to-be-merged list if its %junction read is lower than threshold
#             pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
#             continue
#         median_en1 = np.median(pa_seg_en1_arr[pa_seg_en1_arr[:,1]==first_pa, 0])
#         median_en2 = np.median(pa_seg_en2_arr[pa_seg_en2_arr[:,1]==first_pa, 0])
#         ## if two ends are closest to the same pa -> exclude that pa
#         if find_closest_pa(median_en1, pa_loc_arr, pa_unique_label_arr) == find_closest_pa(median_en2, pa_loc_arr, pa_unique_label_arr):
#             pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
#             continue
#         ## if first pa in to-be-merged list is before its end of segment 1
#         if first_pa == find_closest_pa(median_en1, pa_loc_arr, pa_unique_label_arr): ## strand "-"
#             close_to_en1 +=1 ## debug
#             ## select pa that is the closet to 
#             pa = find_closest_pa(median_en2, pa_loc_arr, pa_unique_label_arr) ## strand "-"
#             ## if pa=None, then continue
#             if pa==None:
#                 ## exclude pa with no other close pa
#                 pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
#                 continue
#             if weight_pa_arr[first_pa] > weight_pa_arr[pa]:
#                 ## merge pa into first_pa
#                 pa_dict, pa_keep, pa_replace = update_pa_dict(first_pa, pa, pa_dict)
#                 ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
#                 pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct, junc_read_arr, total_read_arr, pa_keep, pa_replace)
#             else:
#                 ## merge first_pa into pa
#                 pa_dict, pa_keep, pa_replace = update_pa_dict(pa, first_pa, pa_dict)
#                 ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
#                 pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct, junc_read_arr, total_read_arr, pa_keep, pa_replace)
#                 ## exclude first_pa from to-be-merged array
#                 pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
                
#         elif first_pa == find_closest_pa(median_en2, pa_loc_arr, pa_unique_label_arr): ## strand "-"
#             close_to_en2 +=1 ## debug
#             pa = find_closest_pa(median_en1, pa_loc_arr, pa_unique_label_arr) ## strand "-"
#             ## if pa=None, then continue
#             if pa==None:
#                 ## exclude pa with no other close pa
#                 pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
#                 continue
#             if weight_pa_arr[first_pa] > weight_pa_arr[pa]:
#                 ## merge pa into first_pa
#                 pa_dict, pa_keep, pa_replace = update_pa_dict(first_pa, pa, pa_dict)
#                 ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
#                 pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct, junc_read_arr, total_read_arr, pa_keep, pa_replace)
#             else:
#                 ## merge first_pa into pa
#                 pa_dict, pa_keep, pa_replace = update_pa_dict(pa, first_pa, pa_dict)
#                 ## recalculate percentage of junction read. first_pa=(first_pa+pa), pa=0
#                 pa_junc_pct, junc_read_arr, total_read_arr = recal_pa_junc_pct(pa_junc_pct, junc_read_arr, total_read_arr, pa_keep, pa_replace)
#                 pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
#         else:
#             close_no +=1 ## debug
#             pa_to_merge_arr = pa_to_merge_arr[pa_to_merge_arr != first_pa]
        
#         if len(pa_to_merge_arr) == ori_len:
#             pa_to_merge_arr = pa_to_merge_arr[1:]

#     ## re-label pa id: merge pa with junction reads. Ex: [0,1,2,3,4] ->[1,2,3,4]
#     while np.sum(np.isin(pa_unique_label_arr, [pa for pa in pa_dict.keys()])) > 0:
#         pa_unique_label_arr = pd.Series(pa_unique_label_arr).replace(pa_dict).to_numpy()
#         pa_label_arr = pd.Series(pa_label_arr).replace(pa_dict).to_numpy()
    
    
#     ## exclude pa with pct lower than 5%. Ex: [1,2,3,4] -> [1,3,4]
#     pa_remain_arr = np.array(list(set(pa_unique_label_arr)))
#     pa_total_pct = total_read_arr[pa_remain_arr]/np.sum(total_read_arr[pa_remain_arr])
#     pa_remove_arr = pa_remain_arr[pa_total_pct <= total_read_pct_thres] ## list pa with low percentage
#     pa_remain_arr = pa_remain_arr[pa_total_pct > total_read_pct_thres] ## list pa with high percentage
#     uni_comp = len(pa_remain_arr)
    
#     ## attributes of final class Parameters
#     pa_beta_arr = pa_beta_arr[pa_remain_arr]
#     pa_loc_arr = pa_loc_arr[pa_remain_arr]
#     ## re-label pa id so that they are consecutive numbers and starting from 0. Ex: [1,2,3,4] -> [0,1,2,3]
#     tmp_dict = {pa_remain_arr[new]:new for new in range(len(pa_remain_arr))}
#     for pa in pa_remove_arr:
#         tmp_dict[pa] = uni_comp
#     pa_label_arr = pd.Series(pa_label_arr).replace(tmp_dict).to_numpy()
#     ws_arr = total_read_arr[pa_remain_arr] / np.sum(total_read_arr[pa_remain_arr])
#     utr_st = np.min(utr_st_arr)
#     utr_en = np.max(utr_en_arr)
    
#     para = Parameters(title="Final Result"
#                       , alpha_arr=utr_en - pa_loc_arr + 1 ## strand "-"
#                       , beta_arr=pa_beta_arr
#                       , ws=ws_arr
#                       , K=len(pa_remain_arr)
#                       , L=0
#                       , cb_id_arr=pa_cb_arr[pa_label_arr < uni_comp] ## only keep reads with respective to any pa site
#                       , readID_arr=pa_read_arr[pa_label_arr < uni_comp] ## only keep reads with respective to any pa site
#                      )
#     para.label_arr=pa_label_arr[pa_label_arr < uni_comp] ## only keep reads with respective to any pa site
#     if len(gene_key.split(sep=":"))==1:
#         para.gene_info_str=str(chrom)+":"+gene+":"+str(1)+":"+str(utr_st)+"-"+str(utr_en)+":"+strand ## 4:ENSG00000174780:1:56475376-56477472:+
#     else:
#         para.gene_info_str=str(chrom)+":"+gene_key+":"+str(utr_st)+"-"+str(utr_en)+":"+strand ## 4:ENSG00000174780:2:56475376-56477472:+
#     tmp, counts=np.unique(para.label_arr, return_counts=True)
#     #     return para
#     ## debug
#     if len(pa_dict)>0:
#         print(gene_key, pa_dict)
#     if len(pa_dict)>0:
#         change=1
#     else:
#         change=0
#     return para, junc, change
#     ## debug



















