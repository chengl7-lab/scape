library(GenomicRanges)
library(GenomicFeatures)
library(annotatr)
library(GenomicAlignments)
library(org.Mm.eg.db) ##tien
library(org.Hs.eg.db) ##tien
library(magrittr) ##tien
library(tidyverse) ##tien



annotate_from_gtf <-
  function (gtfFile,
            genome_ver = c('Mm10', 'Hg38'),
            cores = 1) 
  {
    
#     Aim to prepare region names and boundary positions.
#     Region can be exon, intron, last exon, last exon within 1k, coding sequence, 5 UTR, 3 UTR, intergenic, promoter,...

#     Return: GRanges object


    genome_ver <- match.arg(genome_ver)
    species_map <- c('Hg38' = 'Hg',
                      'Hg19' = 'Hg',
                      'Mm10' = 'Mm',
                      'Mm9' = 'Mm'
                    )

      
    # Get org database for mouse/human. For EntrezID to gene symbol (name) mapping and vice versa.
    if (species_map[[genome_ver]] == 'Mm') 
    {
      require(org.Mm.eg.db)
      x_map = get(sprintf('org.Mm.egSYMBOL', genome_ver))
    } 
    else if (species_map[[genome_ver]] == 'Hg') 
    {
      require(org.Hs.eg.db)
      x_map = get(sprintf('org.Hs.egSYMBOL', genome_ver))
    } 
    else 
    {
      stop("Can't recognize the genome version!")
    }

      
    # databases includes transcript annotation and connection between transcript, exon, cds,... Return TxDb object
    # keys: 'CDSID','CDSNAME','EXONID','EXONNAME','GENEID','TXID','TXNAME'
    txdb <- GenomicFeatures::makeTxDbFromGFF(gtfFile, format = "gff3")
    # boundary of each transcript ID grouping by gene ID. Return GRangeList object
    # each entry's key is ensemblID and their values are transcripts and their boundary positions.
    ebg <- GenomicFeatures::transcriptsBy(txdb, by = "gene")

      
    # Get gene symbol (name) that maps to 1 EntrezID
    mapped_genes = AnnotationDbi::mappedkeys(x_map)
    # dataframe includes 2 columns: EntrezID, gene symbol (name)
    eg2symbol = as.data.frame(x_map[mapped_genes])

      
    # Add EnsemblID to dataframe
    if (species_map[[genome_ver]] == 'Mm') {
        eg2symbol$ensemble <- AnnotationDbi::mapIds(
                                org.Mm.eg.db, 
                                keys = eg2symbol$gene_id,
                                keytype = "ENTREZID",
                                column = "ENSEMBL")
    } else if (species_map[[genome_ver]] == 'Hg') {
        eg2symbol$ensemble <- AnnotationDbi::mapIds(
                                org.Hs.eg.db, 
                                keys = eg2symbol$gene_id,
                                keytype = "ENTREZID",
                                column = "ENSEMBL")
    } else {
      stop("Can't recognize the genome version!")
    }

      
    # Query all transcripts in database. Return GRanges object
    tx_gr = GenomicFeatures::transcripts(txdb, columns = c('TXID', 'GENEID', 'TXNAME'))
    
    # Returns dataframe includes 3 columns: 'TXID', 'GENEID', 'TXNAME'
    # The same as converting tx_gr to dataframe
    id_maps = AnnotationDbi::select(txdb,
                                      keys = as.character(GenomicRanges::mcols(tx_gr)$TXID),
                                      columns = c('TXNAME', 'GENEID'),
                                      keytype = 'TXID'
                                    )
    
      
    ## ==================== DEFINING REGION ====================
    
    ## ==================
    ## ====== EXON ======
    # Boundary of exon grouping by transcript ID. Return GRangeList object.
    # each entry's key is transcript ID and their values are exons and their boundary positions.
    exons_grl = GenomicFeatures::exonsBy(txdb, by = 'tx', use.names = TRUE)
      
    # Create Rle of the tx_names where value is transcript ID and length is respective number of exon
    # This is then used for extracting list of unique transcript ID
    exons_txname_rle = S4Vectors::Rle(names(exons_grl), S4Vectors::elementNROWS(exons_grl))
    # Return list of transcript ID only. This is used to add transcript ID to GRanges "exons_gr"
    exons_txname_vec = as.character(exons_txname_rle)
    
    # Unlist and Return GRanges object
    exons_gr = unlist(exons_grl, use.names = FALSE)
      
    # Add transcript ID, EnsemblID, symbol, Entrez ID, feature name
    GenomicRanges::mcols(exons_gr)$tx_name = exons_txname_vec
    GenomicRanges::mcols(exons_gr)$gene_id = id_maps[match(GenomicRanges::mcols(exons_gr)$tx_name, id_maps$TXNAME)
                                                     , 'GENEID']
    GenomicRanges::mcols(exons_gr)$symbol = eg2symbol[match(GenomicRanges::mcols(exons_gr)$gene_id,
                                                            eg2symbol$ensemble)
                                                      , 'symbol']
    GenomicRanges::mcols(exons_gr)$entrez = eg2symbol[match(GenomicRanges::mcols(exons_gr)$gene_id,
                                                            eg2symbol$ensemble)
                                                      , 'gene_id']
    GenomicRanges::mcols(exons_gr)$id = paste0('Exon:ExonRank', exons_gr$exon_rank)


      
      
    ## =======================
    ## ====== LAST EXON ======
    x <- lapply(exons_grl, function(x) {exonrank <- length(x$exon_rank)
                                          if (as.character(strand(x)) == "+") {
                                            x[exonrank]
                                          } else{
                                            x[1]
                                          }
                                        }
               )

    lastexons_grl <- GRangesList(x)
    lastexon_txname_rle = S4Vectors::Rle(names(lastexons_grl),
                                         S4Vectors::elementNROWS(lastexons_grl))
    lastexons_txname_vec = as.character(lastexon_txname_rle)
    lastexons_gr = unlist(lastexons_grl, use.names = FALSE)
      
    # Add transcript ID, EnsemblID, symbol, Entrez ID, feature name
    GenomicRanges::mcols(lastexons_gr)$tx_name = lastexons_txname_vec
    GenomicRanges::mcols(lastexons_gr)$gene_id = id_maps[match(GenomicRanges::mcols(lastexons_gr)$tx_name,
                                                               id_maps$TXNAME),
                                                         'GENEID']
    GenomicRanges::mcols(lastexons_gr)$symbol = eg2symbol[match(GenomicRanges::mcols(lastexons_gr)$gene_id,
                                                                eg2symbol$ensemble)
                                                          , 'symbol']
    GenomicRanges::mcols(lastexons_gr)$entrez = eg2symbol[match(GenomicRanges::mcols(lastexons_gr)$gene_id,
                                                                eg2symbol$ensemble)
                                                          , 'gene_id']
    GenomicRanges::mcols(lastexons_gr)$id = paste0('lastexon:ExonRank', lastexons_gr$exon_rank)

      
      
    ## ========================================
    ## ====== LAST EXON WITHIN 1K WINDOW ======
    pos = lastexons_gr[strand(lastexons_gr) == '+', ]
    neg = lastexons_gr[strand(lastexons_gr) == '-', ]
    start(pos) <- end(pos) + 1
    end(pos) <- end(pos) + 1000
    end(neg) <- start(neg) - 1
    start(neg) <- start(neg) - 1000

    lastexons1k_gr_ <- c(pos, neg)
    GenomicRanges::mcols(lastexons1k_gr_)$id = paste0('LastExon1Kb:ExonRank', lastexons_gr$exon_rank)


      
      
      
    ## =============================
    ## ====== CODING SEQUENCE ======
    cds_grl = GenomicFeatures::cdsBy(txdb, by = 'tx', use.names = TRUE)
    cds_txname_rle = S4Vectors::Rle(names(cds_grl), S4Vectors::elementNROWS(cds_grl))
    cds_txname_vec = as.character(cds_txname_rle)
    # Unlist and add the tx_names
    cds_gr = unlist(cds_grl, use.names = FALSE)
      
    # Add transcript ID, EnsemblID, symbol, Entrez ID, feature name
    GenomicRanges::mcols(cds_gr)$tx_name = cds_txname_vec
    GenomicRanges::mcols(cds_gr)$gene_id = id_maps[match(GenomicRanges::mcols(cds_gr)$tx_name, id_maps$TXNAME)
                                                   , 'GENEID']
    GenomicRanges::mcols(cds_gr)$symbol = eg2symbol[match(GenomicRanges::mcols(cds_gr)$gene_id, eg2symbol$ensemble)
                                                    , 'symbol']
    GenomicRanges::mcols(cds_gr)$entrez = eg2symbol[match(GenomicRanges::mcols(cds_gr)$gene_id, eg2symbol$ensemble)
                                                    , 'gene_id']
    GenomicRanges::mcols(cds_gr)$type = sprintf('CDS')
    GenomicRanges::mcols(cds_gr)$id = paste0('CDS:ExonRank', cds_gr$exon_rank)


      
    ## ====================
    ## ====== 5' UTR ======
    fiveUTRs_grl = GenomicFeatures::fiveUTRsByTranscript(txdb, use.names = TRUE)

    fiveUTRs_txname_rle = S4Vectors::Rle(names(fiveUTRs_grl),
                                         S4Vectors::elementNROWS(fiveUTRs_grl))
    fiveUTRs_txname_vec = as.character(fiveUTRs_txname_rle)
    fiveUTRs_gr = unlist(fiveUTRs_grl, use.names = FALSE)
      
    # Add transcript ID, EnsemblID, symbol, Entrez ID, feature name
    # NOTE: here we match on the tx_name because the tx_id is not given
    GenomicRanges::mcols(fiveUTRs_gr)$tx_name = fiveUTRs_txname_vec
    GenomicRanges::mcols(fiveUTRs_gr)$gene_id = id_maps[match(GenomicRanges::mcols(fiveUTRs_gr)$tx_name, id_maps$TXNAME)
                                                        , 'GENEID']
    GenomicRanges::mcols(fiveUTRs_gr)$symbol = eg2symbol[match(GenomicRanges::mcols(fiveUTRs_gr)$gene_id,
                                                               eg2symbol$ensemble)
                                                         , 'symbol']
    GenomicRanges::mcols(fiveUTRs_gr)$entrez = eg2symbol[match(GenomicRanges::mcols(fiveUTRs_gr)$gene_id,
                                                               eg2symbol$ensemble)
                                                         , 'gene_id']

    GenomicRanges::mcols(fiveUTRs_gr)$type = sprintf('5UTRs')
    GenomicRanges::mcols(fiveUTRs_gr)$id = paste0('5UTR:ExonRank', fiveUTRs_gr$exon_rank)


      
    ## ====================
    ## ====== 3' UTR ======
    threeUTRs_grl = GenomicFeatures::threeUTRsByTranscript(txdb, use.names = TRUE)
      
    threeUTRs_txname_rle = S4Vectors::Rle(names(threeUTRs_grl),
                                          S4Vectors::elementNROWS(threeUTRs_grl))
    threeUTRs_txname_vec = as.character(threeUTRs_txname_rle)
    threeUTRs_gr = unlist(threeUTRs_grl, use.names = FALSE)
      
    # Add transcript ID, EnsemblID, symbol, Entrez ID, feature name
    # NOTE: here we match on the tx_name because the tx_id is not given
    GenomicRanges::mcols(threeUTRs_gr)$tx_name = threeUTRs_txname_vec
    GenomicRanges::mcols(threeUTRs_gr)$gene_id = id_maps[match(GenomicRanges::mcols(threeUTRs_gr)$tx_name,
                                                               id_maps$TXNAME)
                                                         , 'GENEID']
    GenomicRanges::mcols(threeUTRs_gr)$symbol = eg2symbol[match(GenomicRanges::mcols(threeUTRs_gr)$gene_id,
                                                                eg2symbol$ensemble)
                                                          , 'symbol']
    GenomicRanges::mcols(threeUTRs_gr)$entrez = eg2symbol[match(GenomicRanges::mcols(threeUTRs_gr)$gene_id,
                                                                eg2symbol$ensemble)
                                                          , 'gene_id']

    GenomicRanges::mcols(threeUTRs_gr)$id = sprintf('3UTRs')

      
      
    ## ======================================
    ## ====== 3' UTR within 1kb window ======
    pos = threeUTRs_gr[strand(threeUTRs_gr) == '+', ]
    neg = threeUTRs_gr[strand(threeUTRs_gr) == '-', ]

    start(pos) <- end(pos) + 1
    end(pos) <- end(pos) + 1000

    end(neg) <- start(neg) - 1
    start(neg) <- start(neg) - 1000

    threeUTRs1Kb <- c(neg, pos)
    threeUTRs1Kb$id <- '3UTRs_1kb'

      
      
    ## ======================================
    ## ====== 3' UTR within 2kb window ======
    ###for 2kb
    pos = threeUTRs1Kb[strand(threeUTRs1Kb) == '+', ]
    neg = threeUTRs1Kb[strand(threeUTRs1Kb) == '-', ]

    start(pos) <- end(pos) + 1
    end(pos) <- end(pos) + 1000

    end(neg) <- start(neg) - 1
    start(neg) <- start(neg) - 1000

    threeUTRs2Kb <- c(pos, neg)
    threeUTRs2Kb$id <- '3UTRs_2kb'

      

    ## ====================
    ## ====== INTRON ======
    introns_grl = GenomicFeatures::intronsByTranscript(txdb, use.names = TRUE)
      
    introns_txname_rle = S4Vectors::Rle(names(introns_grl), S4Vectors::elementNROWS(introns_grl))
    introns_txname_vec = as.character(introns_txname_rle)
    introns_gr = unlist(introns_grl, use.names = FALSE)
      
    # Add transcript ID, EnsemblID, symbol, Entrez ID, feature name
    # NOTE: here we match on the tx_name because the tx_id is not given
    GenomicRanges::mcols(introns_gr)$tx_name = introns_txname_vec
    GenomicRanges::mcols(introns_gr)$gene_id = id_maps[match(GenomicRanges::mcols(introns_gr)$tx_name, id_maps$TXNAME)
                                                       , 'GENEID']
    GenomicRanges::mcols(introns_gr)$symbol = eg2symbol[match(GenomicRanges::mcols(introns_gr)$gene_id,
                                                              eg2symbol$ensemble)
                                                        , 'symbol']
    GenomicRanges::mcols(introns_gr)$entrez = eg2symbol[match(GenomicRanges::mcols(introns_gr)$gene_id,
                                                              eg2symbol$ensemble)
                                                        , 'gene_id']



    introns_gr <-
      parallel::mclapply(split(introns_gr, introns_gr$tx_name), function(x) {
        if (unique(strand(x)) == "-") {
          x$id <- paste('Intron:Rank', rev(seq(1:length(x))), sep = '')
        } else{
          x$id <- paste('Intron:Rank', seq(1:length(x)), sep = '')
        }
        x
      }, mc.cores = cores)
    names(introns_gr) <- NULL
    introns_gr <- unlist(as(introns_gr, "GRangesList"))




    ## ======================
    ## ====== PROMOTER ======
    promoters_gr = GenomicFeatures::promoters(txdb, upstream = 1000, downstream = 0)
      
    # Add EnsemblID, symbol, type, feature name
    GenomicRanges::mcols(promoters_gr)$gene_id = id_maps[match(GenomicRanges::mcols(promoters_gr)$tx_id, id_maps$TXID)
                                                         , 'GENEID']
    GenomicRanges::mcols(promoters_gr)$symbol = eg2symbol[match(GenomicRanges::mcols(promoters_gr)$gene_id,
                                                                eg2symbol$gene_id)
                                                          , 'symbol']
    GenomicRanges::mcols(promoters_gr)$type = sprintf('%s_genes_promoters', genome_ver)
    GenomicRanges::mcols(promoters_gr)$id = paste0('promoter:', seq_along(promoters_gr))

      
      
    ## ======================================
    ## ====== PROMOTER WITHIN 1 to 5kb ======
    GenomicRanges::mcols(promoters_gr) = GenomicRanges::mcols(promoters_gr)[, c('id', 'tx_name', 'gene_id', 'symbol', 'type')]
    colnames(GenomicRanges::mcols(promoters_gr)) = c('id', 'tx_id', 'gene_id', 'symbol', 'type')


    onetofive_gr = GenomicRanges::flank(promoters_gr,
                                        width = 4000,
                                        start = TRUE,
                                        both = FALSE)
    onetofive_gr = GenomicRanges::trim(onetofive_gr)
    # Add Entrez ID, symbol, and type (all but type are inherited from promoters_gr)
    GenomicRanges::mcols(onetofive_gr)$id = paste0('1to5kb:', seq_along(onetofive_gr))
    GenomicRanges::mcols(onetofive_gr)$type = sprintf('%s_genes_1to5kb', genome_ver)

      
      
    ## ========================
    ## ====== INTERGENIC ======
    genic_gr = c(GenomicRanges::granges(tx_gr))
    GenomicRanges::strand(genic_gr) = '*'
    intergenic_gr =  (genic_gr)
    intergenic_gr = GenomicRanges::gaps(intergenic_gr)

    # A quirk in gaps gives the entire + and - strand of a chromosome, ignore those
    intergenic_gr = intergenic_gr[GenomicRanges::strand(intergenic_gr) == '*']

    GenomicRanges::mcols(intergenic_gr)$tx_name = 'NA'
    GenomicRanges::mcols(intergenic_gr)$gene_id = 'NA'
    GenomicRanges::mcols(intergenic_gr)$symbol = 'NA'
    GenomicRanges::mcols(intergenic_gr)$entrez = 'NA'
    GenomicRanges::mcols(intergenic_gr)$id = 'INTERGENIC'

      
      
      
    ## ====================
    ## ====== RESULT ======
    use_ind <- c("tx_name", "gene_id", "symbol", "entrez", "id")
    ##for cds
    cds_gr <- cds_gr[, use_ind]

    ##for 3utrs
    threeUTRs_gr <- threeUTRs_gr[, use_ind]

    ##for 3utrs-1kb
    threeUTRs1Kb <- threeUTRs1Kb[, use_ind]

    ##for 3utrs-2kb
    threeUTRs2Kb <- threeUTRs2Kb[, use_ind]

    ##for 5UTR
    fiveUTRs_gr <- fiveUTRs_gr[, use_ind]

    ##fo intergenic
    intergenic_gr <- intergenic_gr[, use_ind]

    ##for intron
    introns_gr <- introns_gr[, use_ind]
    ## for exon
    exons_gr <- exons_gr[, use_ind]

    ## 2019-6-14 modified to remove 5'utr extend

    annotation_db <- c(
      introns_gr,
      cds_gr,
      exons_gr,
      threeUTRs1Kb,
      threeUTRs_gr,
      fiveUTRs_gr,
      threeUTRs2Kb,
      intergenic_gr,
      lastexons1k_gr_
    )

    return(annotation_db)
  }



AnnotationSite <-
  function(pAsite,
           gtfFile,
           genome_ver,
           minoverlap=1L,
           annotLevels = c(
             "3UTRs",
             "5UTR", ##tien change order
             "Exon",
             "Intron",
             "CDS",
             "LastExon1Kb",
             "3UTRs_1kb",
             "3UTRs_2kb",
             "INTERGENIC"
           ),
           cores = 1) 
  {
#     """
#     Input:
#         - pAsite: list of pAsite (chr:alpha:beta:strand). Example: [1:6275401:72:+, 1:6275770:9:+]
#     Return: list of three dataframes:
#         - annot_res_lst : annotation that follows order of region of interest and selects randomly one among many regions that are mapped to 1 PA site
#         - annot_res : raw annotation
#         - annot_res_uni : unique raw annotation by distinct region type (annotLevels)
#     """
      
    # convert list to dataframe. 18:31595428:5.0:+:1:ENSG00000118271:2
    pa_info <- as.data.frame(do.call(rbind, (strsplit(pAsite, split = ':'))))
    colnames(pa_info) <- c('chr', 'pa_pos', 'beta', 'strand', 'paID', 'geneID', "utrID") ##tien
#     pa_info$paID <- NULL
#     pa_info$utrID <- NULL
    pa_info$beta_int <- as.numeric(as.character(pa_info$beta))
    pa_info$end <- as.numeric(as.character(pa_info$pa_pos)) + pa_info$beta_int
    pa_info$start <- as.numeric(as.character(pa_info$pa_pos)) - pa_info$beta_int
    pa_info$score <- '.'
    
    # conver dataframe to GRangers object
    # input dataframe musts include columns: chr, start, end, strand, score
    pa_info <- GenomicRanges::makeGRangesFromDataFrame(pa_info, keep.extra.columns = TRUE)
    cat("Done making Granges")
    # path directory
    gtf_prefix <- strsplit(basename(gtfFile), split = '[.]')[[1]][1]
    annot_db_file <- file.path(dirname(gtfFile), paste0(gtf_prefix, '_scape_annotation.Rds')) ##tien

    # prepare GRanges of region of interest from GTF (the same as what is used in calculating PAsite)
    if (file.exists(annot_db_file)) {
      annot_db <- readRDS(annot_db_file)
      cat("Read existing annotation database", "\n")
    } else {
      cat("Creating annotation database ")
      start.time <- Sys.time()
      annot_db <- annotate_from_gtf(gtfFile = gtfFile
                                    , genome_ver = genome_ver
                                    , cores = cores
                                   )
      saveRDS(annot_db, file = annot_db_file)
      end.time <- Sys.time()
      time.taken <- end.time - start.time
      cat("in ", time.taken, "\n")
    }

    # Search each entry of pa_info against GRanges of all regions
    # return Granges object
    annot_res <- annotatr::annotate_regions(regions = pa_info,
                                              annotations = annot_db,
                                              minoverlap = minoverlap,
                                              ignore.strand = FALSE,
                                              quiet = FALSE
                                            )
    cat("Done annotating regions")
    annot_res <- as.data.frame(annot_res, row.names = 1:length(annot_res))

    inds <-c( "seqnames",
                "end",
                "strand",
                "annot.start",
                "annot.end",
                "annot.tx_name",
                "annot.gene_id",
                "annot.symbol",
                "annot.entrez",
                "annot.id", ## type:rank
                "beta",
                "geneID",
                "pa_pos",
                 "utrID",
             "paID"
                 
              )

    annot_res <- annot_res[, inds]
    ## Add 2 more columns to annot_res
    annot_res <- cbind(annot_res
                       , as.data.frame(do.call(rbind, strsplit(annot_res$annot.id, split = ':')))
                      )
     
    colnames(annot_res)[c(16, 17)] <- c('Type', 'Rank')
    annot_res$Type <- factor(annot_res$Type, levels = annotLevels)
      
    ## split dataframe to list where each element has key = chr:end:beta:strand:geneID and its value is a dataframe of all respective regions
    ## After splitting, order (of factor) will be re-arranged to be respective to subset
    annot_res_lst <- split(annot_res
                           , paste(annot_res$seqnames
                                   , annot_res$pa_pos
                                   , annot_res$beta
                                   , annot_res$strand
                                   , annot_res$paID
                                    , annot_res$geneID
                                   , annot_res$utrID
                                   , sep = ':'))

    annot_res_lst <- lapply(annot_res_lst, function(x) {
                                                          ind <- order(x$Type)[1]
                                                          return(x[ind,])
                                                        })

    annot_res_lst <- do.call(rbind, annot_res_lst)
    annot_res_lst$pa_info <- rownames(annot_res_lst)
    colnames(annot_res_lst)[colnames(annot_res_lst) == 'end'] <- 'pa_loc' ##tien
#     annot_res_lst <- annot_res_lst[annot_res_lst$geneID == annot_res_lst$annot.gene_id, ]
#     annot_res <- annot_res[annot_res$geneID == annot_res$annot.gene_id, ]
    ## list of outputs
    out=list() ##tien
      
      
    ## Assign region type that has rank = 1 following given order. 
    ## 1. If pa site is assigned to one type of region but different geneIDs, it will order randomly.
    annot_res_lst <- annot_res_lst[annot_res_lst$geneID == annot_res_lst$annot.gene_id, ]
    out$annot_res_lst <- annot_res_lst ##tien

    ## 2. Raw assignment. One pa site can be assigned to many types of region and many genes
    annot_res <- annot_res[annot_res$geneID == annot_res$annot.gene_id, ]
    out$annot_res <- annot_res ##tien
      
      
    annot_res_uni <- annot_res %>%
                              distinct(seqnames, pa_pos, strand
                                       , annot.gene_id
                                       , annot.symbol
                                       , annot.entrez
                                       , beta
                                       , geneID
                                       , Type
                                       , paID
                                       , utrID
                                      ) ##tien
    annot_res_uni$pa_info <- paste(annot_res_uni$seqnames
                                    , annot_res_uni$pa_pos
                                    , annot_res_uni$beta
                                    , annot_res_uni$strand
                                   , annot_res_uni$paID
                                    , annot_res_uni$geneID
                                   , annot_res_uni$utrID
                                   , sep=":"
                                   )
    annot_res_uni$Type <- factor(annot_res_uni$Type, levels = annotLevels) ##tien

    ## split dataframe to list where each element has key = chr:end:beta:strand:gene and its value is a dataframe of all respective regions
    annot_res_uni_lst <- split(annot_res_uni,
                                paste(annot_res_uni$seqnames
                                      , annot_res_uni$pa_pos
                                      , annot_res_uni$beta
                                      , annot_res_uni$strand
                                   , annot_res_uni$paID
                                      , annot_res_uni$annot.gene_id
                                   , annot_res_uni$utrID
                                      , sep = ':')) ##tien
    annot_res_uni_lst <- lapply(annot_res_uni_lst, function(x) {
                                                  ind <- order(x$Type)[1]
                                                  return(x[ind,])
                                                })
    annot_res_uni_lst <- do.call(rbind, annot_res_uni_lst) ##tien
#     colnames(annot_res_uni_lst)[colnames(annot_res_uni_lst) == 'end'] <- 'pa_loc' ##tien

    ## 3. One pa site can be assigned to only 1 type of region of each gene. It can be assigned to many genes at the same time.  
    annot_res_uni_lst <- annot_res_uni_lst[annot_res_uni_lst$geneID == annot_res_uni_lst$annot.gene_id, ]
    out$annot_res_uni_lst <- annot_res_uni_lst ##tien

      
      
    return(out) ##tien
#     return(annot_res_lst)
  }


