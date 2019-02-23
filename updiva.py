#!/usr/bin/env python3
import sys
import os
import re
import argparse
import gzip
import logging
import pysam
import tempfile
import scipy.stats
import multiprocessing as mp
import numpy as np
from shutil import copyfile,copyfileobj
from itertools import repeat
from collections import defaultdict
from copy import copy
from parse_vcf import VcfReader
from vase.ped_file import PedFile

chrom_re = re.compile(r'''^(chr)?[1-9X][0-9]?$''')

gt_ids = [#counts for testing for excess homozygosity
          'het', 'hom_alt', 'hom_ref',
          # counts for UPD - bi-parental, mat/paternal isodisomy, heterodisomy
          'bpi', 'mat_i_upd', 'mat_a_upd', 'pat_i_upd', 'pat_a_upd'
         ]

class GtCounter(object):
    ''' Count number of hets/homs per contig for a sample.'''

    def __init__(self, samples, counts=dict()):
        self.samp_indices = dict((s, n) for n,s in enumerate(samples))
        self.gt_indices = dict((r, n) for n,r in enumerate(gt_ids))
        self.counts = counts
        self.samples = samples

    def __radd__(self, other):
        return self + other

    def __add__(self, other):
        if isinstance(self, int) and self == 0:
            return GtCounter(other.samples, other.counts)
        elif isinstance(other, int) and other == 0:
            return GtCounter(self.samples, self.counts)
        totals = copy(self.counts)
        for chrom in other.counts:
            if chrom in totals:
                totals[chrom] += other.counts[chrom]
            else:
                totals[chrom] = other.counts[chrom]
        return GtCounter(self.samples, totals)

    def count_genotype(self, chrom, sample, gt):
        if chrom not in self.counts:
            self.counts[chrom] = np.zeros((len(gt_ids), len(self.samples)),
                                          dtype=int)
        self.counts[chrom][self.gt_indices[gt], self.samp_indices[sample]] += 1

def _process_runner(tup):
    kwargs1, kwargs2 = tup
    kwargs2.update(kwargs1)
    return get_gt_counts(**kwargs2)

def main(vcf, ped=None, output=None, coordinate_table=None, threads=1,
         progress_interval=10000):
    parents = None
    children = set()
    females = set()
    samples = None
    logger = get_logger()
    if coordinate_table and not coordinate_table.endswith(".gz"):
        coordinate_table += '.gz'
    if ped is not None:
        logger.info("Parsing PED file.")
        samples, parents, children, females = samples_from_ped(ped, vcf,
                                                               logger)
    else:
        logger.info("No PED file - analyzing all samples for homozygosity " +
                    "only")
        samples = VcfReader(vcf).header.samples
    kwargs = {'vcf': vcf, 'prog_interval': progress_interval,
              'coordinate_table': coordinate_table, 'samples': samples,
              'parents': parents, 'children': children, 'females': females}
    if threads > 1:
        contig_args = ({'contig': x} for x in get_seq_ids(vcf))
        if not contig_args:
            raise RuntimeError("No valid contigs identified in {}".format(fai))
        with mp.Pool(threads) as p:
            pool_results = p.map(_process_runner, zip(contig_args,
                                                      repeat(kwargs)))
            results = (x[0] for x in pool_results)
            tmp_tables = (x[1] for x in pool_results)
        logger.info("Collating counts.")
        results = sum(results)
        if coordinate_table:
            logger.info("Collating coordinates.")
            with open(coordinate_table, 'wb') as wfh:
                for fn in tmp_tables:
                    with open(fn, 'rb') as rfh:
                        copyfileobj(rfh, wfh)
    else:
        kwargs['logger'] = logger
        results,tmp_table = get_gt_counts(**kwargs)
        if coordinate_table:
            copyfile(tmp_table, coordinate_table)
    logger.info("Parsing results.")
    parse_results(results, logger, children, females, output)

def parse_results(gt_counts, logger, children, females, output=None):
    if output is None:
        out = sys.stdout
    else:
        out = open(output, 'wt')
    genomewide_counts = sum(gt_counts.counts[c] for c in gt_counts.counts if
                            "X" not in c)
    gt_indices = dict((r, n) for n,r in enumerate(gt_ids))
    upd_types = ('mat_upd', 'mat_i_upd', 'pat_upd', 'pat_i_upd')
    out.write("\t".join(["Sample", "Chrom", "Test", "N", "Calls", "Fraction",
                     "vs_chrom", "vs_self"]) + "\n")
    logger.info("Calculating genomewide total")
    genome_all_total = np.sum(genomewide_counts[:2]) #does not include hom-ref
    het_i = gt_indices['het'] #for readability
    genome_all_upd = get_upd_counts(genomewide_counts, gt_indices)
    for chrom in gt_counts.counts:
        logger.info("Parsing results for chomosome {}".format(chrom))
        chrom_total = np.sum(gt_counts.counts[chrom][:2])
        chrom_all_hom = chrom_total - np.sum(gt_counts.counts[chrom][het_i])
        chrom_informative_total = np.sum(gt_counts.counts[chrom][3:])
        chrom_upd = get_upd_counts(gt_counts.counts[chrom], gt_indices)
        chrom_upd_totals = dict((k, np.sum(chrom_upd[k])) for k in upd_types)
        for sample in gt_counts.samples:
            if 'X' in chrom and sample not in females:
                continue
            i = gt_counts.samp_indices[sample]
            samp_genomewide_total = np.sum(genomewide_counts[:2,i])
            samp_hom_total = (samp_genomewide_total -
                              np.sum(gt_counts.counts[c][het_i,i] for c in
                                     gt_counts.counts))
            samp_chrom_total = np.sum(gt_counts.counts[chrom][:2,i])
            samp_chrom_hom = (samp_chrom_total -
                              gt_counts.counts[chrom][het_i,i])
            samp_vs_chrom = scipy.stats.binom_test(samp_chrom_hom,
                                                   samp_chrom_total,
                                                   chrom_all_hom/chrom_total,
                                                   alternative='greater')
            samp_vs_self = scipy.stats.binom_test(
                    samp_chrom_hom,
                    samp_chrom_total,
                    samp_hom_total/samp_genomewide_total,
                    alternative='greater')
            frac = "{:.3g}".format(samp_chrom_hom/samp_chrom_total)
            out.write("\t".join(str(x) for x in [sample, chrom, 'Homozygosity',
                                             samp_chrom_hom, samp_chrom_total,
                                             frac, samp_vs_chrom,
                                             samp_vs_self]) + "\n")
            if sample in children:
                samp_informative_total = np.sum(np.sum(genomewide_counts[3:,i]))
                samp_chrom_informative = np.sum(gt_counts.counts[chrom][3:,i])
                samp_upd_totals = dict((k, np.sum(genome_all_upd[k][i])) for k
                                        in upd_types)
                samp_upd_chrom = dict((k,
                                       chrom_upd[k][i])
                                       for k in upd_types)
                for utype in upd_types:
                    samp_vs_chrom = "NA"
                    samp_vs_genome = "NA"
                    samp_vs_self = "NA"
                    if chrom_informative_total > 0:
                        samp_vs_chrom = scipy.stats.binom_test(
                                samp_upd_chrom[utype],
                                samp_chrom_informative,
                                np.sum(chrom_upd[utype])/chrom_informative_total,
                                alternative='greater')
                        samp_vs_self = scipy.stats.binom_test(
                                samp_upd_chrom[utype],
                                samp_chrom_informative,
                                samp_upd_totals[utype]/samp_informative_total,
                                alternative='greater')
                    frac = "{:.3g}".format(
                            samp_upd_chrom[utype]/samp_chrom_informative)
                    out.write("\t".join(str(x) for x in [sample, chrom, utype,
                                                     samp_upd_chrom[utype],
                                                     samp_chrom_informative,
                                                     frac, samp_vs_chrom,
                                                     samp_vs_self]) + "\n")

def get_upd_counts(counts, gt_rows):
    '''
        Returns a dict to counts of total paternal UPD (pat_upd),
        total maternal UPD (mat_upd) and paternal isodisomic UPD
        (pat_i_upd) and maternal isodisomic UPD (mat_i_upd).
        Args:
            counts: 2d-array of gt-rows x N-samples.

            gt_rows:
                    Dict of genotype IDs to row index.
    '''
    mat_upd = counts[gt_rows['mat_i_upd']] + counts[gt_rows['mat_a_upd']]
    pat_upd = counts[gt_rows['pat_i_upd']] + counts[gt_rows['pat_a_upd']]
    return {'mat_upd': mat_upd, 'mat_i_upd': counts[gt_rows['mat_i_upd']],
            'pat_upd': pat_upd, 'pat_i_upd': counts[gt_rows['pat_i_upd']]}

def get_seq_ids(vcf):
    if vcf.endswith(".bcf"):
        idx = vcf + '.csi'
        preset = 'bcf'
    else:
        idx = vcf + '.tbi'
        preset = 'vcf'
    if not os.path.isfile(idx):   #create index if it doesn't exist
        pysam.tabix_index(vcf, preset=preset)
    if preset == 'bcf':
        vf = pysam.VariantFile(vcf)
        return (c for c in vf.index.keys() if chrom_re.match(c))
    else:
        tbx = pysam.TabixFile(vcf)
        return (c for c in tbx.contigs if chrom_re.match(c))

def get_logger(loglevel=logging.INFO, logfile=None):
    logger = logging.getLogger("UPD")
    logger.setLevel(loglevel)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
           '[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logger.level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def initialize_mp_logger(logger, loglevel, logfile=None):
    logger.setLevel(loglevel)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
           '[%(asctime)s] UPD-%(processName)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logger.level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

def get_gt_counts(vcf, samples, coordinate_table=None, parents=None,
                  children=set(), females= set(), contig=None, 
                  prog_interval=10000, logger=None, loglevel=logging.INFO):
    vreader = VcfReader(vcf)
    table_fn = None
    if coordinate_table:
        fh, table_fn = tempfile.mkstemp()
        os.close(fh)
        gzf = gzip.open(table_fn, 'wt')
    if logger is None:
        logger = mp.get_logger()
        if logger.level == logging.NOTSET:
            initialize_mp_logger(logger, loglevel)
    if contig is not None:
        vreader.set_region(contig)
        logger.info("Reading chromosome {}".format(contig))
    else:
        logger.info("Parsing variants")
    gt_fields = ['GT', 'GQ', 'DP', 'AD']
    gt_counter = GtCounter(samples)
    n = 0
    valid = 0
    for record in vreader:
        n += 1
        if n % prog_interval == 0:
            logger.info("Read {:,} records, processed {:,},".format(n, valid) +
                        " {:,} filtered at {}:{}".format(n - valid,
                                                         record.CHROM,
                                                         record.POS))
        if len(record.ALLELES) != 2: #biallelic only
            continue
        if len(record.REF) != 1 or len(record.ALT) != 1: #SNVs only
            continue
        if contig is None and not chrom_re.match(record.CHROM):
            continue
        if record.FILTER != 'PASS':
            continue
        valid += 1
        gts = record.parsed_gts(fields=gt_fields, samples=samples)
        for s in samples:
            #filter on GQ/DP
            if not gt_passes_filters(gts, s):
                continue
            if gts['GT'][s] == (0, 1) or gts['GT'][s] == (1, 0):
                sgt = 'het'
            elif gts['GT'][s] == (1, 1):
                sgt = 'hom_alt'
            elif gts['GT'][s] == (0, 0):
                sgt = 'hom_ref'
            else:
                continue
            if 'X' in record.CHROM and s not in females:
                continue
            gt_counter.count_genotype(record.CHROM, s, sgt)
            if s in children:
                disomy = upd(gts, s, parents[s]['mother'],
                             parents[s]['father'])
                if disomy:
                    gt_counter.count_genotype(record.CHROM, s, disomy)
                    if table_fn:
                        gzf.write("\t".join(str(x) for x in [record.CHROM,
                                                            record.POS, s,
                                                            disomy]) + "\n")

    if contig is not None:
        logger.info("Finished processing chromosome {}".format(contig))
    if table_fn:
        gzf.close()
    return gt_counter, table_fn

def samples_from_ped(ped, vcf, logger):
    parents = defaultdict(dict)
    samples = []
    children = []
    females = []
    pedfile = PedFile(ped)
    vreader = VcfReader(vcf)
    for iid, indv in pedfile.individuals.items():
        if iid not in vreader.header.samples:
            logger.warn("Skipping individual {} - not in VCF".format(iid))
            continue
        samples.append(iid)
        if indv.is_female():
            females.append(iid)
        for p in ['mother', 'father']:
            parents[iid][p] = (getattr(indv, p) if getattr(indv, p) in
                               vreader.header.samples else None)
    #only trios currently supported, but could tweak to be able to parse
    #parent-child duos
    children = set(x for x in parents if parents[x]['mother'] is not None and
                   parents[x]['father'] is not None)
    logger.info("Got {} parent-child trios from PED".format(len(children)))
    return samples, parents, children, females

def gt_passes_filters(gts, s, min_gq=20, min_dp=10, min_ab=0.25):
    if gts['GQ'][s] is None or gts['GQ'][s] < min_gq:
        return False
    if gts['DP'][s] is None or gts['DP'][s] < min_dp:
        return False
    if gts['GT'][s] == (0, 1):
        #filter hets on AB
        dp = sum(gts['AD'][s])
        ad = gts['AD'][s][1]
        if ad is not None and dp > 0:
            ab = ad/dp
            if ab < min_ab:
                return False
    return True

def upd(gts, sample, mother, father):
    if mother is not None and not gt_passes_filters(gts, mother):
        mother = None
    if father is not None and not gt_passes_filters(gts, father):
        father = None
    #TODO - implement detection for single parent families
    if mother is None or father is None:
        return False
    #1/1 Dad + 0/0 Mum + 0/1 Child = BPI
    #0/0 Dad + 1/1 Mum + 0/1 Child = BPI
    if gts['GT'][sample] == (0, 1) or gts['GT'][sample] == (1, 0):
        if gts['GT'][mother] == (0, 0) and gts['GT'][father] == (1, 1):
            return 'bpi'
        if gts['GT'][mother] == (1, 1) and gts['GT'][father] == (0, 0):
            return 'bpi'
        else:
            return False #uninformative
    #0/0 Dad + 0/1 Mum + 1/1 Child = mIUPD
    #1/1 Dad + 0/1 Mum + 0/0 Child = mIUPD
    #0/0 Dad + 1/1 Mum + 1/1 Child = mhiUPD
    #1/1 Dad + 0/0 Mum + 0/0 Child = mhiUPD
    if gts['GT'][sample] == (0, 0):
        inherited,other = 0,1
    elif gts['GT'][sample] == (1, 1):
        inherited ,other = 1,0
    else:
        return False
    if gts['GT'][father] == (other, other):
        if gts['GT'][mother] == (inherited, inherited):
            return 'mat_a_upd'
        if gts['GT'][mother] == (0, 1) or gts['GT'][mother] == (1, 0):
            return 'mat_i_upd'
    elif gts['GT'][mother] == (other, other):
        if gts['GT'][father] == (inherited, inherited):
            return 'pat_a_upd'
        if gts['GT'][father] == (0, 1) or gts['GT'][father] == (1, 0):
            return 'pat_i_upd'
    return False #uninformative

def get_parser():
    parser = argparse.ArgumentParser(
                  description='Identify putative UPD chromosomes in samples.')
    required_args = parser.add_argument_group('Required Arguments')
    optional_args = parser.add_argument_group('Optional Arguments')
    required_args.add_argument("-i", "--vcf", "--input", required=True,
                               help="Input VCF file.")
    optional_args.add_argument("-p", "--ped", help='''PED file detailing any
                               familial relationships for samples in VCF. If
                               no PED file is provided samples will be analyzed
                               for excess homozygosity only.''')
    optional_args.add_argument("-o", "--output", help='''Write summary output
                               data to this file. Defaults to STDOUT.''')
    optional_args.add_argument("-c", "--coordinate_table", help='''Output file
                               for table of sample coordinates and UPD/non-UPD
                               sites. Required if you want to create plots
                               after running this program.''')
    optional_args.add_argument("-t", "--threads", type=int, default=1,
                               help='''Number of threads to use. Default=1.
                               Maximum will be determined by the number of
                               chromosomes in your reference.''')
    optional_args.add_argument("--progress_interval", type=int, default=10000,
                               metavar='N', help='''Report progress every N
                               variants.''')
    return parser

if __name__ == '__main__':
    argparser = get_parser()
    args = argparser.parse_args()
    main(**vars(args))
