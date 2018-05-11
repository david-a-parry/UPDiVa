#!/usr/bin/env python3
import sys
import os
import re
import argparse
import logging
import pysam
import multiprocessing as mp
import numpy as np
import scipy.stats
from itertools import repeat
from collections import defaultdict
from copy import copy
from parse_vcf import VcfReader
from vase.ped_file import PedFile

chrom_re = re.compile(r'''^(chr)?[1-9X][0-9]?$''')
gt_ids = ['het', 'hom_ref', 'hom_alt', 'mat_upd', 'pat_upd']

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

def main(vcf, ped=None, threads=1, progress_interval=10000):
    if threads > 1:
        get_args = {'vcf': vcf, 'ped': ped, 'prog_interval': progress_interval}
        contig_args = ({'contig': x} for x in get_seq_ids(vcf))
        if not contig_args:
            raise RuntimeError("No valid contigs identified in {}".format(fai))
        with mp.Pool(threads) as p:
            results = p.map(_process_runner, zip(contig_args, repeat(get_args)))
        logger = get_logger()
    else:
        logger = get_logger()
        results = [get_gt_counts(vcf=vcf, ped=ped, logger=logger,
                                 prog_interval=progress_interval)]
    logger.info("Collating and parsing results.")
    parse_results(sum(results), ped)

def parse_results(gt_counts, count_upd=True):
    #TODO - address chrX in females
    genomewide_counts = sum(gt_counts.counts[c] for c in gt_counts.counts if
                            "X" not in c)
    gt_indices = dict((r, n) for n,r in enumerate(gt_ids))
    print("\t".join(["Sample", "Chrom", "GT", "N", "Calls", "vs_chrom",
                     "vs_genome", "vs_self"]))
    logger.info("Calculating genomewide total")
    genome_total = np.sum(genomewide_counts[:3])
    for chrom in gt_counts.counts:
        if "X" in chrom:
            continue
        #only het/hom-ref/hom-alt - don't want to count UPD as these are
        #included in hom calls
        chrom_total = np.sum(gt_counts.counts[chrom][:3])
        for gt in gt_ids:
            if not count_upd and gt.endswith("_upd"):
                continue
            logger.info("Parsing {} results for chomosome {}".format(gt,chrom))
            genome_count = np.sum(genomewide_counts[gt_indices[gt]])
            chrom_count = np.sum(gt_counts.counts[chrom][gt_indices[gt]])
            for sample in gt_counts.samples:
                i = gt_counts.samp_indices[sample]
                samp_totals = np.sum(x[:,i] for x in gt_counts.counts.values())
                samp_genomewide_total = np.sum(samp_totals[:3])
                samp_chrom_total = np.sum(gt_counts.counts[chrom][:3,i])
                samp_count = gt_counts.counts[chrom][gt_indices[gt],i]
                samp_vs_chrom = scipy.stats.binom_test(samp_count,
                                                       samp_chrom_total,
                                                       chrom_count/chrom_total,
                                                       alternative='greater')
                samp_vs_genome = scipy.stats.binom_test(
                        samp_count,
                        samp_genomewide_total,
                        genome_count/genome_total,
                        alternative='greater')
                samp_vs_self = scipy.stats.binom_test(
                            samp_count,
                            samp_chrom_total,
                            samp_totals[gt_indices[gt]]/samp_genomewide_total,
                            alternative='greater')
                print("\t".join(str(x) for x in [sample, chrom, gt, samp_count,
                                                 samp_chrom_total,
                                                 samp_vs_chrom, samp_vs_genome,
                                                 samp_vs_self]))

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

def get_gt_counts(vcf, ped=None, contig=None, prog_interval=10000,
                  logger=None, loglevel=logging.INFO):
    vreader = VcfReader(vcf)
    parents = defaultdict(dict)
    samples = []
    if logger is None:
        logger = mp.get_logger()
        if logger.level == logging.NOTSET:
            initialize_mp_logger(logger, loglevel)
    if ped is not None:
        logger.info("Parsing PED file.")
        pedfile = PedFile(ped)
        for iid, indv in pedfile.individuals.items():
            if iid not in vreader.header.samples:
                logger.warn("Skipping individual {} - not in VCF".format(iid))
                continue
            samples.append(iid)
            for p in ['mother', 'father']:
                parents[iid][p] = (getattr(indv, p) if getattr(indv, p) in
                                   vreader.header.samples else None)
    else:
        logger.info("No PED file - analyzing all samples for homozygosity " +
                    "only")
        samples = vreader.header.samples
        for s in samples:
            parents[s]['mother'] = None
            parents[s]['father'] = None
    if contig is not None:
        vreader.set_region(contig)
        logger.info("Reading chromosome {}".format(contig))
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
            homozygous = False
            if gts['GT'][s] == (0, 1):
                sgt = 'het'
            elif gts['GT'][s] == (1, 1):
                sgt = 'hom_alt'
                homozygous = True
            elif gts['GT'][s] == (0, 0):
                sgt = 'hom_ref'
                homozygous = True
            else:
                continue
            gt_counter.count_genotype(record.CHROM, s, sgt)
            if (homozygous):
                disomy = upd(gts, s, parents[s]['mother'], parents[s]['father'])
                if disomy:
                    gt_counter.count_genotype(record.CHROM, s, disomy)
    if contig is not None:
        logger.info("Finished processing chromosome {}".format(contig))
    return gt_counter

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
    for comb in [(0, 1), (1, 0)]:
        if gts['GT'][sample] == (comb[0], comb[0]):
            if gts['GT'][father] == (comb[1], comb[1]):
                if comb[0] in gts['GT'][mother]:
                    return 'mat_upd'
            elif gts['GT'][mother] == (comb[1], comb[1]):
                if comb[0] in gts['GT'][father]:
                    return 'pat_upd'
    return False

def get_parser():
    parser = argparse.ArgumentParser(
                  description='Identify putative UPD chromosomes in samples.')
    parser.add_argument("-i", "--vcf", "--input", required=True,
                        help="Input VCF file.")
    parser.add_argument("-p", "--ped", help='''PED file detailing any familial
                        relationships for samples in VCF.''')
    parser.add_argument("-t", "--threads", type=int, default=1, help='''Number
                         of threads to use. Default=1. Maximum will be
                        determined by the number of chromosomes in your
                        reference.''')
    parser.add_argument("--progress_interval", type=int, default=10000,
                        metavar='N', help='''Report progress every N
                        variants.''')
    return parser

if __name__ == '__main__':
    argparser = get_parser()
    args = argparser.parse_args()
    main(**vars(args))
