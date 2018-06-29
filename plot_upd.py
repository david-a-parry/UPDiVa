#!/usr/bin/env python3
import sys
import os
import argparse
import matplotlib
if os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg') #for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import bisect
from collections import defaultdict

valid_states = ['mat_i_upd', 'mat_a_upd', 'pat_i_upd', 'pat_a_upd', 'bpi',]

lbls = {'mat_i_upd': 'Maternal isodisomic UPD',
        'mat_a_upd': 'Maternal total UPD',
        'pat_i_upd': 'Paternal isodisomic UPD',
        'pat_a_upd': 'Paternal total UPD',
        'bpi'      :  'Bi-parental'}

def states_per_region(df, chrom, sample, w=1000000 ):
    counts = defaultdict(list)
    chrom_df = df[(df.chrom == chrom) & (df.Sample == sample)]
    if len(chrom_df) < 1:
        return
    i = chrom_df.iloc[0].pos
    while i < chrom_df.iloc[-1].pos - w:
        window = chrom_df[(chrom_df.pos >= i) & (chrom_df.pos <= i + w)]
        i += w
        if len(window) == 0:
            continue
        for state in valid_states:
            if state == 'mat_a_upd':
                f = len(window[(window.state == state) |
                               (window.state == 'mat_i_upd')])/len(window)
            elif state == 'pat_a_upd':
                f = len(window[(window.state == state) |
                               (window.state == 'pat_i_upd')])/len(window)
            else:
                f = len(window[window.state == state])/len(window)
            counts['State'].append(state)
            counts['Frac'].append(f)
            counts['Pos'].append((window.pos.min() + window.pos.max())/2e6)
            counts['Calls'].append(len(window))
    return pd.DataFrame(counts)

def add_highlight_to_plot(plt, region, color='black', pivot_table=None):
    start = region[0]
    end = region[1]
    label = None
    if len(region) > 2:
        label = region[2]
    if pivot_table is not None:
        fpos = pivot_table.iloc[0]
        l = bisect.bisect_left(fpos.keys(), float(start)/1e6)
        r = bisect.bisect_left(fpos.keys(), float(end)/1e6)
        plt.plot([l, l], [0, 5], '--', color=color, alpha=0.5)
        plt.plot([r, r], [0, 5], '--', color=color, alpha=0.5)
    else:
        plt.plot([float(start)/1e6, float(start)/1e6], [0, 1], '--',
                 color=color, label=label, alpha=0.5)
        plt.plot([float(end)/1e6, float(end)/1e6], [0, 1], '--', color=color,
                 alpha=0.5)

def plot_upd(df, sample, chrom, out_dir, w=1000000, fig_dimensions=(12, 6),
             marker=None, centromeres=dict(), roi=dict()):
    sys.stderr.write("Processing sample {} chromosome {}...\n".format(sample,
                                                                      chrom))
    states = states_per_region(df, sample=sample, chrom=chrom, w=w)
    if states is None:
        sys.stderr.write("Warning: No data for sample {} ".format(sample) +
                         "chromosome {}".format(chrom))
        return
    colors = sns.color_palette("Paired", len(valid_states))
    fig = plt.figure(figsize=fig_dimensions)
    suptitle = fig.suptitle("{} {}".format(sample, chrom))
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)
    fig.add_subplot(grid[0, 1],)
    plt.plot(states.Pos, states.Calls, color='red', marker=marker,
             label="Calls per {:g} bp".format(w))
    plt.title("Calls per Window")
    plt.ylabel("Calls")
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig.add_subplot(grid[1, 1],)
    for i in range(len(valid_states)):
        s = states[states.State == valid_states[i]]
        plt.plot(s.Pos, s.Frac, color=colors[i],
                 label=lbls[valid_states[i]],  marker=marker)
    if chrom in centromeres:
        add_highlight_to_plot(plt, centromeres[chrom], color='black')
    if chrom in roi and len(roi[chrom]) > 0:
        pal = sns.color_palette("Set2", len(roi[chrom]))
        for region, col in zip(roi[chrom], pal):
            add_highlight_to_plot(plt, region, color=col)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title("States")
    plt.xlabel("Pos (Mb)")
    plt.ylabel("Fraction")
    pivot = states.pivot("State", "Pos", "Calls")
    fig.add_subplot(grid[0, 0],)
    ax = sns.heatmap(pivot)
    ax.set_title("Calls per Window")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_subplot(grid[1, 0],)
    pivot = states.pivot("State", "Pos", "Frac")
    if chrom in centromeres:
        add_highlight_to_plot(plt, centromeres[chrom], color='white',
                              pivot_table=pivot)
    if chrom in roi and len(roi[chrom]) > 0:
        pal = sns.color_palette("Set2", len(roi[chrom]))
        for region, col in zip(roi[chrom], pal):
            add_highlight_to_plot(plt, region, color=col, pivot_table=pivot)
    ax = sns.heatmap(pivot)
    ax.set_title("States")
    xticklabels = []
    for it in ax.get_xticklabels():
        it.set_text('{:0.2f}'.format(float(it.get_text())))
        xticklabels += [it]
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Pos (Mb)")
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2,
                    borderaxespad=0.)
    fig.savefig(os.path.join(out_dir, "{}_{}.png".format(chrom, sample)),
                bbox_extra_artists=(lgd, suptitle), bbox_inches='tight')
    plt.cla()
    plt.close('all')

def data_to_plots(sample_file, pos_file, out_dir, min_fraction, window_size,
                  sig_self, sig_chrom, samples, chromosomes, plot_all=False,
                  marker=None, roi=dict(), centromeres=dict()):
    sys.stderr.write("Reading data...\n")
    pos_state = pd.read_table(pos_file, names=["chrom", "pos", "Sample",
                                               "state"])
    if plot_all:
        samples = list(pos_state.Sample.unique())
        chromosomes = list(pos_state.chrom.unique())
    sample_upd = pd.read_table(sample_file)
    if not samples and not chromosomes: #do all 'significant' sample vs chroms
        sigs = sample_upd[(sample_upd['Test'] != 'Homozygosity') &
                          (sample_upd['vs_chrom'] < sig_self) &
                          (sample_upd['vs_self'] < sig_chrom) &
                          (sample_upd['Fraction'] > min_fraction)]
        for s in sigs.Sample.unique():
            for c in sigs[sigs.Sample == s].Chrom.unique():
                plot_upd(pos_state, sample=s, chrom=c, out_dir=out_dir,
                         marker=marker, w=window_size, centromeres=centromeres,
                         roi=roi)
    else:
        if not samples:
            samples = pos_state.Sample.unique()
        if not chromosomes:
            chromosomes = pos_state.chrom.unique()
        for s in samples:
            for c in chromosomes:
                plot_upd(pos_state, sample=s, chrom=c, out_dir=out_dir,
                         marker=marker, w=window_size, centromeres=centromeres,
                         roi=roi)

def read_roi_bed(bedfile):
    roi = defaultdict(list)
    with open(bedfile, 'rt') as infile:
        for line in infile:
            cols = line.split()
            if len(cols) < 3:
                sys.exit("Not enough columns in bed for {}".format(bedfile))
            region = [int(cols[1]) + 1, int(cols[2])]
            if len(cols) > 3:
                region.append(cols[3])
            roi[cols[0]].append(region)
    return roi

def read_centromere_bed(bedfile):
    centromeres = dict()
    with open(bedfile, 'rt') as infile:
        for line in infile:
            cols = line.split()
            if len(cols) < 3:
                sys.exit("Not enough columns in bed for {}".format(bedfile))
            centromeres[cols[0]] = (int(cols[1]) + 1, int(cols[2]))
    return centromeres

def main(sample_file, coordinate_table, output_directory, min_fraction=0.05,
         window_size=1e6, self_significance_cutoff=1e-5,
         chrom_significance_cutoff=1e-5, samples=[], chromosomes=[],
         build=None, roi=None, plot_all=False, context='talk', marker=None,
         force=False):
    sns.set_context(context)
    sns.set_style('darkgrid')
    centromeres = dict()
    if build is not None:
        cfile = os.path.join(os.path.dirname(__file__), "data",
                             build + "_centromeres.bed")
        if os.path.exists(cfile):
            centromeres = read_centromere_bed(cfile)
        else:
            sys.exit("No centromere file ({}) for build {}".format(cfile,
                                                                   build))
    highlight_regions = dict()
    if roi is not None:
        highlight_regions = read_roi_bed(roi)
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    elif not force:
        sys.exit("Output directory '{}' exists ".format(output_directory) +
                 "- use --force to overwrite")
    data_to_plots(sample_file=sample_file, pos_file=coordinate_table,
                  out_dir=output_directory, min_fraction=min_fraction,
                  window_size=window_size, sig_self=self_significance_cutoff,
                  sig_chrom=chrom_significance_cutoff, samples=samples,
                  chromosomes=chromosomes, marker=marker, plot_all=plot_all,
                  roi=highlight_regions, centromeres=centromeres)

def get_parser():
    parser = argparse.ArgumentParser(
                  description='Plot UPD results.')
    parser.add_argument("-i", "--sample_file", "--input", required=True,
                        help="Table of UPD results produced by upd.py.")
    parser.add_argument("-c", "--coordinate_table", required=True,
						help="Coordinate table produced by upd.py")
    parser.add_argument("-o", "--output_directory", default='upd_plots',
                        help='''Directory to place plot PNG files.
                                Default='upd_plots'.''')
    parser.add_argument("-f", "--min_fraction", type=float, default=0.05,
                        help='''Minimum fraction of chromosome that must be UPD
                        in order to plot. Default=0.05''')
    parser.add_argument("-w", "--window_size", type=float, default=1e6,
                        help='''Windows size to use when calculating ratios of
                        UPD vs biparental inheritance fractions.
                        Default=1e6.''')
    parser.add_argument("-s", "--self_significance_cutoff", type=float,
						default=1e-5, help='''Only plot for samples and
						chromosomes where the p-value vs other chromosomes for
						the same sample is below this threshold.
						Default=1e-5.''')
    parser.add_argument("-x", "--chrom_significance_cutoff", type=float,
						default=1e-5, help='''Only plot for samples and
						chromosomes where the p-value vs the same chromosome in
						other samples is below this threshold.
						Default=1e-5.''')
    parser.add_argument("--samples", nargs='+', help='''Plot these samples
                        (only) irrespective of significance thresholds.''')
    parser.add_argument("--chromosomes", nargs='+', help='''Plot these
                        chromosomes (only) irrespective of significance
                        thresholds.''')
    parser.add_argument("--plot_all", action='store_true', help='''Plot all
                        chromosomes for all samples irrespective of
                        significance thresholds.''')
    parser.add_argument("--context", default='talk',  help='''Seaborn context
                        type (e.g. 'paper', 'talk', 'poster', 'notebook') for
                        setting the plotting context. Default=talk''')
    parser.add_argument("-m", "--marker", help='''Use this marker style for
                        individual points in plots. Default=None (no markers).
                        Valid values are as per matplotlib.''')
    parser.add_argument("-b", "--build", help='''Genome build. If specified,
                        centromeres will be marked if there is a corresponding
                        BED file of centromere locations in the 'data'
                        subdirectory (hg19, hg38, GRCh37, GRCh38 available by
                        default).''')
    parser.add_argument("-r", "--roi", help='''BED file of regions of interest.
                        If a 4th column is present this will be used to label
                        these regions.''')
    parser.add_argument("--force", action='store_true', help='''Overwrite
                        existing output directories.''')
    return parser

if __name__ == '__main__':
    argparser = get_parser()
    args = argparser.parse_args()
    main(**vars(args))
