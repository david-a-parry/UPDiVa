#!/usr/bin/env python3
import sys
import os
import argparse
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
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

def plot_upd(df, sample, chrom, out_dir, w=1000000, fig_dimensions=(12, 6),
             marker=None):
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
                  sig_self, sig_chrom, samples, chromosomes, marker=None):
    sys.stderr.write("Reading data...\n")
    pos_state = pd.read_table(pos_file, names=["chrom", "pos", "Sample",
                                               "state"])
    sample_upd = pd.read_table(sample_file)
    if not samples and not chromosomes: #do all 'significant' sample vs chroms
        sigs = sample_upd[(sample_upd['Test'] != 'Homozygosity') &
                          (sample_upd['vs_chrom'] < sig_self) &
                          (sample_upd['vs_self'] < sig_chrom) &
                          (sample_upd['Fraction'] > min_fraction)]
        for s in sigs.Sample.unique():
            for c in sigs[sigs.Sample == s].Chrom.unique():
                plot_upd(pos_state, sample=s, chrom=c, out_dir=out_dir,
                         marker=marker, w=window_size)
    else:
        if not samples:
            samples = pos_state.Sample.unique()
        if not chromosomes:
            chromosomes = pos_state.chrom.unique()
        for s in samples:
            for c in chromosomes:
                plot_upd(pos_state, sample=s, chrom=c, out_dir=out_dir,
                         marker=marker, w=window_size)

def main(sample_file, coordinate_table, output_directory, min_fraction=0.05,
         window_size=1e6, self_significance_cutoff=1e-5,
         chrom_significance_cutoff=1e-5, samples=[], chromosomes=[],
         context='talk', marker=None, force=False):
    sns.set_context(context)
    sns.set_style('darkgrid')
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    elif not force:
        sys.exit("Output directory '{}' exists ".format(output_directory) +
                 "- use --force to overwrite")
    data_to_plots(sample_file, coordinate_table, output_directory,
                  min_fraction, window_size, self_significance_cutoff,
                  chrom_significance_cutoff, samples, chromosomes, marker)

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
    parser.add_argument("--context", default='talk',  help='''Seaborn context
                        type (e.g. 'paper', 'talk', 'poster', 'notebook') for
                        setting the plotting context. Default=talk''')
    parser.add_argument("-m", "--marker", help='''Use this marker style for
                        individual points in plots. Default=None (no markers).
                        Valid values are as per matplotlib.''')
    parser.add_argument("--force", action='store_true', help='''Overwrite
                        existing output directories.''')
    return parser

if __name__ == '__main__':
    argparser = get_parser()
    args = argparser.parse_args()
    main(**vars(args))
