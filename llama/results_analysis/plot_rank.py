import json
import sys
import matplotlib.pyplot as plt
import os
import numpy as np


def set_font_size():
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


set_font_size()


def plot_across_types(json_paths, labels, to_plot=False, save_path=None):
    """
    """
    all_data = []
    for json_path in json_paths:
        with open(json_path, 'r') as f:
            data = json.load(f)
        all_data.append(data)

    types = {'q_projs': "Q", 'k_projs': "K", 'v_projs': "V", 'o_projs': "Out", 'gate_projs': "Gate",
             'down_projs': "Down Linear", 'up_projs': "Up Linear"}

    def plot_fig(ranks, rank_name, data_name):
        plot_label = data_name
        plt.hist([r for r_list in ranks for r in r_list], density=True, bins=100, alpha=0.7,
                 label=plot_label)

    for type in types:
        for i, data in enumerate(all_data):
            plot_fig(data[type], type, labels[i])
        # Labeling
        plt.xlabel("Singular values")
        plt.ylabel("Density")
        plt.title("Rank Distribution of " + types[type] + " Matrix")

        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
        if to_plot:
            plt.show()
        if save_path is not None:
            plt.savefig(save_path + type, bbox_inches='tight')
        plt.clf()

    # output_path = sys.argv[1] + '/rank_dist_types.png'
    # plt.savefig(output_path, bbox_inches='tight')  # Saves the figure

    # Show the plot
    # plt.show()


if __name__ == '__main__':
    # Check if the script has the necessary command line argument
    if len(sys.argv) < 2:
        print("Usage:")
        sys.exit(1)

    save_path = sys.argv[1]
    labels = sys.argv[2::2]
    json_paths = sys.argv[3::2]
    plot_across_types(json_paths, labels=labels, save_path=save_path)
