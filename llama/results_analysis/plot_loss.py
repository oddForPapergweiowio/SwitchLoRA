"""

Examples:
"""

import matplotlib.pyplot as plt
import json
import sys
import numpy as np
import os
import argparse
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True


def set_font_size():
    SMALL_SIZE = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 24

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


set_font_size()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ylim", type=str, default=None,
                        help="Example usage: --ylim 0.1,3 (Do not add space around comma)")
    parser.add_argument("--to_plot", action='store_true')
    parser.add_argument("--no_save", action='store_true')
    args, unknownargs = parser.parse_known_args()

    if args.ylim is not None:
        ylim_bottom, ylim_top = args.ylim.split(',')
        args.ylim = {}
        if ylim_top != "":
            args.ylim["top"] = float(ylim_top)
        if ylim_bottom != "":
            args.ylim["bottom"] = float(ylim_bottom)
    else:
        args.ylim = {}
    return args, unknownargs


def read_test_loss_data(data_path, to_print=False):
    merged_test_loss = {}
    for dirname in os.listdir(data_path):
        model_path = os.path.join(data_path, dirname)
        if not dirname.startswith("model_") or not os.path.isdir(model_path):
            continue
        json_file = os.path.join(model_path, "loss.json")
        if not os.path.exists(json_file):
            continue
        with open(json_file, 'r') as f:
            data = json.load(f)
            test_loss_pairs = data["test_loss"]

            for step, test_loss in test_loss_pairs:
                if step in merged_test_loss and merged_test_loss[step] != test_loss:
                    raise RuntimeError("Data in " + model_path + " conflicts at step " + step + ".")
                merged_test_loss[step] = test_loss

    if to_print:
        print(merged_test_loss)
    steps = list(merged_test_loss.keys())
    steps = [step for step in steps if step % 1000 == 0]
    test_losses = [merged_test_loss[step] for step in steps]
    return steps, test_losses


def add_value_mark(interval):
    for i in range(10):
        value = interval * i
        plt.axhline(y=value, color='red', linestyle='--', linewidth=0.8)


def plot(results_path, labels, title, args, save_path=None, to_plot=False):
    plt.xlabel("Step")
    plt.ylabel("Loss")

    # Set viewport
    min_step = -1
    for i, path in enumerate(results_path):
        steps, test_losses = read_test_loss_data(path)
        if min_step < 0:
            min_step = max(steps)
        if min_step > max(steps):
            min_step = max(steps)
    print(min_step)

    for i, path in enumerate(results_path):
        label = labels[i]
        steps, test_losses = read_test_loss_data(path)
        plt.plot(steps, test_losses, label=label)

    # In some experiments, Step 38000 and 39000 are missing due to lack of data.
    # So we only plot up to 38000.
    if min_step == 38000 or min_step == 39000 or min_step == 40000:
        min_step = 38000
        plt.xticks(np.append(plt.xticks()[0], min_step))
    plt.xlim(0, min_step)
    plt.ylim(**args.ylim)

    plt.grid(True)
    plt.title(title)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    if to_plot:
        plt.show()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        exit(1)

    args, unknownargs = parse_arguments()
    save_path = unknownargs[0]
    title = unknownargs[1]
    labels = unknownargs[2::2]
    results_path = unknownargs[3::2]
    if args.no_save:
        save_path = None
    plot(results_path, labels, title, args, save_path=save_path, to_plot=args.to_plot)
