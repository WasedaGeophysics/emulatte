from shutil import which
import numpy as np
import matplotlib.pyplot as plt

def fdem_plot(freq, data):
    fc = "#000309"
    tc = "#eee"
    fig = plt.figure(figsize=(6,4), facecolor=fc, dpi=200)
    ax = fig.add_subplot(111)
    cc = ["#5bf", "#07b"]

    ymax = max(data.real.max(), (-data.real).max(), data.imag.max(), (-data.imag).max())

    ylim_max = 10 ** (np.floor(np.log10(ymax)) + 1)
    ylim_min = ylim_max * 1e-6

    ax.loglog(freq, data.real, c=cc[0], linestyle="-")
    ax.loglog(freq, -data.real, c=cc[0], linestyle=":")
    ax.loglog(freq, data.imag, c=cc[1], linestyle="-")
    ax.loglog(freq, -data.imag, c=cc[1], linestyle=":")

    ax.set_xlim(freq.min(), freq.max())
    ax.set_ylim(ylim_min, ylim_max)

    ax.set_facecolor(fc)

    ax.grid(which="major", color="#555", linewidth=0.5)
    ax.grid(which="minor", color="#333", linestyle="--", linewidth=0.5)

    # 枠の色
    ax.spines['left'].set_color(tc)
    ax.spines['bottom'].set_color(tc)
    # ラベルの色
    ax.xaxis.label.set_color(tc)
    ax.yaxis.label.set_color(tc)
    # 目盛りの色
    ax.tick_params(axis='x', colors=tc)
    ax.tick_params(axis='y', colors=tc)
    return fig