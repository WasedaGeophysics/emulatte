from shutil import which
import numpy as np
import matplotlib.pyplot as plt

def fdem_plot(freq, data, data2=None):
    fc = "#fff"
    tc = "#555"
    fig = plt.figure(figsize=(6,4), facecolor=fc, dpi=200)
    ax = fig.add_subplot(111)
    cc = ["#5bf", "#06a", "#fb5", "#b36"]

    ax.loglog(freq, data.real, c=cc[0], linestyle="-")
    ax.loglog(freq, -data.real, c=cc[0], linestyle=":")
    ax.loglog(freq, data.imag, c=cc[1], linestyle="-")
    ax.loglog(freq, -data.imag, c=cc[1], linestyle=":")

    if not data2 is None:
        ax.loglog(freq, data2.real, c=cc[2], linestyle="--")
        ax.loglog(freq, -data2.real, c=cc[2], linestyle=":")
        ax.loglog(freq, data2.imag, c=cc[3], linestyle="--")
        ax.loglog(freq, -data2.imag, c=cc[3], linestyle=":")

    ax.set_xlim(freq.min(), freq.max())
    #ax.set_ylim(ylim_min, ylim_max)

    ax.set_facecolor(fc)

    ax.grid(which="major", color="#ccc", linewidth=0.5)
    ax.grid(which="minor", color="#eee", linestyle="--", linewidth=0.5)

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

def fdem_rel_error_plot(freq, data1, data2):
    re_real = np.abs((data1.real-data2.real)/data2.real) * 100
    re_imag = np.abs((data1.imag-data2.imag)/data2.imag) * 100

    fc = "#000309"
    tc = "#eee"
    fig = plt.figure(figsize=(6,4), facecolor=fc, dpi=200)
    ax = fig.add_subplot(111)
    cc = ["#5bf", "#07b"]

    ax.loglog(freq, re_real, c=cc[0], linestyle="-")
    ax.loglog(freq, re_imag, c=cc[1], linestyle="-")

    ax.set_xlim(freq.min(), freq.max())
    #ax.set_ylim(ylim_min, ylim_max)

    ax.set_facecolor(fc)

    ax.grid(which="major", color="#ccc", linewidth=0.5)
    ax.grid(which="minor", color="#eee", linestyle="--", linewidth=0.5)

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