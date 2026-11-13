"""
Generate the illustrative figures/animations used in the README.

    python docs/make_figures.py

Every figure here is an *explanatory* illustration of the signal model
(modulo folding, wrap cancellation, block search). None of them reproduce
the paper's experimental results.

Requires: numpy, matplotlib, pillow
"""
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#1f5fbf"
RED = "#d62728"
GREEN = "#2ca02c"
GREY = "#8a8f98"
INK = "#1d2330"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#c9ced6",
    "axes.labelcolor": INK,
    "xtick.color": "#4a5263",
    "ytick.color": "#4a5263",
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
})


def fold(x, lam):
    """Centred modulo M_lambda(x) = mod(x + lambda, 2 lambda) - lambda."""
    return np.mod(x + lam, 2 * lam) - lam


def pam_signal(t, symbols, W=1.0):
    T = 1 / (2 * W)
    t_m = (np.arange(1, len(symbols) + 1) - (len(symbols) + 1) // 2) * T
    return (symbols[None, :] * np.sinc(2 * W * (t[:, None] - t_m[None, :]))).sum(1), t_m


def style(ax):
    ax.grid(True, color="#e6e9ee", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- banner
def make_banner():
    fig = plt.figure(figsize=(16, 4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4)
    ax.axis("off")
    grad = np.linspace(0, 1, 512)[None, :]
    ax.imshow(grad, extent=[0, 16, 0, 4], aspect="auto",
              cmap=matplotlib.colors.LinearSegmentedColormap.from_list("bg", ["#0b1020", "#16213e", "#1a1a3a"]))

    t = np.linspace(-6, 6, 4000)
    sym = np.array([7.5, -9.0, 4.0, 10.0, -6.0, 8.5, -3.5, -9.5, 6.0, 9.0, -7.0, 3.0])
    x, _ = pam_signal(t, sym, W=0.5)
    lam = 2.2
    xs = 12.0 + t * 0.62          # waveform lives on the right, text on the left
    ax.plot(xs, 2 + x * 0.14, color="#5b8def", lw=1.2, alpha=0.35)
    y = fold(x, lam)
    jumps = np.where(np.abs(np.diff(y)) > lam)[0] + 1
    for seg in np.split(np.arange(len(t)), jumps):
        for lw, a in ((9, 0.05), (5, 0.10), (2.2, 1.0)):
            ax.plot(xs[seg], 2 + y[seg] * 0.42, color="#ff6b4a", lw=lw, alpha=a, solid_capstyle="round")
    for yy in (2 + lam * 0.42, 2 - lam * 0.42):
        ax.plot([xs[0], xs[-1]], [yy, yy], color="#ffffff", lw=0.8, ls=(0, (4, 4)), alpha=0.35)
    ax.text(xs[-1], 2 + lam * 0.42 + 0.12, "+λ", color="#ffffff", alpha=0.6, fontsize=11, ha="right")
    ax.text(xs[-1], 2 - lam * 0.42 - 0.32, "−λ", color="#ffffff", alpha=0.6, fontsize=11, ha="right")

    ax.text(0.55, 2.6, "Fold First, Detect Directly", color="white", fontsize=32, weight="bold", va="center")
    ax.text(0.58, 1.85, "Unfolding-free ML symbol detection\nfor low-bitrate modulo-ADCs",
            color="#c9d4ff", fontsize=15, va="center", linespacing=1.4)
    ax.text(0.58, 0.95, "ICASSP 2027 (under review)  ·  arXiv:2609.11298  ·  IIT Bombay",
            color="#8c9bd1", fontsize=12, va="center")
    fig.savefig(OUT / "banner.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------- folding animation
def make_folding_gif():
    lam = 2.0
    sym = np.array([9.75, -5.25, 4.75, -9.75, 5.25, 0.25])
    t = np.linspace(-2.2, 2.2, 1200)
    x, t_m = pam_signal(t, sym, W=1.0)
    y = fold(x, lam)
    OF = 4
    ts = np.arange(-2.2, 2.2 + 1e-9, 0.5 / OF)
    xs, _ = pam_signal(ts, sym, W=1.0)
    ys = fold(xs, lam)

    fig = plt.figure(figsize=(11, 5.2), dpi=80)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 0.08, 2.4], hspace=0.45, wspace=0.05,
                          left=0.085, right=0.98, top=0.88, bottom=0.11)
    axT = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 2])
    ax2 = fig.add_subplot(gs[1, 2], sharex=ax1)
    fig.suptitle("A modulo-ADC wraps a large signal back into ±λ before sampling",
                 fontsize=14, weight="bold", color=INK)

    # transfer curve
    xx = np.linspace(-11, 11, 3000)
    yy = fold(xx, lam)
    br = np.where(np.abs(np.diff(yy)) > lam)[0] + 1
    for seg in np.split(np.arange(len(xx)), br):
        axT.plot(xx[seg], yy[seg], color=RED, lw=2)
    axT.plot(xx, xx, color=GREY, lw=1, ls="--")
    axT.set_xlim(-11, 11)
    axT.set_ylim(-11, 11)
    axT.set_aspect("equal")
    axT.axhspan(-lam, lam, color=RED, alpha=0.07)
    axT.set_xlabel("input")
    axT.set_ylabel("output after folding")
    axT.set_title("folding rule")
    style(axT)
    dotT, = axT.plot([], [], "o", color=INK, ms=8, zorder=5)
    guideT, = axT.plot([], [], color=INK, lw=0.8, ls=":")

    for ax in (ax1, ax2):
        style(ax)
        ax.axhspan(-lam, lam, color=RED, alpha=0.07)
        ax.axhline(lam, color=RED, lw=1, ls="--")
        ax.axhline(-lam, color=RED, lw=1, ls="--")
    ax1.set_ylim(-11.5, 11.5)
    ax2.set_ylim(-2.6 * lam / 2, 2.6 * lam / 2)
    ax1.set_xlim(t[0], t[-1])
    ax1.set_title("received signal  (an ordinary ADC would need a huge range)", loc="left")
    ax2.set_title("after folding into ±λ, with the ADC samples  (a few bits are enough)", loc="left")
    ax2.set_xlabel("time")
    ax1.plot(t, x, color=BLUE, lw=1, alpha=0.18)
    l1, = ax1.plot([], [], color=BLUE, lw=2)
    l2 = [ax2.plot([], [], color=RED, lw=2)[0] for _ in range(40)]
    s2, = ax2.plot([], [], "o", color=INK, ms=4)
    c1 = ax1.axvline(t[0], color=INK, lw=1)
    c2 = ax2.axvline(t[0], color=INK, lw=1)
    ax1.text(1.0, 10.2, rf"$\lambda = {lam:g}$,  peak $\approx {np.abs(x).max():.1f}$", ha="right",
             transform=ax1.get_yaxis_transform(), color=INK, fontsize=10)

    n_frames = 70
    idx = np.linspace(5, len(t) - 1, n_frames).astype(int)

    def update(k):
        i = idx[k]
        l1.set_data(t[:i], x[:i])
        jumps = np.where(np.abs(np.diff(y[:i])) > lam)[0] + 1
        segs = np.split(np.arange(i), jumps)
        for j, ln in enumerate(l2):
            if j < len(segs):
                ln.set_data(t[segs[j]], y[segs[j]])
            else:
                ln.set_data([], [])
        m = ts <= t[i]
        s2.set_data(ts[m], ys[m])
        c1.set_xdata([t[i]])
        c2.set_xdata([t[i]])
        dotT.set_data([x[i]], [y[i]])
        guideT.set_data([x[i], x[i]], [x[i], y[i]])
        return [l1, s2, c1, c2, dotT, guideT, *l2]

    anim = FuncAnimation(fig, update, frames=list(range(n_frames)) + [n_frames - 1] * 18, blit=False)
    anim.save(OUT / "modulo_folding.gif", writer=PillowWriter(fps=16))
    plt.close(fig)


# ---------------------------------------------------------------- wrap cancellation
def make_wrap_cancellation():
    rng = np.random.default_rng(4)
    lam, W, OF, R = 1.0, 1.0, 8, 4
    S = np.array([-8.875, -6.625, -4.375, -2.125, 2.125, 4.625, 6.875, 10.375])
    a_true = np.array([6.875, -8.875, 10.375, -2.125])
    a_wrong = a_true.copy()
    a_wrong[2] = 4.625
    T = 1 / (2 * W)
    t_m = (np.arange(1, 5) - 2) * T
    t_s = np.arange(t_m[0] - T + T / OF, t_m[-1] + T - 1e-9, T / OF)
    H = np.sinc(2 * W * (t_s[:, None] - t_m[None, :]))
    sigma = lam / 6
    C = sigma ** 2 * np.sinc(2 * W * (t_s[:, None] - t_s[None, :]))
    w = rng.multivariate_normal(np.zeros(len(t_s)), C + 1e-9 * np.eye(len(t_s)))
    step = 2 * lam / 2 ** R
    yl = fold(H @ a_true + w, lam)
    yl = -lam + (np.floor((yl + lam) / step).clip(0, 2 ** R - 1) + 0.5) * step
    Sig = C + step ** 2 / 12 * np.eye(len(t_s)) + 1e-6 * np.eye(len(t_s))
    Li = np.linalg.inv(np.linalg.cholesky(Sig))

    def resid(a):
        r = fold(yl - H @ a, lam)
        return r, float(np.sum((Li @ r) ** 2))

    r_true, d_true = resid(a_true)
    r_wrong, d_wrong = resid(a_wrong)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), dpi=110, sharey=True)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.80, bottom=0.14, wspace=0.08)
    panels = [
        (yl, INK, "what the receiver sees: folded 4-bit samples", None),
        (r_true, GREEN, "leftover when the guess is CORRECT", d_true),
        (r_wrong, RED, "leftover when ONE of 4 symbols is wrong", d_wrong),
    ]
    for ax, (v, col, ttl, d) in zip(axs, panels):
        style(ax)
        ax.axhspan(-lam, lam, color=RED, alpha=0.05)
        ax.axhline(lam, color=RED, lw=1, ls="--")
        ax.axhline(-lam, color=RED, lw=1, ls="--")
        ml, sl, bl = ax.stem(t_s / T, v, basefmt=" ")
        plt.setp(ml, color=col, ms=4)
        plt.setp(sl, color=col, lw=1.2)
        ax.set_title(ttl, fontsize=11, loc="left")
        ax.set_xlabel("time (symbol periods)")
        ax.set_ylim(-1.35, 1.35)
        if d is not None:
            ax.text(0.98, 0.95, f"noise score = {d:,.0f}", transform=ax.transAxes, ha="right",
                    va="top", fontsize=11, weight="bold", color=col,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1))
    axs[0].set_ylabel("amplitude")
    fig.suptitle("Wrap cancellation: with the right guess the folds cancel, leaving only small noise (low score)",
                 fontsize=13, weight="bold", color=INK)
    fig.savefig(OUT / "wrap_cancellation.png", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- block search animation
def make_block_search_gif():
    Ns, Nb, G = 16, 4, 1
    core = Nb - 2 * G
    starts = sorted(set(list(range(0, Ns - Nb + 1, core)) + [Ns - Nb]))
    wrong_by_pass = [
        {1, 4, 5, 9, 12, 15},
        {4, 9, 12},
        {9},
        set(),
    ]

    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=80)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04)

    def draw(pass_k, blk, merged):
        ax.clear()
        ax.set_xlim(-0.6, Ns + 0.6)
        ax.set_ylim(-3.2, 2.6)
        ax.axis("off")
        wrong = wrong_by_pass[pass_k + (1 if merged else 0)]
        title = (f"Block search · pass {pass_k + 1} · block {blk + 1} of {len(starts)}  "
                 "(blocks are independent, so they run in parallel)") if not merged else \
            (f"Block search · pass {pass_k + 1}: combine the middle symbols of all blocks"
             + ("  ✓ converged" if not wrong_by_pass[pass_k + 1] else ""))
        ax.set_title(title, fontsize=13, weight="bold", color=INK, loc="left")

        for i in range(Ns):
            col = RED if i in wrong else GREEN
            ax.add_patch(FancyBboxPatch((i + 0.08, 0), 0.84, 0.9, boxstyle="round,pad=0.02,rounding_size=0.12",
                                        fc=col, ec="white", alpha=0.85))
            ax.text(i + 0.5, 0.45, rf"$a_{{{i + 1}}}$", ha="center", va="center", color="white", fontsize=11)
        ax.text(-0.5, 1.25, "current guess of each symbol", fontsize=10, color=INK)

        for j, (fc, al, lab) in enumerate(((GREEN, 0.85, "correct"), (RED, 0.85, "wrong"),
                                           (GREY, 0.30, "block edge (discarded)"))):
            x0 = 9.2 + j * 2.1
            ax.add_patch(Rectangle((x0, 2.05), 0.35, 0.35, fc=fc, alpha=al, ec="none"))
            ax.text(x0 + 0.5, 2.22, lab, fontsize=9, color=INK, va="center")

        ts = np.linspace(0, Ns, Ns * 6 + 1)
        ax.plot(ts, np.full_like(ts, -2.2), "|", color=GREY, ms=7)
        ax.text(-0.5, -2.85, "folded samples (6 per symbol)", fontsize=10, color=INK)

        if not merged:
            s = starts[blk]
            ax.add_patch(Rectangle((s, -0.25), Nb, 1.4, fill=False, ec=BLUE, lw=2.5))
            for g in list(range(s, s + G)) + list(range(s + Nb - G, s + Nb)):
                if not ((s == 0 and g < G) or (s == Ns - Nb and g >= Ns - G)):
                    ax.add_patch(Rectangle((g + 0.08, 0), 0.84, 0.9, fc="white", alpha=0.55, ec="none"))
            ax.text(s + Nb / 2, 1.45, f"try all {8 ** Nb:,} options", ha="center", color=BLUE,
                    fontsize=10, weight="bold")
            eta = 0.75
            lo, hi = max(s - eta, 0), min(s + Nb + eta, Ns)
            ax.add_patch(Rectangle((lo, -2.55), hi - lo, 0.7, fc=BLUE, alpha=0.15, ec=BLUE, lw=1.5))
            ax.annotate("", xy=((lo + hi) / 2, -1.8), xytext=(s + Nb / 2, -0.3),
                        arrowprops=dict(arrowstyle="<-", color=BLUE, lw=1.5))
            ax.text(Ns + 0.5, -2.85, "samples used = the block plus a small margin;  "
                                     "other symbols' effect removed using the current guess",
                    fontsize=9, color=BLUE, ha="right")

    frames = []
    for k in range(len(wrong_by_pass) - 1):
        frames += [(k, b, False) for b in range(len(starts))]
        frames += [(k, 0, True)] * 4

    def update(i):
        draw(*frames[i])
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(OUT / "block_search.gif", writer=PillowWriter(fps=3))
    plt.close(fig)


if __name__ == "__main__":
    make_banner()
    make_folding_gif()
    make_wrap_cancellation()
    make_block_search_gif()
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:28s} {p.stat().st_size / 1024:8.1f} KB")
