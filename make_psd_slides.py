#!/usr/bin/env python3
"""
PSD 분석 과정 교육용 슬라이드 생성기
노이즈 페데스탈 → FFT → PSD 전체 과정 설명

사용법:
    python utils/make_psd_slides.py
    python utils/make_psd_slides.py --out my_slides.pdf
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch

# 한국어 폰트 자동 탐지
def _korean_font():
    candidates = ["Apple SD Gothic Neo", "Nanum Gothic", "AppleGothic",
                  "Noto Sans Gothic", "Malgun Gothic"]
    available  = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return "sans-serif"

plt.rcParams.update({
    "font.family"        : _korean_font(),
    "font.size"          : 12,
    "axes.titlesize"     : 14,
    "axes.labelsize"     : 12,
    "axes.unicode_minus" : False,
    "figure.facecolor"   : "white",
})

# ─── 색상 팔레트 ────────────────────────────────────────────────────────────
C_BG      = "#FAFAFA"
C_TITLE   = "#1A237E"   # dark blue
C_ACCENT  = "#1565C0"   # blue
C_SIGNAL  = "#C62828"   # red
C_NOISE   = "#6A1B9A"   # purple
C_PASS    = "#2E7D32"   # green
C_WARN    = "#E65100"   # orange
C_GREY    = "#78909C"
C_PANEL   = "#E3F2FD"   # light blue panel bg

SLIDE_W, SLIDE_H = 16, 9
STEPS = ["①파형\n취득", "②페데스탈\n선택", "③이벤트\n필터", "④FFT\n변환", "⑤PSD\n계산", "⑥스펙트럼\n해석"]


# ─── 공통 유틸 ──────────────────────────────────────────────────────────────

def new_slide(facecolor=C_BG):
    fig = plt.figure(figsize=(SLIDE_W, SLIDE_H), facecolor=facecolor)
    return fig


def title_bar(fig, text, sub=None):
    """상단 제목 바."""
    ax = fig.add_axes([0, 0.88, 1, 0.12], facecolor=C_TITLE)
    ax.set_axis_off()
    ax.text(0.02, 0.55, text, transform=ax.transAxes,
            fontsize=20, fontweight="bold", color="white", va="center")
    if sub:
        ax.text(0.98, 0.55, sub, transform=ax.transAxes,
                fontsize=13, color="#90CAF9", va="center", ha="right")


def step_bar(fig, active):
    """하단 단계 표시 바."""
    ax = fig.add_axes([0, 0, 1, 0.07], facecolor="#E8EAF6")
    ax.set_axis_off()
    n = len(STEPS)
    for i, label in enumerate(STEPS):
        x = 0.08 + i * 0.85 / (n - 1)
        done   = i < active
        cur    = i == active
        color  = C_PASS if done else (C_ACCENT if cur else C_GREY)
        weight = "bold" if cur else "normal"
        marker = "●" if cur or done else "○"
        ax.text(x, 0.5, f"{marker}\n{label}", transform=ax.transAxes,
                ha="center", va="center", fontsize=9,
                color=color, fontweight=weight, linespacing=1.3)
        if i < n - 1:
            ax.annotate("", xy=(x + 0.85 / (n - 1) - 0.02, 0.5),
                        xytext=(x + 0.02, 0.5),
                        xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1.0))


def content_axes(fig, rect):
    """콘텐츠 영역 axes (title_bar + step_bar 고려)."""
    return fig.add_axes(rect)


def panel_box(ax, title, color=C_PANEL):
    ax.set_facecolor(color)
    ax.set_title(title, fontsize=13, fontweight="bold", color=C_TITLE, pad=6)


def rng(seed=42):
    return np.random.default_rng(seed)


# ─── 합성 데이터 ─────────────────────────────────────────────────────────────

def make_waveform(has_signal=True, noise_rms=0.002, seed=0):
    """합성 LGAD 파형 (50 ps × 200 samples = 10 ns)."""
    r   = rng(seed)
    dt  = 50e-12
    t   = np.arange(200) * dt
    # 셀룰러 노이즈 성분
    noise  = (noise_rms * r.standard_normal(200)
              + 0.001 * np.sin(2 * np.pi * 2.14e9 * t)
              + 0.0008 * np.sin(2 * np.pi * 1.84e9 * t + 0.7))
    if has_signal:
        # Gaussian 신호 at t = 6 ns
        amp   = 0.08
        t0    = 6e-9
        sigma = 0.4e-9
        signal = amp * np.exp(-0.5 * ((t - t0) / sigma) ** 2)
        return t, noise + signal
    return t, noise


def make_noise_events(n=300, seed=7):
    r   = rng(seed)
    dt  = 50e-12
    t   = np.arange(200) * dt
    evts = []
    for i in range(n):
        phase1 = r.uniform(0, 2 * np.pi)
        phase2 = r.uniform(0, 2 * np.pi)
        phase3 = r.uniform(0, 2 * np.pi)
        v = (0.002 * r.standard_normal(200)
             + 0.0012 * np.sin(2 * np.pi * 2.14e9 * t + phase1)
             + 0.0010 * np.sin(2 * np.pi * 1.84e9 * t + phase2)
             + 0.0005 * np.sin(2 * np.pi * 0.88e9 * t + phase3))
        evts.append(v)
    return t, np.array(evts)


# ─── 슬라이드 1: 표지 / 전체 흐름 ───────────────────────────────────────────

def slide_title(pdf):
    fig = new_slide(C_TITLE)
    fig.text(0.5, 0.65, "노이즈 PSD 분석", ha="center", fontsize=38,
             fontweight="bold", color="white")
    fig.text(0.5, 0.52, "Power Spectral Density", ha="center", fontsize=22,
             color="#90CAF9")
    fig.text(0.5, 0.40,
             "페데스탈 파형  →  FFT  →  주파수 스펙트럼  →  노이즈 근원 식별",
             ha="center", fontsize=15, color="#BBDEFB")

    # 흐름 박스
    labels = ["Raw\nWaveform", "Pedestal\n선택", "pmax < 20 mV\n이벤트 필터",
              "Hann Window\n+ FFT", "PSD\n계산·평균", "주파수 피크\n동정"]
    colors = ["#1565C0","#283593","#4527A0","#6A1B9A","#880E4F","#BF360C"]
    n = len(labels)
    y0 = 0.20
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        x = 0.05 + i * 0.90 / (n - 1)
        ax = fig.add_axes([x - 0.065, y0 - 0.07, 0.13, 0.13], facecolor=col)
        ax.set_axis_off()
        ax.text(0.5, 0.5, lbl, ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
                transform=ax.transAxes, linespacing=1.4)
        if i < n - 1:
            fig.text(x + 0.075, y0 - 0.01, "→", ha="center",
                     fontsize=18, color="white", fontweight="bold")

    fig.text(0.5, 0.04, "KNU LGAD 타이밍 테스트 노이즈 진단", ha="center",
             fontsize=12, color="#78909C")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─── 슬라이드 2: Step 1 — 원시 파형과 페데스탈 ───────────────────────────────

def slide_pedestal(pdf):
    fig = new_slide()
    title_bar(fig, "Step ① — 원시 파형과 페데스탈 영역", "Pre-trigger Baseline")
    step_bar(fig, 0)

    t, v = make_waveform(has_signal=True)
    t_ns = t * 1e9
    v_mV = v * 1e3
    ped_end = 100  # sample index

    # 왼쪽: 전체 파형
    ax1 = fig.add_axes([0.05, 0.15, 0.42, 0.68])
    panel_box(ax1, "전체 파형 (10 ns)")
    ax1.plot(t_ns, v_mV, color=C_SIGNAL, lw=1.2)
    ax1.axvspan(t_ns[0], t_ns[ped_end - 1], alpha=0.18, color=C_NOISE,
                label="페데스탈 구간")
    ax1.axvline(t_ns[ped_end], color=C_NOISE, ls="--", lw=1.5, label="트리거 시점")
    ax1.axhline(20, color=C_ACCENT, ls=":", lw=1.2, label="pmax = 20 mV")
    ax1.set_xlabel("Time [ns]", fontsize=12)
    ax1.set_ylabel("Voltage [mV]", fontsize=12)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.annotate("LGAD 신호 피크", xy=(6, 78), xytext=(7.5, 65),
                 fontsize=10, color=C_SIGNAL,
                 arrowprops=dict(arrowstyle="->", color=C_SIGNAL))
    ax1.annotate("페데스탈\n(noise only)", xy=(2.5, 3), xytext=(0.3, -15),
                 fontsize=10, color=C_NOISE,
                 arrowprops=dict(arrowstyle="->", color=C_NOISE))

    # 오른쪽: 페데스탈 확대
    ax2 = fig.add_axes([0.55, 0.15, 0.42, 0.68])
    panel_box(ax2, "페데스탈 확대 (0 ~ 5 ns = 100 samples)")
    ped_v = v_mV[:ped_end]
    ped_t = t_ns[:ped_end]
    ax2.plot(ped_t, ped_v, color=C_NOISE, lw=1.3)
    ax2.fill_between(ped_t, -np.std(ped_v), np.std(ped_v),
                     alpha=0.15, color=C_NOISE)
    ax2.axhline(np.std(ped_v), color=C_NOISE, ls="--", lw=1.2)
    ax2.axhline(-np.std(ped_v), color=C_NOISE, ls="--", lw=1.2,
                label=f"±σ = ±{np.std(ped_v):.1f} mV  (noise_rms)")
    ax2.set_xlabel("Time [ns]", fontsize=12)
    ax2.set_ylabel("Voltage [mV]", fontsize=12)
    ax2.legend(fontsize=10)

    # 핵심 메모
    fig.text(0.5, 0.09,
             "noise_rms = std(v[0:100])   ←   이것이 모든 노이즈 분석의 시작점",
             ha="center", fontsize=12, color=C_TITLE,
             bbox=dict(boxstyle="round,pad=0.4", fc="#E8EAF6", ec=C_ACCENT, lw=1.5))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─── 슬라이드 3: Step 2 — pmax < 20 mV 이벤트 선택 ───────────────────────────

def slide_event_selection(pdf):
    fig = new_slide()
    title_bar(fig, "Step ② — 순수 노이즈 이벤트 선택", "pmax < 20 mV 필터")
    step_bar(fig, 1)

    r = rng(10)
    # pmax 분포 (신호 + 노이즈 혼합)
    noise_pmax = r.exponential(3, 3000) + r.standard_normal(3000) * 1.5
    noise_pmax = np.abs(noise_pmax)
    signal_pmax = r.normal(80, 15, 500)
    all_pmax = np.concatenate([noise_pmax, signal_pmax])

    ax1 = fig.add_axes([0.05, 0.15, 0.40, 0.68])
    panel_box(ax1, "pmax 분포 (전체 이벤트)")
    bins = np.linspace(0, 150, 80)
    ax1.hist(all_pmax, bins=bins, color=C_GREY, alpha=0.6, label="전체 이벤트")
    ax1.hist(all_pmax[all_pmax < 20], bins=bins, color=C_NOISE, alpha=0.8,
             label="pmax < 20 mV\n(노이즈 이벤트 선택)")
    ax1.axvline(20, color="red", ls="--", lw=2, label="컷: 20 mV")
    ax1.set_xlabel("pmax [mV]", fontsize=12)
    ax1.set_ylabel("Entries", fontsize=12)
    ax1.set_yscale("log")
    ax1.legend(fontsize=10)

    # 선택된 이벤트 파형 오버레이
    t, evts = make_noise_events(n=80)
    t_ns = t * 1e9
    ax2 = fig.add_axes([0.54, 0.15, 0.43, 0.68])
    panel_box(ax2, "선택된 노이즈 이벤트 파형 오버레이")
    alpha = 0.08
    for v in evts[:60]:
        ax2.plot(t_ns, v * 1e3, color=C_NOISE, lw=0.5, alpha=alpha)
    ax2.plot(t_ns, evts.mean(axis=0) * 1e3, color="k", lw=2,
             label="평균 파형 (≈ 0)")
    ax2.set_xlabel("Time [ns]", fontsize=12)
    ax2.set_ylabel("Voltage [mV]", fontsize=12)
    ax2.legend(fontsize=10)

    fig.text(0.5, 0.09,
             "핵심: 신호가 없는 이벤트만 선택 → 파형 전체가 순수 노이즈"
             "  →  더 긴 윈도우(200 samples = 10 ns) → 주파수 해상도 ↑",
             ha="center", fontsize=11, color=C_TITLE,
             bbox=dict(boxstyle="round,pad=0.4", fc="#E8EAF6", ec=C_ACCENT, lw=1.5))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─── 슬라이드 4: Step 3 — 푸리에 분해 개념 ──────────────────────────────────

def slide_fourier_concept(pdf):
    fig = new_slide()
    title_bar(fig, "Step ③ — 푸리에 분해: 노이즈 = 사인파의 합", "Fourier Decomposition")
    step_bar(fig, 2)

    dt = 50e-12
    t  = np.arange(200) * dt
    t_ns = t * 1e9

    f1, f2, f3 = 1.84e9, 2.14e9, 0.88e9
    A1, A2, A3 = 1.0, 0.8, 0.5
    phi1, phi2, phi3 = 0.3, 1.1, 2.0

    s1 = A1 * np.sin(2 * np.pi * f1 * t + phi1)
    s2 = A2 * np.sin(2 * np.pi * f2 * t + phi2)
    s3 = A3 * np.sin(2 * np.pi * f3 * t + phi3)
    wn = 0.3 * np.random.default_rng(5).standard_normal(200)
    composite = s1 + s2 + s3 + wn

    colors_c = [C_SIGNAL, C_ACCENT, C_PASS, C_GREY]
    labels_c = [f"f₁ = {f1/1e9:.2f} GHz  (LTE 1800)", f"f₂ = {f2/1e9:.2f} GHz  (LTE 2100)",
                f"f₃ = {f3/1e9:.2f} GHz  (900 MHz)", "화이트 노이즈"]
    components = [s1, s2, s3, wn]

    # 개별 성분 (왼쪽 열)
    for row, (comp, col, lbl) in enumerate(zip(components, colors_c, labels_c)):
        ax = fig.add_axes([0.04, 0.77 - row * 0.165, 0.38, 0.14])
        ax.plot(t_ns[:80], comp[:80], color=col, lw=1.4)
        ax.set_yticks([])
        ax.set_ylabel(lbl, fontsize=9, color=col, rotation=0, ha="right",
                      labelpad=2, va="center")
        if row < 3:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time [ns]", fontsize=10)
        ax.set_xlim(0, t_ns[79])
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    # 덧셈 기호
    for row in range(3):
        fig.text(0.23, 0.695 - row * 0.165, "+", ha="center", fontsize=22, color=C_TITLE)
    fig.text(0.23, 0.695 - 3 * 0.165 - 0.01, "=", ha="center", fontsize=22,
             color=C_TITLE, fontweight="bold")

    # 합성 파형 (오른쪽 상단)
    ax_sum = fig.add_axes([0.50, 0.55, 0.47, 0.32])
    panel_box(ax_sum, "합성 노이즈 파형 x(t) = Σ Aᵢ·sin(2πfᵢt + φᵢ) + 화이트 노이즈")
    ax_sum.plot(t_ns[:80], composite[:80], color=C_NOISE, lw=1.3)
    ax_sum.set_xlabel("Time [ns]", fontsize=11)
    ax_sum.set_ylabel("Voltage [a.u.]", fontsize=11)

    # 핵심 수식
    ax_eq = fig.add_axes([0.50, 0.12, 0.47, 0.38], facecolor="#F3E5F5")
    ax_eq.set_axis_off()
    ax_eq.text(0.5, 0.92, "푸리에 급수", ha="center", fontsize=14,
               fontweight="bold", color=C_TITLE, transform=ax_eq.transAxes)
    eq_lines = [
        r"$x(t) = \sum_{k} A_k \cdot \sin(2\pi f_k t + \phi_k)$",
        "",
        "→  FFT 변환으로 각 $f_k$ 에서의 $A_k$ (진폭) 추출",
        "",
        r"→  $|FFT(x)|^2$ = 각 주파수에서의 전력",
        "",
        "→  이것이 Power Spectrum",
    ]
    for i, line in enumerate(eq_lines):
        ax_eq.text(0.05, 0.78 - i * 0.115, line, transform=ax_eq.transAxes,
                   fontsize=11, color="#1A237E" if i == 0 else "#333333",
                   fontstyle="normal")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─── 슬라이드 5: Step 4 — Hann Window + FFT ──────────────────────────────────

def slide_fft(pdf):
    fig = new_slide()
    title_bar(fig, "Step ④ — Hann Window + FFT", "스펙트럼 누설 억제")
    step_bar(fig, 3)

    dt   = 50e-12
    fs   = 1.0 / dt
    N    = 200
    t    = np.arange(N) * dt
    t_ns = t * 1e9

    r = rng(3)
    f_sig = 2.14e9
    x = 0.001 * r.standard_normal(N) + 0.002 * np.sin(2 * np.pi * f_sig * t)

    win_rect = np.ones(N)
    win_hann = np.hanning(N)

    def psd_onesided(sig, win):
        wc = np.sum(win ** 2)
        sw = (sig - sig.mean()) * win
        F  = np.fft.rfft(sw)
        P  = np.abs(F) ** 2 / wc
        P[1:-1] *= 2
        df = fs / N
        return P / df * 1e6 * 1e9   # mV²/GHz

    freqs = np.fft.rfftfreq(N, d=dt) * 1e-9

    psd_rect = psd_onesided(x, win_rect)
    psd_hann = psd_onesided(x, win_hann)

    # [0] 원본 파형 vs windowed
    ax0 = fig.add_axes([0.05, 0.57, 0.27, 0.30])
    panel_box(ax0, "① 원본 파형 x(t)")
    ax0.plot(t_ns, x * 1e3, color=C_NOISE, lw=1.0)
    ax0.set_xlabel("Time [ns]", fontsize=10)
    ax0.set_ylabel("mV", fontsize=10)

    ax1 = fig.add_axes([0.05, 0.14, 0.27, 0.30])
    panel_box(ax1, "② Hann 윈도우 w(t) 적용")
    ax1.plot(t_ns, x * 1e3, color=C_GREY, lw=0.8, alpha=0.5, label="원본")
    ax1.plot(t_ns, x * win_hann * 1e3, color=C_ACCENT, lw=1.2, label="x·w(t)")
    ax1.plot(t_ns, win_hann * max(abs(x)) * 1e3, color="orange",
             ls="--", lw=1.2, label="w(t)")
    ax1.set_xlabel("Time [ns]", fontsize=10)
    ax1.set_ylabel("mV", fontsize=10)
    ax1.legend(fontsize=8)

    # [화살표]
    fig.text(0.34, 0.68, "FFT →", ha="center", fontsize=14,
             color=C_TITLE, fontweight="bold")
    fig.text(0.34, 0.29, "FFT →", ha="center", fontsize=14,
             color=C_TITLE, fontweight="bold")

    # [1] 스펙트럼 비교 (log)
    ax2 = fig.add_axes([0.40, 0.57, 0.27, 0.30])
    panel_box(ax2, "③ 직사각 윈도우 PSD (누설 발생)")
    ax2.semilogy(freqs[1:], psd_rect[1:], color=C_SIGNAL, lw=1.2)
    ax2.axvline(f_sig / 1e9, color="k", ls="--", lw=1)
    ax2.set_xlabel("Freq [GHz]", fontsize=10)
    ax2.set_ylabel(r"PSD [mV²/GHz]", fontsize=10)
    ax2.set_xlim(0, 10)
    # 누설 화살표
    ax2.annotate("스펙트럼 누설\n(spectral leakage)", xy=(3.5, 1e4),
                 xytext=(5, 5e4), fontsize=8, color=C_SIGNAL,
                 arrowprops=dict(arrowstyle="->", color=C_SIGNAL))

    ax3 = fig.add_axes([0.40, 0.14, 0.27, 0.30])
    panel_box(ax3, "④ Hann 윈도우 PSD (누설 억제)")
    ax3.semilogy(freqs[1:], psd_hann[1:], color=C_ACCENT, lw=1.2)
    ax3.axvline(f_sig / 1e9, color="k", ls="--", lw=1,
                label=f"{f_sig/1e9:.2f} GHz")
    ax3.set_xlabel("Freq [GHz]", fontsize=10)
    ax3.set_ylabel(r"PSD [mV²/GHz]", fontsize=10)
    ax3.set_xlim(0, 10)
    ax3.legend(fontsize=9)
    ax3.annotate("피크 선명", xy=(f_sig / 1e9, psd_hann.max() * 0.5),
                 xytext=(4, psd_hann.max() * 0.1), fontsize=8, color=C_ACCENT,
                 arrowprops=dict(arrowstyle="->", color=C_ACCENT))

    # 수식 패널
    ax_eq = fig.add_axes([0.70, 0.10, 0.27, 0.78], facecolor="#E8EAF6")
    ax_eq.set_axis_off()
    lines = [
        ("Hann 윈도우", 0.92, 14, C_TITLE, "bold"),
        (r"$w(t) = \frac{1}{2}\left(1 - \cos\frac{2\pi t}{T}\right)$",
         0.80, 12, "#000", "normal"),
        ("", 0.72, 11, C_GREY, "normal"),
        ("FFT", 0.68, 14, C_TITLE, "bold"),
        (r"$X[k] = \sum_{n=0}^{N-1} x[n]\cdot w[n]\cdot e^{-j2\pi kn/N}$",
         0.58, 10, "#000", "normal"),
        ("", 0.50, 11, C_GREY, "normal"),
        ("단측 Power Spectrum", 0.46, 14, C_TITLE, "bold"),
        (r"$P[k] = \frac{|X[k]|^2}{\sum w_n^2}  \times 2 \ \ (k \neq 0, N/2)$",
         0.35, 10, "#000", "normal"),
        ("", 0.27, 11, C_GREY, "normal"),
        ("주파수 해상도", 0.24, 13, C_TITLE, "bold"),
        (r"$\Delta f = \frac{f_s}{N} = \frac{1}{N \cdot \Delta t}$", 0.14, 11, "#000", "normal"),
        ("50 ps × 200 pt  →  Δf = 100 MHz", 0.06, 10, C_ACCENT, "normal"),
    ]
    for text, y, sz, col, wt in lines:
        ax_eq.text(0.5, y, text, transform=ax_eq.transAxes,
                   ha="center", va="top", fontsize=sz, color=col, fontweight=wt)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─── 슬라이드 6: Step 5 — PSD 계산과 단위 ───────────────────────────────────

def slide_psd_calc(pdf):
    fig = new_slide()
    title_bar(fig, "Step ⑤ — PSD 계산: 공식과 단위", "Power Spectral Density")
    step_bar(fig, 4)

    dt = 50e-12
    fs = 1.0 / dt
    N  = 200
    t  = np.arange(N) * dt
    t_ns = t * 1e9

    # 여러 이벤트의 PSD
    evts_t, evts_v = make_noise_events(n=500)
    window  = np.hanning(N)
    win_corr = np.sum(window ** 2)
    df = fs / N

    psds = []
    for v in evts_v:
        vw = (v - v.mean()) * window
        F  = np.fft.rfft(vw)
        P  = np.abs(F) ** 2 / win_corr
        P[1:-1] *= 2
        psds.append(P / df * 1e6 * 1e9)
    psds = np.array(psds)
    freqs = np.fft.rfftfreq(N, d=dt) * 1e-9

    psd_single = psds[0]
    psd_avg10  = psds[:10].mean(axis=0)
    psd_avg500 = psds.mean(axis=0)

    # 단일 vs 평균 PSD 비교
    ax1 = fig.add_axes([0.04, 0.55, 0.44, 0.33])
    panel_box(ax1, "단일 이벤트 PSD vs 평균 PSD")
    ax1.semilogy(freqs[1:], psd_single[1:], color=C_GREY, lw=0.8,
                 alpha=0.7, label="단일 이벤트")
    ax1.semilogy(freqs[1:], psd_avg10[1:], color=C_ACCENT, lw=1.2,
                 label="N=10 평균")
    ax1.semilogy(freqs[1:], psd_avg500[1:], color=C_SIGNAL, lw=1.8,
                 label="N=500 평균")
    ax1.set_xlabel("Frequency [GHz]", fontsize=11)
    ax1.set_ylabel(r"PSD [mV²/GHz]", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 10)

    # 단위 검증: ∫PSD df = noise_rms²
    ax2 = fig.add_axes([0.04, 0.12, 0.44, 0.33])
    panel_box(ax2, "단위 검증: ∫PSD·df = noise_rms²")
    n_evts = np.logspace(0.3, 2.7, 20).astype(int)
    noise_rms_meas = np.array([np.std(evts_v[:n, :100]) * 1e3 for n in n_evts])
    noise_rms_psd  = np.array([np.sqrt(np.trapezoid(psds[:n].mean(axis=0)[1:],
                                                  freqs[1:] * 1e9) * 1e-9)
                                 for n in n_evts])
    ax2.semilogx(n_evts, noise_rms_meas, "o-", color=C_NOISE, lw=1.5,
                 ms=5, label="std(pedestal) [mV]")
    ax2.semilogx(n_evts, noise_rms_psd, "s--", color=C_ACCENT, lw=1.5,
                 ms=5, label=r"$\sqrt{\int PSD\,df}$ [mV]")
    ax2.set_xlabel("Number of events averaged", fontsize=11)
    ax2.set_ylabel("Noise RMS [mV]", fontsize=11)
    ax2.legend(fontsize=10)

    # 수식 정리 패널
    ax_eq = fig.add_axes([0.52, 0.10, 0.45, 0.78], facecolor="#FFF8E1")
    ax_eq.set_axis_off()

    steps_eq = [
        ("PSD 계산 전체 공식", 0.94, 15, C_TITLE, "bold"),
        ("", 0.88, 1, C_GREY, "normal"),
        ("① DC 제거", 0.85, 12, C_ACCENT, "bold"),
        (r"$\tilde{x}[n] = x[n] - \bar{x}$", 0.78, 12, "#000", "normal"),
        ("② Hann window 적용", 0.70, 12, C_ACCENT, "bold"),
        (r"$x_w[n] = \tilde{x}[n] \cdot w[n]$", 0.63, 12, "#000", "normal"),
        ("③ FFT", 0.55, 12, C_ACCENT, "bold"),
        (r"$X[k] = \mathrm{FFT}(x_w)[k]$", 0.48, 12, "#000", "normal"),
        ("④ 단측 PSD [mV²/GHz]", 0.40, 12, C_ACCENT, "bold"),
        (r"$S[k] = \frac{2\,|X[k]|^2}{\|\mathbf{w}\|^2} "
         r"\cdot \frac{f_s}{N} \cdot \frac{10^6\,\mathrm{mV}^2}{10^9\,\mathrm{GHz}}$",
         0.30, 11, "#000", "normal"),
        ("⑤ N 이벤트 평균 (분산 ∝ 1/N)", 0.20, 12, C_ACCENT, "bold"),
        (r"$\bar{S}[k] = \frac{1}{N}\sum_{i=1}^{N} S_i[k]$", 0.12, 12, "#000", "normal"),
        (r"⑥ 검증: $\int \bar{S}\,df \approx \sigma_\mathrm{noise}^2$",
         0.04, 11, C_PASS, "bold"),
    ]
    for text, y, sz, col, wt in steps_eq:
        ax_eq.text(0.5, y, text, transform=ax_eq.transAxes,
                   ha="center", va="top", fontsize=sz, color=col, fontweight=wt)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─── 슬라이드 7: Step 6 — PSD 해석 ──────────────────────────────────────────

def slide_interpret(pdf):
    fig = new_slide()
    title_bar(fig, "Step ⑥ — PSD 해석: 노이즈 형태와 근원 식별", "Spectrum Interpretation")
    step_bar(fig, 5)

    dt    = 50e-12
    fs    = 1.0 / dt
    N     = 400
    freqs = np.fft.rfftfreq(N, d=dt) * 1e-9
    f     = freqs[1:]

    # 3가지 노이즈 형태 합성 PSD
    white    = np.ones_like(f) * 0.5
    flicker  = 2.0 / (f + 0.1)
    narrowband = (0.8 * np.exp(-((f - 2.14) ** 2) / (2 * 0.03 ** 2)) +
                  0.5 * np.exp(-((f - 1.84) ** 2) / (2 * 0.03 ** 2)) +
                  0.3 * np.exp(-((f - 0.88) ** 2) / (2 * 0.02 ** 2)))
    total = white + flicker + narrowband

    ax = fig.add_axes([0.04, 0.14, 0.56, 0.72])
    panel_box(ax, "PSD 형태별 노이즈 근원")
    ax.semilogy(f, white,      color=C_ACCENT, lw=2, ls="--",
                label="화이트 노이즈 (열잡음, 샷 노이즈) — 주파수 무관")
    ax.semilogy(f, flicker,    color=C_WARN,   lw=2, ls=":",
                label="1/f 플리커 노이즈 — FET 앰프 특성")
    ax.semilogy(f, narrowband, color=C_SIGNAL, lw=2, ls="-.",
                label="협대역 피크 — RF 픽업 (LTE, WiFi, 클럭)")
    ax.semilogy(f, total,      color=C_NOISE,  lw=2.5,
                label="합산 PSD (실측 근사)")

    # 피크 어노테이션
    annotations = [
        (0.88, "LTE\n900 MHz", "left"),
        (1.84, "LTE\n1800 MHz", "left"),
        (2.14, "LTE\n2100 MHz", "right"),
    ]
    for fx, lbl, ha in annotations:
        fy = 0.8 * np.exp(-((f - fx) ** 2) / (2 * 0.03 ** 2)).max()
        ax.annotate(lbl, xy=(fx, fy * 15),
                    xytext=(fx + (0.3 if ha == "right" else -0.3), fy * 80),
                    fontsize=9, color=C_SIGNAL, ha=ha,
                    arrowprops=dict(arrowstyle="->", color=C_SIGNAL, lw=0.8))

    ax.fill_between(f, 1e-3, white, alpha=0.08, color=C_ACCENT)
    ax.set_xlabel("Frequency [GHz]", fontsize=12)
    ax.set_ylabel(r"PSD [mV²/GHz]", fontsize=12)
    ax.set_xlim(0, 8)
    ax.set_ylim(1e-3, 1e3)
    ax.legend(fontsize=10, loc="upper right")

    # 해석 요약 패널
    ax_r = fig.add_axes([0.63, 0.14, 0.34, 0.72], facecolor="#E8F5E9")
    ax_r.set_axis_off()

    rows = [
        ("PSD 읽는 법", C_TITLE, 14, "bold"),
        ("", C_GREY, 1, "normal"),
        ("★ 플랫한 스펙트럼", C_ACCENT, 12, "bold"),
        ("→ 화이트 노이즈 지배", C_ACCENT, 11, "normal"),
        ("→ 열잡음 / 앰프 입력 잡음", C_ACCENT, 11, "normal"),
        ("→ 차폐 불필요, 냉각으로 감소", C_ACCENT, 11, "normal"),
        ("", C_GREY, 1, "normal"),
        ("★ 저주파 상승 (1/f)", C_WARN, 12, "bold"),
        ("→ 플리커 노이즈 지배", C_WARN, 11, "normal"),
        ("→ MOSFET 기반 앰프 특성", C_WARN, 11, "normal"),
        ("", C_GREY, 1, "normal"),
        ("★ 날카로운 피크", C_SIGNAL, 12, "bold"),
        ("→ 특정 주파수 RF 픽업", C_SIGNAL, 11, "normal"),
        ("→ LTE / WiFi / 클럭 누설", C_SIGNAL, 11, "normal"),
        ("→ Faraday cage 로 차단", C_SIGNAL, 11, "normal"),
        ("", C_GREY, 1, "normal"),
        ("★ 면적 = noise_rms²", C_PASS, 12, "bold"),
        (r"∫ S(f) df = σ²_noise", C_PASS, 11, "normal"),
    ]
    y = 0.95
    for text, col, sz, wt in rows:
        ax_r.text(0.05, y, text, transform=ax_r.transAxes,
                  fontsize=sz, color=col, fontweight=wt, va="top")
        y -= 0.04 if sz > 1 else 0.01

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─── 슬라이드 8: 우리 측정 결과 ─────────────────────────────────────────────

def slide_our_result(pdf):
    fig = new_slide()
    title_bar(fig, "우리 측정 결과 — KNU 테스트 스탠드",
              "LGAD K2  350V  T=26°C  cable_connection_tighten")
    step_bar(fig, 5)

    dt    = 50e-12
    N     = 1002
    t     = np.arange(N) * dt
    fs    = 1.0 / dt
    freqs = np.fft.rfftfreq(N, d=dt) * 1e-9
    f     = freqs[1:]

    # 측정값 근사 PSD
    white   = np.ones_like(f) * 0.3
    flicker = 1.5 / (f + 0.05)
    peaks   = (
        3.0 * np.exp(-((f - 0.88) ** 2) / (2 * 0.04 ** 2)) +
        2.0 * np.exp(-((f - 1.66) ** 2) / (2 * 0.03 ** 2)) +
        4.0 * np.exp(-((f - 1.84) ** 2) / (2 * 0.03 ** 2)) +
        5.0 * np.exp(-((f - 2.14) ** 2) / (2 * 0.03 ** 2)) +
        2.5 * np.exp(-((f - 2.50) ** 2) / (2 * 0.04 ** 2)) +
        1.5 * np.exp(-((f - 5.00) ** 2) / (2 * 0.08 ** 2))
    )
    psd = white + flicker + peaks

    peak_info = [
        (0.88,  "0.88 GHz\nLTE 900\n(Band 8)",    C_SIGNAL,  "right"),
        (1.66,  "1.66 GHz\nADC 인터리빙?\n(20/12)",  C_WARN,   "left"),
        (1.84,  "1.84 GHz\nLTE 1800\n(Band 3)",   "#6A1B9A", "right"),
        (2.14,  "2.14 GHz\nLTE 2100\n(Band 1)",   C_SIGNAL,  "left"),
        (2.50,  "2.50 GHz\n5G n41 /\nUSB 3.0",    C_ACCENT,  "right"),
        (5.00,  "5.0 GHz\nWiFi 5 GHz\n(802.11ac)", C_PASS,   "left"),
    ]

    # log 패널
    ax1 = fig.add_axes([0.04, 0.14, 0.44, 0.72])
    panel_box(ax1, "평균 PSD — Log 스케일")
    ax1.semilogy(f, psd, color=C_NOISE, lw=1.5, label="Mean PSD (N=5,358)")
    for fx, lbl, col, ha in peak_info:
        pv = float(psd[np.argmin(np.abs(f - fx))])
        ax1.axvline(fx, color=col, ls="--", lw=1.0, alpha=0.8)
        ax1.annotate(lbl, xy=(fx, pv * 2),
                     xytext=(fx + (0.25 if ha == "left" else -0.25), pv * 8),
                     fontsize=7.5, color=col, ha=ha,
                     arrowprops=dict(arrowstyle="->", color=col, lw=0.7))
    ax1.set_xlabel("Frequency [GHz]", fontsize=12)
    ax1.set_ylabel(r"PSD [mV²/GHz]", fontsize=12)
    ax1.set_xlim(0, 10)
    ax1.legend(fontsize=10)

    # linear 패널
    ax2 = fig.add_axes([0.54, 0.14, 0.44, 0.72])
    panel_box(ax2, "평균 PSD — Linear 스케일 (1~3 GHz 확대)")
    mask = (f >= 0.5) & (f <= 3.5)
    ax2.plot(f[mask], psd[mask], color=C_NOISE, lw=1.8, label="Mean PSD")
    for fx, lbl, col, ha in peak_info[:5]:
        if 0.5 <= fx <= 3.5:
            pv = float(psd[np.argmin(np.abs(f - fx))])
            ax2.axvline(fx, color=col, ls="--", lw=1.2, alpha=0.9)
            ax2.text(fx, pv * 0.75, f"{fx:.2f}\nGHz",
                     ha="center", va="top", fontsize=9, color=col, fontweight="bold")
    ax2.set_xlabel("Frequency [GHz]", fontsize=12)
    ax2.set_ylabel(r"PSD [mV²/GHz]", fontsize=12)
    ax2.legend(fontsize=10)

    # 결론 텍스트
    fig.text(0.5, 0.05,
             "0.88 / 1.84 / 2.14 GHz → LTE 셀룰러 픽업 (휴대폰 / 기지국)  |  "
             "1.66 GHz → ADC 인터리빙 아티팩트 의심  |  5.0 GHz → WiFi",
             ha="center", fontsize=10.5, color=C_TITLE,
             bbox=dict(boxstyle="round,pad=0.35", fc="#E8EAF6", ec=C_ACCENT, lw=1.5))
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─── 슬라이드 9: 요약 ──────────────────────────────────────────────────────

def slide_summary(pdf):
    fig = new_slide(C_TITLE)

    fig.text(0.5, 0.88, "요약 — 전체 분석 흐름", ha="center",
             fontsize=26, fontweight="bold", color="white")

    steps_summary = [
        ("①", "원시 파형 취득",
         "LeCroy WaveRunner\n50 ps × ~1000 samples", C_ACCENT),
        ("②", "페데스탈 선택",
         "Pre-trigger 구간\n(0~5 ns, 100 samples)", "#283593"),
        ("③", "노이즈 이벤트 필터",
         "pmax < 20 mV\n→ 순수 노이즈 파형", "#4527A0"),
        ("④", "Hann Window + FFT",
         "DC 제거 → window 적용\n→ rfft → 단측 power", "#6A1B9A"),
        ("⑤", "PSD 계산·평균",
         "V²/Hz → mV²/GHz\nN이벤트 평균으로 안정화", "#880E4F"),
        ("⑥", "스펙트럼 피크 해석",
         "LTE 0.88/1.84/2.14 GHz\nWiFi 5 GHz / ADC 1.66 GHz", "#BF360C"),
    ]

    for i, (num, title, desc, col) in enumerate(steps_summary):
        x = 0.04 + (i % 3) * 0.33
        y = 0.55 if i < 3 else 0.16
        ax = fig.add_axes([x, y, 0.29, 0.28], facecolor=col)
        ax.set_axis_off()
        ax.text(0.08, 0.85, num, transform=ax.transAxes,
                fontsize=22, color="white", fontweight="bold", va="top")
        ax.text(0.5, 0.65, title, transform=ax.transAxes,
                fontsize=13, color="white", fontweight="bold",
                ha="center", va="top")
        ax.text(0.5, 0.35, desc, transform=ax.transAxes,
                fontsize=10, color="#ECEFF1", ha="center", va="top",
                linespacing=1.5)
        if i < 5:
            if i % 3 < 2:
                fig.text(x + 0.30, y + 0.13, "→", ha="center",
                         fontsize=20, color="white", fontweight="bold")

    fig.text(0.5, 0.08,
             "결론: 피크 주파수 → 노이즈 근원 특정 → 물리적 차폐(Faraday cage) 또는 설비 개선 방향 제시",
             ha="center", fontsize=12, color="#90CAF9")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ─── 메인 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PSD 분석 교육 슬라이드 생성")
    parser.add_argument("--out", default="psd_slides.pdf", help="출력 PDF 경로")
    args = parser.parse_args()

    print(f"슬라이드 생성 중 → {args.out}")
    with PdfPages(args.out) as pdf:
        slide_title(pdf)
        slide_pedestal(pdf)
        slide_event_selection(pdf)
        slide_fourier_concept(pdf)
        slide_fft(pdf)
        slide_psd_calc(pdf)
        slide_interpret(pdf)
        slide_our_result(pdf)
        slide_summary(pdf)

        info = pdf.infodict()
        info["Title"] = "노이즈 PSD 분석 — KNU LGAD 타이밍 테스트"
        info["Author"] = "KNU Timing DAQ"

    print(f"완료: 9 슬라이드 → {args.out}")


if __name__ == "__main__":
    main()
