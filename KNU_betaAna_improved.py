#!/usr/bin/env python3
"""
KNU_betaAna_improved.py

KNU 알고리즘 개선 버전 (PLAN.md 개선 1~6 구현):
  개선 1: 피크 7포인트 Gaussian fit → pmax_fit, tmax_fit
  개선 2: CFD backward search (피크에서 왼쪽으로 역탐색)
  개선 3: area_new 방식 collected charge (10-20% 외삽 → Simpson)
  개선 5: rise time (10%→90%), dV/dt @ CFD, pulse_width
  개선 6: 자동 pmax/tmax 컷 (2-pass, global max 오른쪽 minimum)

사용법:
  python KNU_betaAna_improved.py --file <ROOT파일> --voltage <전압> [--outdir <출력루트>]
  python KNU_betaAna_improved.py --file ... --voltage 330 --pmax-cut 460
  python KNU_betaAna_improved.py --file ... --voltage 330 --tmax-window 24.5 1.0
  python KNU_betaAna_improved.py --file ... --voltage 330 --chunk-size 2000
"""

import argparse
import os
import sys
import warnings

import numpy as np
import uproot
import mplhep as hep
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy import integrate, signal
from scipy.stats import moyal

hep.style.use("CMS")

CFD_FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

# =============================================================================
# 분석 파라미터 — 여기서 조정
# =============================================================================

# ── 신호 최소 threshold (Pass 1 빠른 필터) ───────────────────────────────────
LOOSE_THR_CH1 = 0.001   # LGAD 최소 진폭 [V]  (1 mV)
LOOSE_THR_CH2 = 0.000   # MCP  최소 진폭 [V]  (0 = 비활성화)

# ── 파형 처리 ──────────────────────────────────────────────────────────────────
BASELINE_SAMPLES   = 100   # 페데스탈·noise RMS 계산에 사용할 pre-trigger 샘플 수
AREA_THR_LO        = 0.10  # 전하 면적 적분 하한 분율 (pmax 의 N%)
AREA_THR_HI        = 0.20  # 전하 면적 외삽 상한 분율
CFD_DVDT_N_FIT     = 5     # CFD crossing 기울기(dV/dt) 선형회귀 샘플 수

# ── Gaussian 피크 피팅 ──────────────────────────────────────────────────────────
GAUSS_FIT_HALF_PTS = 3     # 피크 양쪽으로 쓸 포인트 수 (총 2×N+1 포인트)
GAUSS_FIT_T_BOUND  = 10    # 시간 bound: tmax ± N×dt 범위로 제한

# ── 자동 컷 (auto_pmax_cut / auto_tmax_cut) ─────────────────────────────────────
AUTOCUT_PMAX_FALLBACK_MV = 100.0  # valley 미검출 시 fallback pmax 컷 [mV]
AUTOCUT_PMAX_BINS        = 200    # pmax 히스토그램 빈 수 (valley 탐색용)
AUTOCUT_VALLEY_RATIO     = 0.7    # valley 가 noise peak 대비 이 비율 미만이면 유효
AUTOCUT_FWHM_MULT        = 2      # fallback: noise_peak + N×FWHM
AUTOCUT_TMAX_BINS        = 200    # tmax 히스토그램 빈 수
AUTOCUT_TMAX_HALF_NS     = 1.0    # tmax 자동 컷 윈도우 반폭 [ns]

# ── 타이밍 분해능 피팅 ──────────────────────────────────────────────────────────
DT_HIST_BINS          = 200    # ΔT 히스토그램 빈 수
DT_HIST_HALF_RANGE_PS = 1000.0 # ΔT 히스토그램 범위: median ± N ps
CHI2_NDF_HI           = 1.5   # chi²/ndf 이 값 이상이면 rebin 플롯 추가 생성
CHI2_NDF_LO           = 0.5   # chi²/ndf 이 값 미만이면 rebin 플롯 추가 생성

# ── 전하 피팅 ────────────────────────────────────────────────────────────────────
CHARGE_FIT_BINS = 80   # Landau 피팅용 전하 히스토그램 빈 수

# ── 앰프 이득 보정 (charge ÷ gain) ───────────────────────────────────────────────
AMP_GAIN_MAP = {"20dB": 10.0, "40dB": 122.0}

# ── 플롯 Y축 범위 ─────────────────────────────────────────────────────────────────
NOISE_SCAN_YLIM       = (30, 90)  # noise RMS cut scan σ_t [ps]
NOISE_SCAN_YLIM_SR    = (35, 45) # Sr 파일 전용 줌인 범위 [ps]
CHARGE_SCAN_YLIM_ZOOM = (50, 70) # 전하 cut scan 줌인 범위 [mV·ns]

# =============================================================================
# 헬퍼 함수
# =============================================================================

def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def landau_pdf(x, mpv, sigma, amplitude):
    return amplitude * moyal.pdf(x, loc=mpv, scale=sigma)


def fit_charge_mpv(arr, n_bins=CHARGE_FIT_BINS):
    """Landau(Moyal) fit → (mpv, mpv_err). 실패 시 (nan, nan)."""
    arr = arr[np.isfinite(arr) & (arr > 0)]
    if len(arr) < 10:
        return np.nan, np.nan
    counts, bins = np.histogram(arr, bins=n_bins)
    bc = (bins[:-1] + bins[1:]) / 2
    try:
        p0 = [float(np.median(arr)), float(np.std(arr)) * 0.15,
              float(max(counts)) * float(np.std(arr)) * 0.15 * 2]
        popt, pcov = curve_fit(landau_pdf, bc, counts, p0=p0,
                               sigma=np.sqrt(np.maximum(counts, 1)),
                               absolute_sigma=True, maxfev=3000)
        return float(popt[0]), float(np.sqrt(pcov[0, 0]))
    except Exception:
        return np.nan, np.nan


def gaussian_peak_fit(v, t, imax, dt):
    """
    피크 근방 7포인트 Gaussian fit.
    반환: (pmax_fit [V], tmax_fit [s])  — 실패 시 raw 값 반환
    """
    lo = max(0, imax - GAUSS_FIT_HALF_PTS)
    hi = min(len(v), imax + GAUSS_FIT_HALF_PTS + 1)
    if hi - lo < GAUSS_FIT_HALF_PTS + 2:
        return float(v[imax]), float(t[imax])
    x_g = t[lo:hi]
    y_g = v[lo:hi]
    v_raw = float(v[imax])
    t_raw = float(t[imax])
    try:
        p0 = [v_raw, t_raw, 3 * dt]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(gaussian, x_g, y_g, p0=p0,
                                maxfev=500,
                                bounds=([0, t_raw - GAUSS_FIT_T_BOUND*dt, dt*0.5],
                                        [v_raw * 3, t_raw + GAUSS_FIT_T_BOUND*dt, 20*dt]))
        if popt[0] > 0 and abs(popt[1] - t_raw) < GAUSS_FIT_T_BOUND * dt:
            return float(popt[0]), float(popt[1])
    except Exception:
        pass
    return v_raw, t_raw


def xlinear_inter(x0, y0, x1, y1, ytarget):
    """선형 보간: (x0,y0)-(x1,y1) 직선에서 y=ytarget인 x"""
    denom = y1 - y0
    if abs(denom) < 1e-30:
        return 0.5 * (x0 + x1)
    return x0 + (ytarget - y0) * (x1 - x0) / denom


def compute_area_new(v, t, imax, pmax_fit):
    """
    Torino C++ New_Pulse_Area 방식.
    10-20% 외삽으로 펄스 경계(start_time, end_time) 결정 후
    해당 구간으로 CubicSpline을 구성해 정적분.
    반환: 면적 [V*s], 실패 시 np.nan
    """
    _10p = AREA_THR_LO * pmax_fit
    _20p = AREA_THR_HI * pmax_fit
    n    = len(v)

    # 왼쪽 경계: imax에서 왼쪽으로 역탐색
    t10_left = t20_left = None
    for j in range(imax, 0, -1):
        if t20_left is None and v[j] <= _20p:
            t20_left = xlinear_inter(t[j], v[j], t[j+1], v[j+1], _20p)
        if t10_left is None and v[j] <= _10p:
            t10_left = xlinear_inter(t[j], v[j], t[j+1], v[j+1], _10p)
        if t10_left is not None and t20_left is not None:
            break

    if t10_left is None or t20_left is None:
        return np.nan

    start_time = xlinear_inter(t10_left, _10p, t20_left, _20p, 0.0)

    # 오른쪽 경계: imax에서 오른쪽으로 탐색
    t10_right = t20_right = None
    for j in range(imax, n - 1):
        if t20_right is None and v[j] <= _20p:
            t20_right = xlinear_inter(t[j], v[j], t[j-1], v[j-1], _20p)
        if t10_right is None and v[j] <= _10p:
            t10_right = xlinear_inter(t[j], v[j], t[j-1], v[j-1], _10p)
        if t10_right is not None and t20_right is not None:
            break

    if t10_right is None or t20_right is None:
        return np.nan

    end_time = xlinear_inter(t10_right, _10p, t20_right, _20p, 0.0)

    if (np.isinf(start_time) or np.isinf(end_time) or
            np.isnan(start_time) or np.isnan(end_time) or
            start_time >= end_time):
        return np.nan

    # start_time/end_time 구간으로 CubicSpline 구성 (각 1샘플 여유)
    i0 = max(0,     int(np.searchsorted(t, start_time)) - 1)
    i1 = min(n,     int(np.searchsorted(t, end_time))   + 2)

    if i1 - i0 >= 4:
        cs = CubicSpline(t[i0:i1], v[i0:i1])
        return float(cs.integrate(start_time, end_time))

    # Simpson fallback
    if i1 <= i0 + 1:
        return np.nan
    return float(integrate.simpson(v[i0:i1], x=t[i0:i1]))


def get_rising_cfd(v_fine, t_fine, pmax_fit, frac, n_fit=CFD_DVDT_N_FIT):
    """
    backward search: 피크에서 왼쪽으로 역탐색해 CFD crossing 반환.
    dvdt는 crossing 근방 n_fit 샘플 선형 회귀로 계산해 ADC 양자화 이산성 제거.
    반환: (t_cfd [s], dvdt [V/s]) — 실패 시 (np.nan, np.nan)
    """
    peak_idx = int(np.argmax(v_fine))
    level = frac * pmax_fit
    below = np.where(v_fine[:peak_idx] < level)[0]
    if len(below) == 0:
        return np.nan, np.nan
    i0 = int(below[-1])
    if i0 + 1 >= len(v_fine):
        return np.nan, np.nan
    dv = v_fine[i0+1] - v_fine[i0]
    dtf = t_fine[i0+1] - t_fine[i0]
    if abs(dv) < 1e-30:
        return np.nan, np.nan
    t_cfd = t_fine[i0] + (level - v_fine[i0]) * dtf / dv

    # dV/dt: crossing 근방 n_fit 샘플 선형 회귀 (단일 차분의 ADC 양자화 아티팩트 제거)
    half = n_fit // 2
    lo = max(0, i0 - half)
    hi = min(len(v_fine), i0 + half + 2)
    if hi - lo >= 3:
        coeffs = np.polyfit(t_fine[lo:hi], v_fine[lo:hi], 1)
        dvdt = float(coeffs[0])
    else:
        dvdt = dv / dtf
    return float(t_cfd), float(dvdt)


def get_falling_cfd(v_fine, t_fine, pmax_fit, frac):
    """
    falling edge CFD: 피크 이후에서 threshold 이하로 처음 떨어지는 교점.
    반환: t_cfd [s] — 실패 시 np.nan
    """
    peak_idx = int(np.argmax(v_fine))
    level = frac * pmax_fit
    post = v_fine[peak_idx:]
    below = np.where(post < level)[0]
    if len(below) == 0:
        return np.nan
    i_rel = int(below[0])
    i0 = peak_idx + i_rel - 1
    if i0 < 0 or i0 + 1 >= len(v_fine):
        return np.nan
    dv = v_fine[i0+1] - v_fine[i0]
    dtf = t_fine[i0+1] - t_fine[i0]
    if abs(dv) < 1e-30:
        return np.nan
    t_cfd = t_fine[i0] + (level - v_fine[i0]) * dtf / dv
    return float(t_cfd)


# =============================================================================
# 클리핑 감지
# =============================================================================

def find_adc_rails(file_path, chunk_size=1000, fmt="knu"):
    """전체 파일을 스캔해 CH1·CH2 raw ADC 절댓값 레일(최대·최소)을 반환.
    반환: (ch1_max, ch1_min, ch2_max, ch2_min)  [raw V 단위]
    """
    tree_key = "Events" if fmt == "knu" else "wfm"
    ch1_key  = "ch1"    if fmt == "knu" else "w1"
    ch2_key  = "ch2"    if fmt == "knu" else "w2"
    ch1_max, ch1_min = -np.inf,  np.inf
    ch2_max, ch2_min = -np.inf,  np.inf
    for batch in uproot.iterate(f"{file_path}:{tree_key}", [ch1_key, ch2_key],
                                step_size=chunk_size, library="np"):
        if fmt == "torino":
            ch1 = np.array([np.asarray(x) for x in batch[ch1_key]], dtype=np.float64)
            ch2 = np.array([np.asarray(x) for x in batch[ch2_key]], dtype=np.float64)
        else:
            ch1 = batch[ch1_key]
            ch2 = batch[ch2_key]
        ch1_max = max(ch1_max, float(ch1.max()))
        ch1_min = min(ch1_min, float(ch1.min()))
        ch2_max = max(ch2_max, float(ch2.max()))
        ch2_min = min(ch2_min, float(ch2.min()))
    return ch1_max, ch1_min, ch2_max, ch2_min


def batch_clip_mask(ch_raw, rail_max, rail_min):
    """(N_events,) bool — True if the raw waveform peak/trough reaches the
    global ADC rail value (= oscilloscope vertical clipping).
    rail_max / rail_min: 전체 파일의 절대 max/min (find_adc_rails 로 취득).
    """
    ev_max = ch_raw.max(axis=1)
    ev_min = ch_raw.min(axis=1)
    return (ev_max >= rail_max) | (ev_min <= rail_min)


# =============================================================================
# 자동 컷 함수 (개선 6)
# =============================================================================

def auto_pmax_cut(pmax_arr, fallback_low=AUTOCUT_PMAX_FALLBACK_MV):
    """
    pmax 분포에서 noise/signal 경계 valley 자동 탐지.
    알고리즘: global max(노이즈 피크) 오른쪽 구간에서 argmin → valley.
    반환: (cut_low [mV], success)
    """
    arr = pmax_arr[np.isfinite(pmax_arr) & (pmax_arr > 0)]
    if len(arr) < 10:
        return fallback_low, False

    counts, bins = np.histogram(arr, bins=AUTOCUT_PMAX_BINS, range=(arr.min(), arr.max()))
    bc = (bins[:-1] + bins[1:]) / 2

    # Step 1: global maximum = 노이즈 피크
    global_max_idx = int(np.argmax(counts))
    x_noise_peak = float(bc[global_max_idx])

    # Step 2: 노이즈 피크 오른쪽에서 valley 탐색 (argmin)
    right = counts[global_max_idx + 1:]
    if len(right) > 2:
        valley_rel = int(np.argmin(right))
        valley_idx = global_max_idx + 1 + valley_rel
        if counts[valley_idx] < counts[global_max_idx] * AUTOCUT_VALLEY_RATIO:
            return float(bc[valley_idx]), True

    # Step 3: fallback — x_noise_peak + 2×FWHM_noise (왼쪽 반폭 기준)
    half = counts[global_max_idx] / 2.0
    left_half = np.where(counts[:global_max_idx + 1] >= half)[0]
    if len(left_half) > 0:
        half_width = x_noise_peak - float(bc[left_half[0]])
        fwhm_est = 2 * half_width
        return x_noise_peak + AUTOCUT_FWHM_MULT * fwhm_est, False

    return fallback_low, False


def auto_tmax_cut(tmax_arr):
    """
    tmax 분포에서 최빈값(histogram peak) 기준 ±1 ns 윈도우 반환.
    반환: (cut_lo [ns], cut_hi [ns], success)
    """
    arr = tmax_arr[np.isfinite(tmax_arr)]
    counts, bins = np.histogram(arr, bins=AUTOCUT_TMAX_BINS)
    bc = (bins[:-1] + bins[1:]) / 2
    center = float(bc[int(np.argmax(counts))])
    return center - AUTOCUT_TMAX_HALF_NS, center + AUTOCUT_TMAX_HALF_NS, True


# =============================================================================
# Pass 1: 빠른 스캔
# =============================================================================

def scan_events(file_path, chunk_size=1000, ch1_negative=False,
                mcp_cut_mV=0.0, tmax_lo_ns=None, tmax_hi_ns=None,
                clip_cut=True, adc_rails=None):
    """
    Pass 1: 파일 전체를 청크 단위 벡터 연산으로 스캔.
    spline / Gaussian fit 없이 raw pmax·tmax만 수집.
    mcp_cut_mV > 0 이면 MCP pmax 컷을 Pass 1에서 적용.
    tmax_lo_ns / tmax_hi_ns 가 주어지면 (수동 컷) tmax 윈도우를 Pass 1에서 적용.
    clip_cut=True 이면 양 채널 클리핑 이벤트 제거.
    adc_rails: (ch1_max, ch1_min, ch2_max, ch2_min) — None 이면 자동 탐색.
    반환: (time_arr, pmax_all [mV], tmax_all [ns], pmax2_all [mV])
    """
    with uproot.open(file_path) as f:
        time_arr = f["Events"]["time"].array(library="np")[0]

    pmax_list  = []
    tmax_list  = []
    pmax2_list = []
    pmax1_clipped_list = []
    pmax2_clipped_list = []
    n_total = 0
    n_clipped = 0
    mcp_cut_V = mcp_cut_mV / 1e3

    if clip_cut:
        if adc_rails is None:
            print(f"[Pass 1] ADC 레일 탐색 중...")
            adc_rails = find_adc_rails(file_path, chunk_size)
        ch1_rmax, ch1_rmin, ch2_rmax, ch2_rmin = adc_rails
        ch1_range = (ch1_rmax - ch1_rmin) * 1e3
        ch2_range = (ch2_rmax - ch2_rmin) * 1e3
        ch1_vdiv  = ch1_range / 10
        ch2_vdiv  = ch2_range / 10
        ch1_lsb   = ch1_range / 256
        ch2_lsb   = ch2_range / 256
        print(f"[Pass 1] ADC 레일: CH1 [{ch1_rmin*1e3:.1f}, {ch1_rmax*1e3:.1f}] mV"
              f"  range {ch1_range:.1f} mV  {ch1_vdiv:.1f} mV/div  {ch1_lsb:.3f} mV/count")
        print(f"         ADC 레일: CH2 [{ch2_rmin*1e3:.1f}, {ch2_rmax*1e3:.1f}] mV"
              f"  range {ch2_range:.1f} mV  {ch2_vdiv:.1f} mV/div  {ch2_lsb:.3f} mV/count")

    print(f"[Pass 1] 스캔 시작: {file_path}")

    for batch in uproot.iterate(f"{file_path}:Events", ["ch1", "ch2"],
                                 step_size=chunk_size, library="np"):
        ch1 = batch["ch1"].astype(np.float64)
        ch2 = batch["ch2"].astype(np.float64)
        n_total += len(ch1)

        ped1 = ch1[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        ped2 = ch2[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        v1 = -(ch1 - ped1) if ch1_negative else (ch1 - ped1)
        v2 = -(ch2 - ped2)

        pmax1    = v1.max(axis=1)
        pmax2    = v2.max(axis=1)
        imax1    = v1.argmax(axis=1)
        tmax1_ns = time_arr[imax1] * 1e9

        loose = (pmax1 >= LOOSE_THR_CH1) & (pmax2 >= LOOSE_THR_CH2)
        if mcp_cut_V > 0:
            loose &= (pmax2 >= mcp_cut_V)
        if tmax_lo_ns is not None and tmax_hi_ns is not None:
            loose &= (tmax1_ns >= tmax_lo_ns) & (tmax1_ns <= tmax_hi_ns)
        if clip_cut:
            clipped = (batch_clip_mask(ch1, ch1_rmax, ch1_rmin) |
                       batch_clip_mask(ch2, ch2_rmax, ch2_rmin))
            n_clipped += int(clipped.sum())
            loose_and_clipped = loose & clipped
            if loose_and_clipped.sum() > 0:
                pmax1_clipped_list.append(pmax1[loose_and_clipped] * 1e3)
                pmax2_clipped_list.append(pmax2[loose_and_clipped] * 1e3)
            loose &= ~clipped

        if loose.sum() > 0:
            pmax_list.append(pmax1[loose] * 1e3)
            tmax_list.append(tmax1_ns[loose])
            pmax2_list.append(pmax2[loose] * 1e3)

    pmax_all  = np.concatenate(pmax_list)  if pmax_list  else np.array([])
    tmax_all  = np.concatenate(tmax_list)  if tmax_list  else np.array([])
    pmax2_all = np.concatenate(pmax2_list) if pmax2_list else np.array([])
    pmax1_clipped_all = np.concatenate(pmax1_clipped_list) if pmax1_clipped_list else np.array([])
    pmax2_clipped_all = np.concatenate(pmax2_clipped_list) if pmax2_clipped_list else np.array([])

    clip_str = f", 클리핑 제거: {n_clipped}" if clip_cut else ""
    print(f"[Pass 1] 완료: {n_total} 이벤트 스캔{clip_str}, loose 통과: {len(pmax_all)}")
    return time_arr, pmax_all, tmax_all, pmax2_all, pmax1_clipped_all, pmax2_clipped_all


# =============================================================================
# Pass 2: 정밀 분석
# =============================================================================

def process_events(file_path, time_arr, pmax_cut_mV, tmax_lo_ns, tmax_hi_ns, chunk_size=1000,
                   ch1_negative=False, pmax_cut_ch2_mV=0.0, clip_cut=True, adc_rails=None,
                   pmax_cut_hi_mV=0.0, pmax_cut_ch2_hi_mV=0.0):
    """
    Pass 2: 컷 통과 이벤트에 대해 Gaussian fit + CubicSpline + CFD 수행.
    noise_rms 컷은 main()에서 mask로 적용 (비교 플롯 생성을 위해 모든 이벤트 저장).
    clip_cut=True 이면 양 채널 클리핑 이벤트 제거.
    adc_rails: (ch1_max, ch1_min, ch2_max, ch2_min) — None 이면 자동 탐색.
    pmax_cut_hi_mV: LGAD pmax 상한 컷 [mV] (0 = 비활성).
    pmax_cut_ch2_hi_mV: MCP pmax 상한 컷 [mV] (0 = 비활성).
    반환: list of dict (records)
    """
    dt = float(time_arr[1] - time_arr[0])
    pmax_cut_V      = pmax_cut_mV      / 1e3
    pmax_cut_hi_V   = pmax_cut_hi_mV   / 1e3
    pmax_cut_ch2_hi_V = pmax_cut_ch2_hi_mV / 1e3
    records = []
    n_total = 0
    n_clipped = 0
    n_clipped_passed_pmax = 0

    if clip_cut:
        if adc_rails is None:
            print(f"[Pass 2] ADC 레일 탐색 중...")
            adc_rails = find_adc_rails(file_path, chunk_size)
        ch1_rmax, ch1_rmin, ch2_rmax, ch2_rmin = adc_rails

    print(f"[Pass 2] 정밀 분석 시작")
    hi_str = f" ~ {pmax_cut_hi_mV:.0f}" if pmax_cut_hi_mV > 0 else "+"
    print(f"         컷: pmax [{pmax_cut_mV:.0f}{hi_str}] mV, tmax {tmax_lo_ns:.3f}~{tmax_hi_ns:.3f} ns")
    if clip_cut:
        print(f"         클리핑 컷: 활성 (글로벌 ADC 레일 도달 이벤트 제거)")

    for batch in uproot.iterate(f"{file_path}:Events", ["ch1", "ch2"],
                                  step_size=chunk_size, library="np"):
        ch1 = batch["ch1"].astype(np.float64)
        ch2 = batch["ch2"].astype(np.float64)
        n_batch = len(ch1)
        n_total += n_batch

        # 청크 전체 벡터 연산으로 컷 마스크 계산
        ped1 = ch1[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        ped2 = ch2[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        v1_all = -(ch1 - ped1) if ch1_negative else (ch1 - ped1)
        v2_all = -(ch2 - ped2)

        pmax1_all   = v1_all.max(axis=1)
        pmax2_all   = v2_all.max(axis=1)
        imax1_all   = v1_all.argmax(axis=1)
        tmax1_ns_all = time_arr[imax1_all] * 1e9   # s → ns

        pmax_cut_ch2_V = pmax_cut_ch2_mV / 1e3
        cut = ((pmax1_all   >= LOOSE_THR_CH1) &
               (pmax2_all   >= LOOSE_THR_CH2) &
               (pmax1_all   >= pmax_cut_V) &
               (pmax2_all   >= pmax_cut_ch2_V) &
               (tmax1_ns_all >= tmax_lo_ns) &
               (tmax1_ns_all <= tmax_hi_ns))
        if pmax_cut_hi_V > 0:
            cut &= (pmax1_all <= pmax_cut_hi_V)
        if pmax_cut_ch2_hi_V > 0:
            cut &= (pmax2_all <= pmax_cut_ch2_hi_V)
        if clip_cut:
            clipped = (batch_clip_mask(ch1, ch1_rmax, ch1_rmin) |
                       batch_clip_mask(ch2, ch2_rmax, ch2_rmin))
            n_clipped += int(clipped.sum())
            n_clipped_passed_pmax += int((cut & clipped).sum())
            cut &= ~clipped

        for i in np.where(cut)[0]:
            v1 = v1_all[i]
            v2 = v2_all[i]
            noise1_rms = float(np.std(v1[:BASELINE_SAMPLES]))
            noise2_rms = float(np.std(v2[:BASELINE_SAMPLES]))

            # ============================================================
            # CH1 LGAD 분석
            # ============================================================
            imax1 = int(imax1_all[i])
            pmax1, tmax1 = gaussian_peak_fit(v1, time_arr, imax1, dt)

            if pmax1 <= 0:
                continue

            # CFD: raw data + linear interpolation
            # σ_jitter (중앙값 ~30 ps) >> 선형 보간 오차 (~1-5 ps) 이므로
            # CubicSpline fine grid는 timing 정밀도 향상에 실질적 기여 없음
            toa_rise1  = {}
            dvdt_rise1 = {}
            for frac in CFD_FRACS + [0.9]:
                t_c, dv_c = get_rising_cfd(v1, time_arr, pmax1, frac)
                toa_rise1[frac]  = t_c
                dvdt_rise1[frac] = dv_c

            t10 = toa_rise1.get(0.1, np.nan)
            t90 = toa_rise1.get(0.9, np.nan)
            rise_time = (t90 - t10) * 1e12 if (np.isfinite(t10) and np.isfinite(t90)) else np.nan

            toa_fall1 = {}
            for frac in CFD_FRACS:
                toa_fall1[frac] = get_falling_cfd(v1, time_arr, pmax1, frac)

            charge_vs = compute_area_new(v1, time_arr, imax1, pmax1)

            # ============================================================
            # CH2 MCP 분석 (ToA만 필요)
            # ============================================================
            imax2 = int(np.argmax(v2))
            pmax2, _ = gaussian_peak_fit(v2, time_arr, imax2, dt)

            if pmax2 <= 0:
                continue

            toa_rise2 = {}
            for frac in CFD_FRACS:
                t_c2, _ = get_rising_cfd(v2, time_arr, pmax2, frac)
                toa_rise2[frac] = t_c2

            # ============================================================
            # Record 구성
            # ============================================================
            rec = {
                "pmax"      : pmax1 * 1e3,
                "pmax_mcp"     : pmax2 * 1e3,
                "tmax"         : tmax1 * 1e9,
                "noise_rms"    : noise1_rms * 1e3,
                "noise_rms_mcp": noise2_rms * 1e3,
                "charge"    : charge_vs * 1e12 if np.isfinite(charge_vs) else np.nan,
                "rise_time" : rise_time,
            }

            for frac in CFD_FRACS:
                key = int(frac * 100)
                t1c = toa_rise1[frac]
                t2c = toa_rise2[frac]
                dv  = dvdt_rise1[frac]
                tf  = toa_fall1[frac]

                rec[f"toa_{key}"]         = t1c * 1e9 if np.isfinite(t1c) else np.nan
                rec[f"dt_{key}"]          = (t2c - t1c) * 1e12 if (np.isfinite(t1c) and np.isfinite(t2c)) else np.nan
                rec[f"dvdt_{key}"]        = dv / 1e6 if np.isfinite(dv) else np.nan
                rec[f"pulse_width_{key}"] = (tf - t1c) * 1e12 if (np.isfinite(tf) and np.isfinite(t1c)) else np.nan

            records.append(rec)

    if clip_cut:
        pmax_frac = 100.0 * n_clipped_passed_pmax / n_clipped if n_clipped > 0 else 0.0
        clip_str = (f", 클리핑 제거: {n_clipped}"
                    f"  (pmax/tmax 컷 통과 후 제거: {n_clipped_passed_pmax} [{pmax_frac:.1f}%])")
    else:
        clip_str = ""
    print(f"[Pass 2] 완료: {len(records)} / {n_total} 이벤트 분석{clip_str}")
    return records


def records_to_arrays(records):
    """list of dict → dict of np.ndarray"""
    if not records:
        return {}
    keys = records[0].keys()
    return {k: np.array([r[k] for r in records], dtype=np.float64) for k in keys}


# =============================================================================
# 플랏 함수
# =============================================================================

def plot_toa_dist(data, cut_mask, plot_label, file_tag, plot_dir,
                  cfd_keys=(20, 30), n_bins=200):
    """
    LGAD / MCP 각 CFD 분율별 Time of Arrival 분포 (Gaussian fit).
    MCP ToA = toa_lgad [ns] + dt [ps] / 1e3
    2행(LGAD / MCP) × len(cfd_keys)열 레이아웃.
    """
    n_col = len(cfd_keys)
    fig, axes = plt.subplots(2, n_col, figsize=(7 * n_col, 9), sharey="row")
    if n_col == 1:
        axes = axes.reshape(2, 1)
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    for col, key in enumerate(cfd_keys):
        toa_key = f"toa_{key}"
        dt_key  = f"dt_{key}"
        if toa_key not in data or dt_key not in data:
            continue

        lgad_toa = data[toa_key][cut_mask]          # ns
        dt_ps    = data[dt_key][cut_mask]            # ps
        mcp_toa  = lgad_toa + dt_ps / 1e3           # ns

        for row, (arr, label, color) in enumerate([
            (lgad_toa, "LGAD CH1", "royalblue"),
            (mcp_toa,  "MCP CH2",  "tomato"),
        ]):
            ax = axes[row, col]
            valid = arr[np.isfinite(arr)]
            if len(valid) < 10:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center")
                continue

            mu0, s0 = float(np.mean(valid)), float(np.std(valid))
            lo, hi  = mu0 - 4 * s0, mu0 + 4 * s0
            counts, edges = np.histogram(valid, bins=n_bins, range=(lo, hi))
            bc = (edges[:-1] + edges[1:]) / 2

            ax.bar(bc, counts, width=(edges[1] - edges[0]),
                   color=color, alpha=0.55, label=label)

            # Gaussian fit
            try:
                popt, _ = curve_fit(
                    gaussian, bc, counts.astype(float),
                    p0=[counts.max(), mu0, s0],
                    bounds=([0, lo, 1e-4], [counts.max() * 5, hi, s0 * 5]),
                    maxfev=2000,
                )
                x_fit = np.linspace(lo, hi, 500)
                ax.plot(x_fit, gaussian(x_fit, *popt),
                        color="k", lw=1.6, ls="--")
                mu_fit, sig_fit = popt[1], abs(popt[2])
                stat_txt = (f"mean = {mu_fit:.3f} ns\n"
                            f"$\\sigma$ = {sig_fit*1e3:.1f} ps\n"
                            f"N = {len(valid):,}")
            except Exception:
                mu_fit, sig_fit = mu0, s0
                stat_txt = (f"mean = {mu_fit:.3f} ns\n"
                            f"$\\sigma$ = {sig_fit*1e3:.1f} ps  (RMS)\n"
                            f"N = {len(valid):,}")

            ax.text(0.97, 0.95, stat_txt, transform=ax.transAxes,
                    fontsize=11, ha="right", va="top",
                    bbox=dict(boxstyle="round", fc="white", alpha=0.8))
            ax.set_title(f"{label}  CFD {key}%", fontsize=13)
            ax.set_xlabel("ToA [ns]")
            ax.set_ylabel("Events")
            ax.legend(fontsize=10)

    ax00 = axes[0, 0]
    ax00.text(0.02, 1.06, plot_label, transform=ax00.transAxes,
              fontsize=13, fontweight="bold", va="bottom")
    axes[0, -1].text(1.0, 1.06, r"$\beta$-Test @KNU",
                     transform=axes[0, -1].transAxes,
                     fontsize=12, ha="right", va="bottom")
    fig.suptitle("Time of Arrival distribution", fontsize=14)

    plt.tight_layout()
    out = os.path.join(plot_dir, f"toa_dist_{file_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → toa_dist_{file_tag}.png")


def plot_pmax_dist(data, cut_low, voltage, plot_dir):
    arr = data["pmax"]
    ok  = np.isfinite(arr)
    mask_cut = ok & (arr >= cut_low)

    _, bin_edges = np.histogram(arr[ok], bins=100)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(arr[ok],       bins=bin_edges, histtype="step",       lw=1.5, color="steelblue", label="All events")
    ax.hist(arr[mask_cut], bins=bin_edges, histtype="stepfilled", alpha=0.5, color="steelblue", label="After cut")
    ax.axvline(cut_low, color="red", ls="--", lw=1.5, label=f"Auto cut = {cut_low:.0f} mV")
    ax.set_xlabel("Peak Amplitude (Gaussian fit) [mV]")
    ax.set_ylabel("Entries")
    ax.set_yscale("log")
    ax.legend(fontsize=13)
    ax.text(0.02, 1.01, f"LGAD BV-{voltage}V", transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes, fontsize=14, ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"pmax_dist_{voltage}V.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_mcp_pmax_dist(pmax2_all, mcp_lo, mcp_hi, plot_label, file_tag, plot_dir,
                       pmax1_all=None, lgad_pmax_cut=0.0,
                       pmax1_clipped=None, pmax2_clipped=None):
    """
    MCP(CH2) pmax 분포 — 최대 4단계:
      Layer 1 (gray step)       : 전체 이벤트 (LGAD pmax cut 이전, clipped 포함)
      Layer 2 (steelblue step)  : LGAD pmax cut 이후, clip cut 이전
      Layer 3 (royalblue fill)  : + clip cut  →  Layer 2 vs 3 = clipping 효과
      Layer 4 (tomato fill)     : + MCP pmax cut  (mcp_lo > 0 일 때만)
    """
    fin2 = np.isfinite(pmax2_all)
    if pmax1_all is not None:
        fin2 &= np.isfinite(pmax1_all)
    p2 = pmax2_all[fin2]
    p1 = pmax1_all[fin2] if pmax1_all is not None else None

    if len(p2) == 0:
        return

    # ── Layer 1: 전체 이벤트 (LGAD pmax cut 이전, clipped 포함) ─────────────
    have_clipped = (pmax1_clipped is not None and len(pmax1_clipped) > 0
                    and pmax2_clipped is not None and len(pmax2_clipped) > 0)
    if have_clipped:
        p2_all_clipped = pmax2_clipped[np.isfinite(pmax2_clipped)]
    else:
        p2_all_clipped = np.array([])
    p2_before_pmax_cut = (np.concatenate([p2, p2_all_clipped])
                          if len(p2_all_clipped) > 0 else p2.copy())
    n_all = len(p2_before_pmax_cut)

    # ── LGAD pmax cut 마스크 ─────────────────────────────────────────────────
    lgad_mask = (p1 >= lgad_pmax_cut) if (p1 is not None and lgad_pmax_cut > 0) else np.ones(len(p2), dtype=bool)

    # ── Layer 2: LGAD pmax cut 이후, clip cut 이전 ───────────────────────────
    if have_clipped:
        clip_lgad_mask = (pmax1_clipped >= lgad_pmax_cut) if lgad_pmax_cut > 0 else np.ones(len(pmax1_clipped), dtype=bool)
        p2_clipped_sig = pmax2_clipped[clip_lgad_mask & np.isfinite(pmax2_clipped)]
    else:
        p2_clipped_sig = np.array([])
    p2_before_clip = (np.concatenate([p2[lgad_mask], p2_clipped_sig])
                      if len(p2_clipped_sig) > 0 else p2[lgad_mask].copy())
    n_before_clip = len(p2_before_clip)

    # ── Layer 3: + clip cut ──────────────────────────────────────────────────
    n_after_clip  = int(lgad_mask.sum())
    n_clipped_sig = len(p2_clipped_sig)
    clip_frac     = 100.0 * n_clipped_sig / n_before_clip if n_before_clip > 0 else 0.0

    # ── Layer 4 (optional): + MCP pmax cut ──────────────────────────────────
    mcp_mask = lgad_mask.copy()
    if mcp_lo > 0:
        mcp_mask &= (p2 >= mcp_lo)
    if mcp_hi > 0:
        mcp_mask &= (p2 <= mcp_hi)

    # 빈 범위: 전체 분포 (pmax cut 이전) 기반
    _, bin_edges = np.histogram(p2_before_pmax_cut, bins=100)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Layer 1: 전체 (LGAD pmax cut 이전, clipped 포함)
    ax.hist(p2_before_pmax_cut, bins=bin_edges, histtype="step", lw=1.5, color="gray", alpha=0.8,
            label=f"All events  (N={n_all:,})")

    # Layer 2: LGAD pmax cut 이후, clip cut 이전
    ax.hist(p2_before_clip, bins=bin_edges, histtype="step", lw=1.8, color="steelblue",
            label=f"LGAD pmax ≥ {lgad_pmax_cut:.0f} mV  (N={n_before_clip:,})")

    # Layer 3: + clip cut (clipping 효과)
    clip_label = (f"+ clip cut  (N={n_after_clip:,},  -{n_clipped_sig:,} [{clip_frac:.1f}%] clipped)"
                  if n_clipped_sig > 0 else f"+ clip cut  (N={n_after_clip:,})")
    ax.hist(p2[lgad_mask], bins=bin_edges, histtype="stepfilled", alpha=0.45, color="royalblue",
            label=clip_label)

    # Layer 4: MCP pmax cut (mcp_lo > 0 일 때만)
    if mcp_lo > 0:
        ax.hist(p2[mcp_mask], bins=bin_edges, histtype="stepfilled", alpha=0.55, color="tomato",
                label=f"+ MCP cut [{mcp_lo:.0f}"
                      + (f"–{mcp_hi:.0f}" if mcp_hi > 0 else "+")
                      + f"] mV  (N={mcp_mask.sum():,})")
        ax.axvline(mcp_lo, color="red",     ls="--", lw=1.5, label=f"MCP lo = {mcp_lo:.0f} mV")
        if mcp_hi > 0:
            ax.axvline(mcp_hi, color="darkred", ls="--", lw=1.5, label=f"MCP hi = {mcp_hi:.0f} mV")

    ax.set_xlabel("MCP Peak Amplitude [mV]")
    ax.set_ylabel("Entries")
    ax.set_yscale("log")
    ax.legend(fontsize=11)
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes, fontsize=14, ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"mcp_pmax_dist_{file_tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_tmax_dist(tmax_all, cut_lo, cut_hi, voltage, plot_dir):
    """Pass 1의 raw tmax_all 배열 사용 (전체 분포 표시)."""
    arr = tmax_all[np.isfinite(tmax_all)]
    mask_cut = (arr >= cut_lo) & (arr <= cut_hi)

    _, bin_edges = np.histogram(arr, bins=100)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(arr,           bins=bin_edges, histtype="step",       lw=1.5, color="darkorange", label="All events")
    ax.hist(arr[mask_cut], bins=bin_edges, histtype="stepfilled", alpha=0.5, color="darkorange", label="After cut")
    ax.axvline(cut_lo, color="red", ls="--", lw=1.5, label=f"Cut lo = {cut_lo:.2f} ns")
    ax.axvline(cut_hi, color="red", ls="--", lw=1.5, label=f"Cut hi = {cut_hi:.2f} ns")
    ax.set_xlabel("Peak Time (raw) [ns]")
    ax.set_ylabel("Entries")
    ax.legend(fontsize=13)
    ax.text(0.02, 1.01, f"LGAD BV-{voltage}V", transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes, fontsize=14, ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"tmax_dist_{voltage}V.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_noise_rms(data, cut_mask, voltage, plot_dir, pmax_min=0.0):
    pmax_ok  = np.isfinite(data["pmax"]) & (data["pmax"] >= pmax_min)
    arr1     = data["noise_rms"][cut_mask & pmax_ok]
    arr1     = arr1[np.isfinite(arr1)]
    arr2     = data.get("noise_rms_mcp", np.array([]))
    arr2     = arr2[cut_mask & pmax_ok] if len(arr2) == len(cut_mask) else np.array([])
    arr2     = arr2[np.isfinite(arr2)]

    has_mcp  = len(arr2) > 0
    n_col    = 2 if has_mcp else 1
    fig, axes = plt.subplots(1, n_col, figsize=(9 * n_col, 6))
    if n_col == 1:
        axes = [axes]

    def _panel(ax, arr, label, color):
        ax.hist(arr, bins=80, histtype="stepfilled", alpha=0.5, color=color)
        ax.hist(arr, bins=80, histtype="step",       lw=1.5,   color=color)
        ax.set_xlabel("Noise RMS [mV]")
        ax.set_ylabel("Entries")
        ax.set_title(label, fontsize=13)
        ax.text(0.97, 0.95,
                f"Mean   = {np.mean(arr):.3f} mV\n"
                f"Median = {np.median(arr):.3f} mV\n"
                f"Std    = {np.std(arr):.3f} mV",
                transform=ax.transAxes, fontsize=12, ha="right", va="top",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    _panel(axes[0], arr1, f"LGAD CH1  BV-{voltage}V", "purple")
    if has_mcp:
        _panel(axes[1], arr2, f"MCP CH2  BV-{voltage}V", "tomato")

    axes[0].text(0.02, 1.01, f"LGAD BV-{voltage}V", transform=axes[0].transAxes,
                 fontsize=14, fontweight="bold", va="bottom")
    axes[-1].text(1.0, 1.01, r"$\beta$-Test @KNU", transform=axes[-1].transAxes,
                  fontsize=14, ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"noise_rms_{voltage}V.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_snr(data, cut_mask, voltage, plot_dir):
    pmax_arr  = data["pmax"][cut_mask]
    noise_arr = data["noise_rms"][cut_mask]
    ok  = np.isfinite(pmax_arr) & np.isfinite(noise_arr) & (noise_arr > 0)
    snr = pmax_arr[ok] / noise_arr[ok]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(snr, bins=80, histtype="stepfilled", alpha=0.5, color="darkorange")
    ax.hist(snr, bins=80, histtype="step",       lw=1.5,   color="darkorange")
    ax.set_xlabel("Signal-to-Noise Ratio (pmax / noise RMS)")
    ax.set_ylabel("Entries")
    ax.text(0.97, 0.95, f"Median = {np.median(snr):.1f}\nMean  = {np.mean(snr):.1f}",
            transform=ax.transAxes, fontsize=13, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.text(0.02, 1.01, f"LGAD BV-{voltage}V", transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes, fontsize=14, ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"snr_dist_{voltage}V.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_charge_dist(data, cut_mask, plot_label, file_tag, plot_dir):
    arr = data["charge"]
    ok  = np.isfinite(arr) & (arr > 0) & (arr < 2000)
    arr_cut = arr[ok & cut_mask]

    counts, bins = np.histogram(arr_cut, bins=80)
    bc   = (bins[:-1] + bins[1:]) / 2
    bw   = bins[1] - bins[0]
    errs = np.sqrt(counts)          # Poisson 통계 에러

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.fill_between(np.repeat(bins, 2)[1:-1],
                    np.repeat(counts, 2),
                    step="mid", color="green", alpha=0.25)
    ax.errorbar(bc, counts, yerr=errs,
                fmt="none", ecolor="green", elinewidth=1.2, capsize=2)
    ax.step(bins[:-1], counts, where="post", color="green", lw=1.5,
            label=f"N = {len(arr_cut)}")

    mpv = mpv_err = sigma_L = sigma_L_err = np.nan
    try:
        p0_mpv   = float(np.median(arr_cut))
        p0_sigma = float(np.std(arr_cut)) * 0.15
        p0_amp   = float(max(counts)) * p0_sigma * 2
        popt, pcov = curve_fit(landau_pdf, bc, counts,
                               p0=[p0_mpv, p0_sigma, p0_amp],
                               sigma=np.sqrt(np.maximum(counts, 1)),
                               absolute_sigma=True, maxfev=2000)
        perr      = np.sqrt(np.diag(pcov))
        mpv       = float(popt[0]);  mpv_err     = float(perr[0])
        sigma_L   = float(popt[1]);  sigma_L_err = float(perr[1])
        x_fine = np.linspace(bins[0], bins[-1], 500)
        ax.plot(x_fine, landau_pdf(x_fine, *popt), "r-", lw=2,
                label="Landau fit")
    except Exception as e:
        print(f"[WARNING] Landau fit 실패: {e}")

    # 통계 박스
    _n = len(arr_cut)
    stats_lines = [f"N = {_n}"]
    if np.isfinite(mpv):
        stats_lines += [
            f"MPV   = {mpv:.1f} ± {mpv_err:.1f} mV·ns",
            f"σ_L   = {sigma_L:.1f} ± {sigma_L_err:.1f} mV·ns",
        ]
    ax.text(0.97, 0.97, "\n".join(stats_lines),
            transform=ax.transAxes, fontsize=11, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey", alpha=0.85))

    ax.set_xlabel("Pulse Area [mV·ns]")
    ax.set_ylabel(f"Entries / {bw:.1f} mV·ns")
    ax.legend(fontsize=13)
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes, fontsize=14, ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"charge_dist_{file_tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def gauss(x, a, x0, s):
    return a * np.exp(-0.5 * ((x - x0) / s) ** 2)


def fit_sigma_dt(arr, hw=800):
    """dT 배열에 Gaussian fit → σ, σ_err [ps].  peak ±hw ps 창 피팅 (legacy/bin-plot용)."""
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return np.nan, np.nan
    med = float(np.median(arr))
    counts, bins = np.histogram(arr, bins=100, range=(med - 3000, med + 3000))
    bc = (bins[:-1] + bins[1:]) / 2
    peak_idx = int(np.argmax(counts))
    peak_c   = float(bc[peak_idx])
    mask = (bc >= peak_c - hw) & (bc <= peak_c + hw) & (counts > 0)
    if mask.sum() < 5:
        return np.nan, np.nan
    try:
        p0 = [float(counts[peak_idx]), peak_c, 100.0]
        popt, pcov = curve_fit(gauss, bc[mask], counts[mask], p0=p0,
                               sigma=np.sqrt(np.maximum(counts[mask], 1)),
                               absolute_sigma=True,
                               bounds=([0, peak_c-hw, 1], [np.inf, peak_c+hw, np.inf]))
        return float(abs(popt[2])), float(np.sqrt(pcov[2, 2]))
    except Exception:
        return np.nan, np.nan


def fit_sigma_dt_full(arr, n_bins=DT_HIST_BINS, half_range=DT_HIST_HALF_RANGE_PS):
    """전체 범위 Gaussian fit (auto_plot / plot_time_resolution 과 동일 방식).
    plot_mcp_threshold_scan 등 summary 플롯에서 plot_time_resolution 과 일관성 보장."""
    arr = arr[np.isfinite(arr)]
    if len(arr) < 20:
        return np.nan, np.nan
    med = float(np.median(arr))
    lo, hi = med - half_range, med + half_range
    counts, bins = np.histogram(arr, bins=n_bins, range=(lo, hi))
    bc = (bins[:-1] + bins[1:]) / 2
    try:
        peak_idx = int(np.argmax(counts))
        p0 = [float(counts[peak_idx]), float(bc[peak_idx]), 100.0]
        popt, pcov = curve_fit(gauss, bc, counts, p0=p0,
                               sigma=np.sqrt(np.maximum(counts, 1)),
                               absolute_sigma=True, maxfev=5000)
        return float(abs(popt[2])), float(np.sqrt(pcov[2, 2]))
    except Exception:
        return np.nan, np.nan


def plot_time_resolution_legacy(data, cut_mask, plot_label, file_tag, plot_dir):
    """[백업] 구 버전: bins=80, range=median±2000 ps, peak 창 피팅."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    axes_flat = axes.flatten()
    sigmas = {}

    for idx, frac in enumerate(CFD_FRACS):
        ax  = axes_flat[idx]
        key = int(frac * 100)
        arr = data[f"dt_{key}"]
        arr = arr[cut_mask & np.isfinite(arr)]

        if len(arr) < 10:
            ax.text(0.5, 0.5, "Insufficient Data", transform=ax.transAxes, ha="center")
            continue

        med = float(np.median(arr))
        rng = (med - 2000, med + 2000)
        ax.hist(arr, bins=80, range=rng, histtype="stepfilled", alpha=0.35, color="steelblue")
        counts, bins, _ = ax.hist(arr, bins=80, range=rng,
                                  histtype="step", lw=1.5, color="steelblue")
        bc = (bins[:-1] + bins[1:]) / 2

        sigma, sigma_err = fit_sigma_dt(arr)
        sigmas[key] = (sigma, sigma_err)

        if np.isfinite(sigma):
            try:
                peak_idx = int(np.argmax(counts))
                peak_c   = float(bc[peak_idx])
                hw = max(sigma * 3, 200)
                mask_fit = (bc >= peak_c - hw) & (bc <= peak_c + hw) & (counts > 0)
                p0 = [float(np.max(counts)), peak_c, sigma]
                popt, pcov = curve_fit(gauss, bc[mask_fit], counts[mask_fit], p0=p0,
                                       sigma=np.sqrt(np.maximum(counts[mask_fit], 1)),
                                       absolute_sigma=True)
                perr = np.sqrt(np.diag(pcov))
                sigma_p, sigma_p_err = abs(popt[2]), perr[2]

                y_exp = gauss(bc[mask_fit], *popt)
                chi2  = float(np.sum((counts[mask_fit] - y_exp) ** 2
                                     / np.maximum(counts[mask_fit], 1)))
                ndf   = int(mask_fit.sum()) - 3
                chi2_label = (f"χ²/ndf = {chi2:.1f}/{ndf} = {chi2/ndf:.2f}"
                              if ndf > 0 else "")

                x_fit = np.linspace(peak_c - hw, peak_c + hw, 400)
                ax.plot(x_fit, gauss(x_fit, *popt), "r-", lw=2,
                        label=(f"σ = {sigma_p:.1f} ± {sigma_p_err:.1f} ps\n"
                               f"N = {len(arr)}\n{chi2_label}"))
                sigmas[key] = (sigma_p, sigma_p_err)
            except Exception:
                ax.text(0.65, 0.8, "Fit Failed", transform=ax.transAxes, color="red")

        ax.set_xlabel(f"$\\Delta T = t_{{MCP}} - t_{{LGAD}}$ (CFD {key}%) [ps]")
        ax.set_ylabel("Entries")
        ax.legend(fontsize=14, loc="upper left")
        ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom")
        ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
                fontsize=12, ha="right", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"time_resolution_legacy_{file_tag}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    print("\n=== Time Resolution [legacy] ===")
    for key, (s, se) in sigmas.items():
        status = f"{s:.1f} ± {se:.1f} ps" if np.isfinite(s) else "Fit Failed"
        print(f"  CFD {key}%: {status}")

    return sigmas


def _plot_dt_rebin(arr, plot_label, file_tag, plot_dir, original_chi2ndf):
    """CFD20 dt 분포를 여러 binning으로 그려 비교. chi2/ndf가 불량할 때 호출."""
    bin_candidates = [50, 80, 120, 160]
    med = float(np.median(arr))
    lo_dt, hi_dt = med - 1000.0, med + 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(
        f"{plot_label}  —  CFD 20% $\\Delta T$ rebinning study"
        f"  (original bins=200, $\\chi^2$/ndf = {original_chi2ndf:.2f})",
        fontsize=14, fontweight="bold",
    )

    for ax, n_bins in zip(axes.flatten(), bin_candidates):
        counts, bins = np.histogram(arr, bins=n_bins, range=(lo_dt, hi_dt))
        bc = (bins[:-1] + bins[1:]) / 2
        ax.hist(arr, bins=n_bins, range=(lo_dt, hi_dt),
                histtype="step", lw=1.5, color="navy",
                label=f"bins = {n_bins}")

        try:
            peak_idx = int(np.argmax(counts))
            p0 = [float(counts[peak_idx]), float(bc[peak_idx]), 100.0]
            bin_err = np.sqrt(np.maximum(counts, 1))
            popt, pcov = curve_fit(gauss, bc, counts, p0=p0,
                                   sigma=bin_err, absolute_sigma=True,
                                   maxfev=5000)
            perr = np.sqrt(np.diag(pcov))
            mu_ps        = float(popt[1])
            sigma_ps     = float(abs(popt[2]))
            sigma_err_ps = float(perr[2])

            y_exp = gauss(bc, *popt)
            chi2  = float(np.sum((counts - y_exp) ** 2 / np.maximum(counts, 1)))
            ndf   = int((counts > 0).sum()) - 3
            chi2_ndf = chi2 / ndf if ndf > 0 else float("nan")

            x_fit = np.linspace(lo_dt, hi_dt, 600)
            ax.plot(x_fit, gauss(x_fit, *popt), "r-", lw=2,
                    label=(fr"$\mu$ = {mu_ps:.1f} ps,  "
                           fr"$\sigma$ = {sigma_ps:.1f} ± {sigma_err_ps:.1f} ps"
                           f"\nN = {len(arr)}"
                           f"\n$\\chi^2$/ndf = {chi2:.1f}/{ndf} = {chi2_ndf:.2f}"))
        except Exception:
            ax.text(0.65, 0.8, "Fit Failed", transform=ax.transAxes, color="red")

        ax.set_xlabel(r"$\Delta T = t_{\mathrm{MCP}} - t_{\mathrm{LGAD}}$ (CFD 20%) [ps]")
        ax.set_ylabel("Entries")
        ax.legend(fontsize=12, loc="upper left")
        ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")
        ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
                fontsize=11, ha="right", va="bottom")

    plt.tight_layout()
    out_path = os.path.join(plot_dir, f"time_resolution_cfd20_rebin_{file_tag}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → CFD20 rebin study saved: {out_path}")


def plot_time_resolution(data, cut_mask, plot_label, file_tag, plot_dir):
    """auto_plot 스타일: bins=200, range=median±1000 ps, 전체 범위 Gaussian 피팅."""
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    axes_flat = axes.flatten()
    sigmas = {}
    chi2ndf_per_key = {}   # chi2/ndf 추적 (rebin 판단용)

    for idx, frac in enumerate(CFD_FRACS):
        ax  = axes_flat[idx]
        key = int(frac * 100)
        arr = data[f"dt_{key}"]
        arr = arr[cut_mask & np.isfinite(arr)]

        if len(arr) < 10:
            ax.text(0.5, 0.5, "Insufficient Data", transform=ax.transAxes, ha="center")
            continue

        med = float(np.median(arr))
        lo_dt, hi_dt = med - DT_HIST_HALF_RANGE_PS, med + DT_HIST_HALF_RANGE_PS
        n_bins = DT_HIST_BINS

        counts, bins = np.histogram(arr, bins=n_bins, range=(lo_dt, hi_dt))
        bc = (bins[:-1] + bins[1:]) / 2
        ax.hist(arr, bins=n_bins, range=(lo_dt, hi_dt),
                histtype="step", lw=1.5, color="navy",
                label=f"$\\Delta T = t_{{MCP}} - t_{{LGAD}}$  (CFD {key}%)")

        sigmas[key] = (np.nan, np.nan)

        try:
            peak_idx = int(np.argmax(counts))
            p0 = [float(counts[peak_idx]), float(bc[peak_idx]), 100.0]
            bin_err = np.sqrt(np.maximum(counts, 1))
            popt, pcov = curve_fit(gauss, bc, counts, p0=p0,
                                   sigma=bin_err, absolute_sigma=True,
                                   maxfev=5000)
            perr = np.sqrt(np.diag(pcov))
            mu_ps    = float(popt[1])
            sigma_ps = float(abs(popt[2]))
            sigma_err_ps = float(perr[2])

            y_exp = gauss(bc, *popt)
            chi2  = float(np.sum((counts - y_exp) ** 2 / np.maximum(counts, 1)))
            ndf   = int((counts > 0).sum()) - 3
            chi2_ndf = chi2 / ndf if ndf > 0 else float("nan")
            chi2ndf_per_key[key] = chi2_ndf

            x_fit = np.linspace(lo_dt, hi_dt, 600)
            ax.plot(x_fit, gauss(x_fit, *popt), "r-", lw=2,
                    label=(fr"Gaussian:  $\mu$ = {mu_ps:.1f} ps,  "
                           fr"$\sigma$ = {sigma_ps:.1f} ± {sigma_err_ps:.1f} ps"
                           f"\nN = {len(arr)}"
                           f"\n$\\chi^2$/ndf = {chi2:.1f}/{ndf}"
                           f" = {chi2/ndf:.2f}" if ndf > 0 else ""))
            sigmas[key] = (sigma_ps, sigma_err_ps)
        except Exception:
            ax.text(0.65, 0.8, "Fit Failed", transform=ax.transAxes, color="red")

        ax.set_xlabel(f"$\\Delta T = t_{{MCP}} - t_{{LGAD}}$ (CFD {key}%) [ps]")
        ax.set_ylabel("Entries")
        ax.legend(fontsize=13, loc="upper left")
        ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom")
        ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
                fontsize=12, ha="right", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"time_resolution_{file_tag}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # CFD20 chi2/ndf 가 불량하면 (≥1.5 또는 <0.5) rebin 비교 플롯 추가 생성
    cfd20_chi2ndf = chi2ndf_per_key.get(20, float("nan"))
    if np.isfinite(cfd20_chi2ndf) and (cfd20_chi2ndf >= CHI2_NDF_HI or cfd20_chi2ndf < CHI2_NDF_LO):
        arr20 = data["dt_20"]
        arr20 = arr20[cut_mask & np.isfinite(arr20)]
        print(f"  [CFD20 chi2/ndf = {cfd20_chi2ndf:.2f}] → rebin study 생성 중...")
        _plot_dt_rebin(arr20, plot_label, file_tag, plot_dir, cfd20_chi2ndf)

    print("\n=== Time Resolution ===")
    for key, (s, se) in sigmas.items():
        status = f"{s:.1f} ± {se:.1f} ps" if np.isfinite(s) else "Fit Failed"
        print(f"  CFD {key}%: {status}")

    return sigmas


def plot_rise_time(data, cut_mask, voltage, plot_dir):
    arr = data["rise_time"]
    arr = arr[cut_mask & np.isfinite(arr) & (arr > 0) & (arr < 5000)]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(arr, bins=80, histtype="step", lw=1.5, color="teal")
    ax.set_xlabel("Rise Time (10% → 90%) [ps]")
    ax.set_ylabel("Entries")
    ax.text(0.6, 0.85, f"Mean = {np.mean(arr):.0f} ps\nRMS  = {np.std(arr):.0f} ps",
            transform=ax.transAxes, fontsize=14)
    ax.text(0.02, 1.01, f"LGAD BV-{voltage}V", transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes, fontsize=14, ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"rise_time_dist_{voltage}V.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_dvdt(data, cut_mask, voltage, plot_dir, rep_fracs=(20, 30, 50)):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["steelblue", "tomato", "seagreen"]
    for color, frac in zip(colors, rep_fracs):
        arr = data[f"dvdt_{frac}"]
        arr = arr[cut_mask & np.isfinite(arr) & (arr > 0)]
        if len(arr) == 0:
            continue
        ax.hist(arr, bins=80, histtype="step", lw=1.8, color=color, label=f"CFD {frac}%")
    ax.set_xlabel("dV/dt @ CFD crossing [mV/ns]")
    ax.set_ylabel("Entries")
    ax.legend(fontsize=13)
    ax.text(0.02, 1.01, f"LGAD BV-{voltage}V", transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes, fontsize=14, ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"dvdt_dist_{voltage}V.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_jitter_breakdown(data, cut_mask, voltage, plot_dir, sigmas):
    """
    CFD 분율별 σ_total vs σ_jitter vs σ_other 분해.
    σ_jitter = median(σ_noise / (dV/dt))
    σ_other  = sqrt(max(σ_total^2 - σ_jitter^2, 0))
    """
    fracs_pct = [int(f*100) for f in CFD_FRACS]
    s_total   = []
    s_jitter  = []
    s_other   = []

    for key in fracs_pct:
        sig, _ = sigmas.get(key, (np.nan, np.nan))
        s_total.append(sig)

        noise_arr = data["noise_rms"][cut_mask]
        dvdt_arr  = data[f"dvdt_{key}"][cut_mask]
        ok = np.isfinite(noise_arr) & np.isfinite(dvdt_arr) & (dvdt_arr > 0)
        if ok.sum() > 5:
            jitter_per_ev = noise_arr[ok] / dvdt_arr[ok]  # ns
            jitter_ps = float(np.median(jitter_per_ev)) * 1e3  # ns→ps
        else:
            jitter_ps = np.nan
        s_jitter.append(jitter_ps)

        s_other.append(
            float(np.sqrt(max(sig**2 - jitter_ps**2, 0)))
            if (np.isfinite(sig) and np.isfinite(jitter_ps)) else np.nan
        )

    x = np.array(fracs_pct, dtype=float)
    s_total  = np.array(s_total)
    s_jitter = np.array(s_jitter)
    s_other  = np.array(s_other)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, s_total,  "o-", lw=2, label="σ_total",  color="steelblue")
    ax.plot(x, s_jitter, "s--",lw=2, label="σ_jitter", color="tomato")
    ax.plot(x, s_other,  "^:", lw=2, label="σ_other",  color="seagreen")
    ax.set_xlabel("CFD Fraction [%]")
    ax.set_ylabel("σ [ps]")
    ax.legend(fontsize=13)
    ax.set_xticks(fracs_pct)
    ax.text(0.02, 1.01, f"LGAD BV-{voltage}V", transform=ax.transAxes, fontsize=14, fontweight="bold", va="bottom")
    ax.text(1.0,  1.01, r"$\beta$-Test @KNU (KNU improved)", transform=ax.transAxes,
            fontsize=12, ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"jitter_breakdown_{voltage}V.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_pmax_tmax_dist(pmax_all, tmax_all, pmax2_all, cut_low, cut_lo_t, cut_hi_t, mcp_cut,
                         plot_label, file_tag, plot_dir, pmax_bins=100, pmax_range=None,
                         cut_hi_p=0.0, mcp_cut_hi=0.0, pmax1_clipped=None):
    """Pmax(log) + Pmax(linear) + Tmax 분포 통합 (1×3)."""
    fig, axes = plt.subplots(1, 3, figsize=(27, 6))

    valid    = np.isfinite(pmax_all) & np.isfinite(tmax_all) & np.isfinite(pmax2_all)
    arr_p    = pmax_all[valid]
    arr_t    = tmax_all[valid]
    arr_p2   = pmax2_all[valid]
    mask_all = (arr_p  >= cut_low) & \
               (arr_t  >= cut_lo_t) & (arr_t <= cut_hi_t) & \
               (arr_p2 >= mcp_cut)
    if cut_hi_p > 0:
        mask_all &= (arr_p <= cut_hi_p)
    if mcp_cut_hi > 0:
        mask_all &= (arr_p2 <= mcp_cut_hi)

    # 클리핑 제거된 이벤트 중 pmax cut 통과분
    have_clipped = pmax1_clipped is not None and len(pmax1_clipped) > 0
    if have_clipped:
        p1c = pmax1_clipped[np.isfinite(pmax1_clipped)]
        clip_sig_mask = p1c >= cut_low
        if cut_hi_p > 0:
            clip_sig_mask &= p1c <= cut_hi_p
        p1c_sig = p1c[clip_sig_mask]
    else:
        p1c_sig = np.array([])

    # pmax cut 통과 (clip 제거 후) + pmax cut 통과 (클리핑 제거분) = clip 이전 신호 분포
    p_after_pmax  = arr_p[arr_p >= cut_low] if cut_hi_p == 0 else arr_p[(arr_p >= cut_low) & (arr_p <= cut_hi_p)]
    p_before_clip = np.concatenate([p_after_pmax, p1c_sig]) if len(p1c_sig) > 0 else p_after_pmax

    n_before_clip  = len(p_before_clip)
    n_after_clip   = len(p_after_pmax)
    n_clipped_sig  = len(p1c_sig)
    clip_frac      = 100.0 * n_clipped_sig / n_before_clip if n_before_clip > 0 else 0.0

    hist_range_p = tuple(pmax_range) if pmax_range is not None else None
    # bin range를 클리핑 이벤트 포함 기준으로 잡아야 rail 근처 pile-up이 보임
    _base_for_bins = p_before_clip if len(p_before_clip) > 0 else arr_p
    _, bin_edges_p = np.histogram(_base_for_bins, bins=pmax_bins, range=hist_range_p)

    # --- pmax: log scale (왼쪽) + linear scale (가운데) ---
    for ax, yscale in zip(axes[:2], ("log", "linear")):
        # layer 1: 전체 (clip 적용 후, pmax cut 없음) — 노이즈 포함 배경
        ax.hist(arr_p, bins=bin_edges_p, histtype="step", lw=1.2, color="gray", alpha=0.7,
                label=f"All events (after clip)  (N={len(arr_p):,})")
        # layer 2: pmax cut 통과, clip 포함 (클리핑 신호 합산)
        ax.hist(p_before_clip, bins=bin_edges_p, histtype="step", lw=1.8, color="steelblue",
                label=f"LGAD pmax cut (before clip)  (N={n_before_clip:,})")
        # layer 3: pmax cut 통과, clip 제거 후 (실제 분석 신호)
        clip_label = (f"+ clip cut  (N={n_after_clip:,},  -{n_clipped_sig:,} clipped [{clip_frac:.1f}%])"
                      if n_clipped_sig > 0 else f"+ clip cut  (N={n_after_clip:,})")
        ax.hist(p_after_pmax, bins=bin_edges_p, histtype="stepfilled", alpha=0.5, color="steelblue",
                label=clip_label)
        lo_label = f"Cut lo = {cut_low:.0f} mV" if cut_hi_p > 0 else f"Auto cut = {cut_low:.0f} mV"
        ax.axvline(cut_low, color="red", ls="--", lw=1.5, label=lo_label)
        if cut_hi_p > 0:
            ax.axvline(cut_hi_p, color="darkred", ls="--", lw=1.5, label=f"Cut hi = {cut_hi_p:.0f} mV")
        ax.set_xlabel("Peak Amplitude (Gaussian fit) [mV]")
        ax.set_ylabel("Entries")
        ax.set_yscale(yscale)
        if pmax_range is not None:
            ax.set_xlim(*pmax_range)
        ax.legend(fontsize=13)
        ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="bottom")
        ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
                fontsize=14, ha="right", va="bottom")

    # --- tmax (오른쪽) ---
    ax = axes[2]
    _, bin_edges_t = np.histogram(arr_t, bins=100)
    ax.hist(arr_t,              bins=bin_edges_t, histtype="step",       lw=1.5,    color="darkorange", label="All events")
    ax.hist(arr_t[mask_all],    bins=bin_edges_t, histtype="stepfilled", alpha=0.5, color="darkorange", label="After all cuts")
    ax.axvline(cut_lo_t, color="red", ls="--", lw=1.5, label=f"Cut lo = {cut_lo_t:.2f} ns")
    ax.axvline(cut_hi_t, color="red", ls="--", lw=1.5, label=f"Cut hi = {cut_hi_t:.2f} ns")
    ax.set_xlabel("Peak Time (raw) [ns]")
    ax.set_ylabel("Entries")
    ax.legend(fontsize=13)
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
            fontsize=14, fontweight="bold", va="bottom")
    ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
            fontsize=14, ha="right", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"pmax_tmax_dist_{file_tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_summary_dist(data, cut_mask, plot_label, file_tag, plot_dir, sigmas, rep_fracs=(20, 30, 50)):
    """noise_rms + dV/dt + rise_time + jitter breakdown 통합 (2×2)."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # [0,0] Noise RMS
    ax = axes[0, 0]
    arr = data["noise_rms"][cut_mask]
    arr = arr[np.isfinite(arr)]
    ax.hist(arr, bins=80, histtype="stepfilled", alpha=0.5, color="purple")
    ax.hist(arr, bins=80, histtype="step",       lw=1.5,   color="purple")
    ax.set_xlabel("Noise RMS [mV]")
    ax.set_ylabel("Entries")
    ax.text(0.97, 0.95, f"Mean = {np.mean(arr):.1f} mV\nStd  = {np.std(arr):.1f} mV",
            transform=ax.transAxes, fontsize=13, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
            fontsize=12, ha="right", va="bottom")

    # [0,1] dV/dt
    ax = axes[0, 1]
    colors_dv = ["steelblue", "tomato", "seagreen"]
    for color, frac in zip(colors_dv, rep_fracs):
        arr_dv = data[f"dvdt_{frac}"][cut_mask]
        arr_dv = arr_dv[np.isfinite(arr_dv) & (arr_dv > 0)]
        if len(arr_dv) == 0:
            continue
        label = f"CFD {frac}%  (mean={np.mean(arr_dv):.0f}, std={np.std(arr_dv):.0f} mV/ns)"
        ax.hist(arr_dv, bins=80, histtype="step", lw=1.8, color=color, label=label)
    ax.set_xlabel(r"$(dV/dt)\,|_\mathrm{CFD}$ [mV/ns]")
    ax.set_ylabel("Entries")
    ax.legend(fontsize=11)
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
            fontsize=12, ha="right", va="bottom")

    # [1,0] Rise Time
    ax = axes[1, 0]
    arr_rt = data["rise_time"][cut_mask]
    arr_rt = arr_rt[np.isfinite(arr_rt) & (arr_rt > 0) & (arr_rt < 5000)]
    ax.hist(arr_rt, bins=80, histtype="stepfilled", alpha=0.5, color="teal")
    ax.hist(arr_rt, bins=80, histtype="step",       lw=1.5,   color="teal")
    ax.set_xlabel("Rise Time (10% → 90%) [ps]")
    ax.set_ylabel("Entries")
    ax.text(0.97, 0.95, f"Mean = {np.mean(arr_rt):.0f} ps\nStd  = {np.std(arr_rt):.0f} ps",
            transform=ax.transAxes, fontsize=13, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
            fontsize=12, ha="right", va="bottom")

    # [1,1] Jitter Breakdown (그룹 막대)
    ax = axes[1, 1]
    fracs_pct = [int(f * 100) for f in CFD_FRACS]
    s_total  = []
    s_jitter = []
    s_other  = []

    for key in fracs_pct:
        sig, _ = sigmas.get(key, (np.nan, np.nan))
        s_total.append(sig)
        noise_arr = data["noise_rms"][cut_mask]
        dvdt_arr  = data[f"dvdt_{key}"][cut_mask]
        ok = np.isfinite(noise_arr) & np.isfinite(dvdt_arr) & (dvdt_arr > 0)
        if ok.sum() > 5:
            jitter_ps = float(np.median(noise_arr[ok] / dvdt_arr[ok])) * 1e3
        else:
            jitter_ps = np.nan
        s_jitter.append(jitter_ps)
        s_other.append(
            float(np.sqrt(max(sig**2 - jitter_ps**2, 0)))
            if (np.isfinite(sig) and np.isfinite(jitter_ps)) else np.nan
        )

    s_total  = np.array(s_total)
    s_jitter = np.array(s_jitter)
    s_other  = np.array(s_other)
    x     = np.arange(len(fracs_pct))
    width = 0.22

    ax.bar(x - width, s_total,  width, label="σ_total",  color="steelblue", alpha=0.8)
    ax.bar(x,         s_jitter, width, label="σ_jitter", color="tomato",    alpha=0.8)
    ax.bar(x + width, s_other,  width, label="σ_other",  color="seagreen",  alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}%" for f in fracs_pct])
    ax.set_xlabel("CFD Fraction [%]")
    ax.set_ylabel("σ [ps]")
    ax.legend(fontsize=13)
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
            fontsize=12, ha="right", va="bottom")

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"summary_dist_{file_tag}.png"), dpi=150, bbox_inches="tight")
    plt.close()


# =============================================================================
# Noise cut 비교 플롯
# =============================================================================

def plot_noise_cut_comparison(data, noise_mask, noise_cut_mV, plot_label, file_tag, plot_dir,
                              noise_sigma_bounds=None):
    """
    noise_rms cut 전·후 주요 변수 분포 비교 (2×3).
    회색 step: 전체 / 파란 filled: 통과 / 빨간 filled: 탈락
    noise_sigma_bounds: (lo, hi) [mV] — 시그마 컷 경계 (있을 때만 표시)
    """
    n_all  = int(len(noise_mask))
    n_pass = int(noise_mask.sum())
    n_fail = n_all - n_pass

    panels = [
        ("pmax",      "Peak Amplitude [mV]",        None, None),
        ("charge",    "Pulse Area [mV·ns]",            0,    None),
        ("rise_time", "Rise Time (10→90%) [ps]",     0,    3000),
        ("dvdt_20",   "dV/dt @ CFD20% [mV/ns]",      0,    None),
        ("dt_20",     r"$\Delta T$ CFD20% [ps]",     None, None),
        ("noise_rms", "Noise RMS [mV]",               0,    None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))

    for ax, (key, xlabel, xlo, xhi) in zip(axes.flatten(), panels):
        arr = data.get(key)
        if arr is None:
            ax.text(0.5, 0.5, f"키 없음: {key}", transform=ax.transAxes, ha="center")
            continue

        fin      = np.isfinite(arr)
        arr_all  = arr[fin]
        m_pass   = noise_mask[fin]
        arr_pass = arr_all[m_pass]
        arr_fail = arr_all[~m_pass]

        if len(arr_all) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        # x 범위
        if key == "dt_20":
            med = float(np.median(arr_all))
            lo  = xlo if xlo is not None else med - 2000
            hi  = xhi if xhi is not None else med + 2000
        else:
            p1, p99 = np.percentile(arr_all, 1), np.percentile(arr_all, 99)
            lo = xlo if xlo is not None else p1
            hi = xhi if xhi is not None else p99 * 1.05
        lo = max(lo, arr_all.min()) if xlo is None else lo
        hi = min(hi, arr_all.max()) if xhi is None else hi

        bins = np.linspace(lo, hi, 80)

        ax.hist(arr_all,  bins=bins, histtype="step",       color="grey",      lw=1.8,
                label=f"All  N={len(arr_all)}", zorder=2)
        ax.hist(arr_pass, bins=bins, histtype="stepfilled", color="steelblue", alpha=0.45, zorder=3)
        ax.hist(arr_pass, bins=bins, histtype="step",       color="steelblue", lw=1.5,
                label=f"Pass N={len(arr_pass)}", zorder=3)
        if len(arr_fail) > 0:
            ax.hist(arr_fail, bins=bins, histtype="stepfilled", color="tomato", alpha=0.40, zorder=4)
            ax.hist(arr_fail, bins=bins, histtype="step",       color="tomato", lw=1.5,
                    label=f"Fail N={len(arr_fail)}", zorder=4)

        if key == "noise_rms":
            if noise_cut_mV > 0:
                ax.axvline(noise_cut_mV, color="red", ls="--", lw=2,
                           label=f"Abs cut = {noise_cut_mV:.1f} mV")
            if noise_sigma_bounds is not None:
                ax.axvline(noise_sigma_bounds[0], color="darkorange", ls=":", lw=2,
                           label=f"σ lo = {noise_sigma_bounds[0]:.2f} mV")
                ax.axvline(noise_sigma_bounds[1], color="darkorange", ls=":", lw=2,
                           label=f"σ hi = {noise_sigma_bounds[1]:.2f} mV")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Entries")
        ax.legend(fontsize=11, loc="upper right")
        ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")
        ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
                fontsize=11, ha="right", va="bottom")

    cut_desc = []
    if noise_cut_mV > 0:
        cut_desc.append(f"abs ≤ {noise_cut_mV:.1f} mV")
    if noise_sigma_bounds is not None:
        cut_desc.append(f"σ [{noise_sigma_bounds[0]:.2f}, {noise_sigma_bounds[1]:.2f}] mV")
    cut_str = "  |  ".join(cut_desc) if cut_desc else "no cut"
    fig.suptitle(
        f"Noise RMS Cut Comparison  |  {cut_str}  |  "
        f"pass {n_pass}/{n_all} ({100*n_pass/n_all:.1f}%)",
        fontsize=14, y=1.01
    )
    plt.tight_layout()
    out = os.path.join(plot_dir, f"noise_cut_comparison_{file_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → noise_cut_comparison_{file_tag}.png")


def plot_mcp_cut_comparison(data, base_mask, mcp_mask, mcp_lo_mV, mcp_hi_mV,
                             plot_label, file_tag, plot_dir):
    """
    MCP pmax cut 전·후 주요 변수 분포 비교 (2×3).
    회색 step: MCP cut 없음 (base_mask만) / 파란 filled: MCP cut 통과 / 빨간 filled: 탈락
    """
    n_all  = int(base_mask.sum())
    n_pass = int((base_mask & mcp_mask).sum())
    n_fail = n_all - n_pass

    panels = [
        ("pmax",      "Peak Amplitude (LGAD) [mV]",      None, None),
        ("pmax_mcp",  "MCP Peak Amplitude [mV]",           0,    None),
        ("charge",    "Pulse Area [mV·ns]",                0,    None),
        ("rise_time", "Rise Time (10→90%) [ps]",          0,    3000),
        ("dvdt_20",   r"$dV/dt$ @ CFD20% [mV/ns]",        0,    None),
        ("dt_20",     r"$\Delta T$ CFD20% [ps]",          None, None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(21, 12))

    for ax, (key, xlabel, xlo, xhi) in zip(axes.flatten(), panels):
        arr = data.get(key)
        if arr is None:
            ax.text(0.5, 0.5, f"키 없음: {key}", transform=ax.transAxes, ha="center")
            continue

        fin       = np.isfinite(arr)
        arr_all   = arr[fin & base_mask]
        m_pass    = mcp_mask[fin & base_mask]
        arr_pass  = arr_all[m_pass]
        arr_fail  = arr_all[~m_pass]

        if len(arr_all) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        if key == "dt_20":
            med = float(np.median(arr_all))
            lo  = xlo if xlo is not None else med - 2000
            hi  = xhi if xhi is not None else med + 2000
        else:
            p1, p99 = np.percentile(arr_all, 1), np.percentile(arr_all, 99)
            lo = xlo if xlo is not None else p1
            hi = xhi if xhi is not None else p99 * 1.05
        lo = max(lo, float(arr_all.min())) if xlo is None else lo
        hi = min(hi, float(arr_all.max())) if xhi is None else hi

        bins = np.linspace(lo, hi, 80)

        ax.hist(arr_all,  bins=bins, histtype="step",       color="grey",      lw=1.8,
                label=f"No MCP cut  N={len(arr_all)}", zorder=2)
        ax.hist(arr_pass, bins=bins, histtype="stepfilled", color="steelblue", alpha=0.45, zorder=3)
        ax.hist(arr_pass, bins=bins, histtype="step",       color="steelblue", lw=1.5,
                label=f"Pass  N={len(arr_pass)}", zorder=3)
        if len(arr_fail) > 0:
            ax.hist(arr_fail, bins=bins, histtype="stepfilled", color="tomato", alpha=0.40, zorder=4)
            ax.hist(arr_fail, bins=bins, histtype="step",       color="tomato", lw=1.5,
                    label=f"Fail  N={len(arr_fail)}", zorder=4)

        if key == "pmax_mcp":
            if mcp_lo_mV > 0:
                ax.axvline(mcp_lo_mV, color="red", ls="--", lw=2,
                           label=f"MCP lo = {mcp_lo_mV:.0f} mV")
            if mcp_hi_mV > 0:
                ax.axvline(mcp_hi_mV, color="darkred", ls="--", lw=2,
                           label=f"MCP hi = {mcp_hi_mV:.0f} mV")

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Entries")
        ax.legend(fontsize=11, loc="upper right")
        ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")
        ax.text(1.0,  1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
                fontsize=11, ha="right", va="bottom")

    mcp_range_str = f"MCP ≥ {mcp_lo_mV:.0f}"
    if mcp_hi_mV > 0:
        mcp_range_str += f" – {mcp_hi_mV:.0f}"
    mcp_range_str += " mV"
    fig.suptitle(
        f"MCP pmax Cut Comparison  |  {mcp_range_str}  |  "
        f"pass {n_pass}/{n_all} ({100*n_pass/n_all:.1f}%)",
        fontsize=14, y=1.01
    )
    plt.tight_layout()
    out = os.path.join(plot_dir, f"mcp_cut_comparison_{file_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → mcp_cut_comparison_{file_tag}.png")


# =============================================================================
# 진단 플롯 3종
# =============================================================================

def plot_noise_rms_vs_pmax(data, noise_mask, noise_cut_mV, plot_label, file_tag, plot_dir,
                           noise_sigma_bounds=None):
    """noise_rms vs pmax scatter: pass=steelblue, fail=tomato."""
    pmax_arr  = data["pmax"]
    noise_arr = data["noise_rms"]
    ok = np.isfinite(pmax_arr) & np.isfinite(noise_arr)

    fig, ax = plt.subplots(figsize=(9, 6))

    m_fail = ok & ~noise_mask
    m_pass = ok &  noise_mask
    if m_fail.sum() > 0:
        ax.scatter(pmax_arr[m_fail], noise_arr[m_fail],
                   s=3, alpha=0.3, color="tomato",
                   label=f"Fail N={m_fail.sum()}  (noise > {noise_cut_mV:.0f} mV)",
                   rasterized=True)
    ax.scatter(pmax_arr[m_pass], noise_arr[m_pass],
               s=3, alpha=0.3, color="steelblue",
               label=f"Pass N={m_pass.sum()}",
               rasterized=True)

    if noise_cut_mV > 0:
        ax.axhline(noise_cut_mV, color="red", ls="--", lw=1.5,
                   label=f"Abs cut = {noise_cut_mV:.1f} mV")
    if noise_sigma_bounds is not None:
        ax.axhline(noise_sigma_bounds[0], color="darkorange", ls=":", lw=1.5,
                   label=f"σ lo = {noise_sigma_bounds[0]:.2f} mV")
        ax.axhline(noise_sigma_bounds[1], color="darkorange", ls=":", lw=1.5,
                   label=f"σ hi = {noise_sigma_bounds[1]:.2f} mV")

    ax.set_xlabel("Peak Amplitude [mV]")
    ax.set_ylabel("Noise RMS [mV]")
    ax.legend(fontsize=12, markerscale=5)
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
            fontsize=12, ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"noise_rms_vs_pmax_{file_tag}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → noise_rms_vs_pmax_{file_tag}.png")


def _plot_dt_bin(sub, sigma, sigma_err, bin_idx, edge_lo, edge_hi,
                 cfd_key, n_events, plot_label, file_tag, cfd_dir):
    """dT 분포 히스토그램 + Gaussian fit → cfd_dir/bin_<N>.png 저장."""
    sub = sub[np.isfinite(sub)]
    if len(sub) < 5:
        return

    data_mean = float(np.mean(sub))
    data_std  = float(np.std(sub))

    # range: mean ± 3σ (fit σ 우선, 실패시 data std)
    sig_ref   = float(sigma) if np.isfinite(sigma) else data_std
    sig_ref   = max(sig_ref, 10.0)
    lo_range  = data_mean - 3.0 * sig_ref
    hi_range  = data_mean + 3.0 * sig_ref

    # 최적 bin 수: Freedman-Diaconis, 범위 내 데이터 기준
    sub_in = sub[(sub >= lo_range) & (sub <= hi_range)]
    n_in   = max(len(sub_in), 1)
    q75, q25 = np.percentile(sub_in, [75, 25]) if n_in >= 4 else (data_std, 0.0)
    iqr = q75 - q25
    if iqr > 0:
        h_fd = 2.0 * iqr / (n_in ** (1.0 / 3.0))
        n_bins_opt = int(np.ceil((hi_range - lo_range) / h_fd))
    else:
        n_bins_opt = int(np.ceil(np.sqrt(n_in)))
    n_bins_opt = int(np.clip(n_bins_opt, 8, 80))
    bin_width  = (hi_range - lo_range) / n_bins_opt

    fig, ax = plt.subplots(figsize=(7, 5))
    counts, bins, _ = ax.hist(sub, bins=n_bins_opt, range=(lo_range, hi_range),
                              histtype="stepfilled", alpha=0.35, color="steelblue")
    ax.hist(sub, bins=n_bins_opt, range=(lo_range, hi_range),
            histtype="step", lw=1.5, color="steelblue")
    bc = (bins[:-1] + bins[1:]) / 2

    chi2_str = ""
    if np.isfinite(sigma):
        peak_idx = int(np.argmax(counts))
        peak_c   = float(bc[peak_idx])
        xfit     = np.linspace(lo_range, hi_range, 400)
        try:
            hw_fit   = max(sigma * 3, 200)
            mask_fit = (bc >= peak_c - hw_fit) & (bc <= peak_c + hw_fit) & (counts > 0)
            p0_fit   = [float(counts[peak_idx]), peak_c, sigma]
            popt_b, pcov_b = curve_fit(
                gauss, bc[mask_fit], counts[mask_fit], p0=p0_fit,
                sigma=np.sqrt(np.maximum(counts[mask_fit], 1)),
                absolute_sigma=True)
            perr_b = np.sqrt(np.diag(pcov_b))
            sigma     = abs(popt_b[2])
            sigma_err = perr_b[2]
            y_exp = gauss(bc[mask_fit], *popt_b)
            chi2  = float(np.sum((counts[mask_fit] - y_exp) ** 2
                                 / np.maximum(counts[mask_fit], 1)))
            ndf   = int(mask_fit.sum()) - 3
            chi2_str = (f"χ²/ndf = {chi2:.1f}/{ndf} = {chi2/ndf:.2f}"
                        if ndf > 0 else "")
            ax.plot(xfit, gauss(xfit, *popt_b), color="tomato", lw=2,
                    label=rf"Gaussian fit  $\sigma$ = {sigma:.1f} ± {sigma_err:.1f} ps")
        except Exception:
            ax.plot(xfit, gauss(xfit, float(counts[peak_idx]), peak_c, sigma),
                    color="tomato", lw=2,
                    label=rf"Gaussian fit  $\sigma$ = {sigma:.1f} ± {sigma_err:.1f} ps")
        ax.legend(fontsize=11)

    # 통계 정보 텍스트 박스
    stats_lines = [
        f"N = {len(sub)}  (range: {n_in})",
        f"Mean = {data_mean:.1f} ps",
        f"Std = {data_std:.1f} ps",
    ]
    if np.isfinite(sigma):
        stats_lines.append(rf"Fit $\sigma$ = {sigma:.1f} ± {sigma_err:.1f} ps")
    if chi2_str:
        stats_lines.append(chi2_str)
    stats_lines.append(f"Bins = {n_bins_opt},  width = {bin_width:.1f} ps")
    ax.text(0.97, 0.97, "\n".join(stats_lines),
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="grey", alpha=0.85))

    noise_center = 0.5 * (edge_lo + edge_hi)
    ax.set_xlabel(r"$\Delta t$ [ps]")
    ax.set_ylabel(f"Entries / {bin_width:.1f} ps")
    ax.set_title(
        f"CFD {cfd_key}%  |  noise RMS bin {bin_idx + 1}: "
        f"[{edge_lo:.2f}, {edge_hi:.2f}] mV",
        fontsize=11
    )
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom")
    ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
            fontsize=10, ha="right", va="bottom")
    plt.tight_layout()
    out = os.path.join(cfd_dir, f"bin{bin_idx + 1:02d}_noise{noise_center:.2f}mV_{file_tag}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def plot_noise_rms_vs_time_resolution(data, noise_mask, noise_cut_mV, plot_label,
                                       file_tag, plot_dir,
                                       cfd_keys=(10, 20, 30, 40, 50), n_bins=15,
                                       noise_sigma_bounds=None):
    """noise_rms 균등 bin별 σ_dT 프로파일 플롯 (CFD 5개 한 그래프)."""
    COLORS = {10: "royalblue", 20: "forestgreen", 30: "darkorange",
              40: "purple", 50: "crimson"}

    noise_arr = data["noise_rms"]
    ok_noise  = np.isfinite(noise_arr)
    if ok_noise.sum() < 30:
        print(f"[WARNING] plot_noise_rms_vs_time_resolution: 이벤트 부족, 스킵")
        return

    n_base  = noise_arr[ok_noise]
    lo      = np.percentile(n_base, 2)
    hi      = np.percentile(n_base, 98)
    edges   = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(10, 6))

    for cfd_key in cfd_keys:
        dt_key = f"dt_{cfd_key}"
        if dt_key not in data:
            continue
        dt_arr = data[dt_key]
        ok = np.isfinite(noise_arr) & np.isfinite(dt_arr)
        if ok.sum() < 30:
            continue

        n_arr = noise_arr[ok]
        d_arr = dt_arr[ok]

        sig_vals, sig_errs = [], []
        for i in range(n_bins):
            m_bin = (n_arr >= edges[i]) & (n_arr < edges[i + 1])
            s, se = fit_sigma_dt(d_arr[m_bin])
            sig_vals.append(s)
            sig_errs.append(se)

        sig_vals = np.array(sig_vals)
        sig_errs = np.array(sig_errs)
        valid_b  = np.isfinite(sig_vals) & np.isfinite(sig_errs)

        ax.errorbar(centers[valid_b], sig_vals[valid_b], yerr=sig_errs[valid_b],
                    fmt="o-", color=COLORS.get(cfd_key, "grey"), lw=1.8, capsize=4,
                    label=f"CFD {cfd_key}%")

    if noise_cut_mV > 0:
        ax.axvline(noise_cut_mV, color="red", ls="--", lw=1.5,
                   label=f"Abs cut = {noise_cut_mV:.1f} mV")
    if noise_sigma_bounds is not None:
        ax.axvline(noise_sigma_bounds[0], color="darkorange", ls=":", lw=1.5,
                   label=f"σ lo = {noise_sigma_bounds[0]:.2f} mV")
        ax.axvline(noise_sigma_bounds[1], color="darkorange", ls=":", lw=1.5,
                   label=f"σ hi = {noise_sigma_bounds[1]:.2f} mV")

    ax.set_xlabel("Noise RMS [mV]")
    ax.set_ylabel(r"Time Resolution $\sigma$ [ps]")
    ax.legend(fontsize=11)
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
            fontsize=12, ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"noise_rms_vs_timeres_{file_tag}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → noise_rms_vs_timeres_{file_tag}.png")


def plot_mcp_threshold_scan(data, cut_mask, mcp_lo, mcp_hi, n_steps,
                             plot_label, file_tag, plot_dir,
                             cfd_keys=(20, 30, 50)):
    """
    MCP threshold 를 mcp_lo → mcp_hi 까지 n_steps 단계로 올려가며
    각 컷에서 Gaussian fit σ_dT 를 계산해 프로파일 플롯으로 저장.
    data["pmax_mcp"] 필드 필요 (Pass 2에서 항상 저장).
    """
    if "pmax_mcp" not in data:
        print("[WARNING] plot_mcp_threshold_scan: pmax_mcp 필드 없음, 스킵")
        return

    COLORS = {20: "royalblue", 30: "forestgreen", 50: "crimson",
              10: "darkorange", 40: "purple", 60: "grey"}

    thresholds = np.linspace(mcp_lo, mcp_hi, int(n_steps))
    pmax_mcp   = data["pmax_mcp"]
    base_ok    = cut_mask & np.isfinite(pmax_mcp)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()   # 오른쪽 축: N_events

    for cfd_key in cfd_keys:
        dt_key = f"dt_{cfd_key}"
        if dt_key not in data:
            continue
        dt_arr = data[dt_key]
        ok_dt  = np.isfinite(dt_arr)

        sig_vals, sig_errs, n_counts = [], [], []
        for thr in thresholds:
            mask = base_ok & ok_dt & (pmax_mcp >= thr)
            s, se = fit_sigma_dt_full(dt_arr[mask])   # plot_time_resolution 과 동일 방식
            sig_vals.append(s)
            sig_errs.append(se)
            n_counts.append(int(mask.sum()))

        sig_vals = np.array(sig_vals)
        sig_errs = np.array(sig_errs)
        n_counts = np.array(n_counts)
        valid    = np.isfinite(sig_vals) & np.isfinite(sig_errs)

        color = COLORS.get(cfd_key, "grey")
        ax.errorbar(thresholds[valid], sig_vals[valid], yerr=sig_errs[valid],
                    fmt="o-", color=color, lw=1.8, capsize=4,
                    label=f"CFD {cfd_key}%")

    # N_events (첫 번째 CFD 기준, 대표값) 를 오른쪽 축에 표시
    ref_key = cfd_keys[0]
    dt_ref  = data.get(f"dt_{ref_key}")
    if dt_ref is not None:
        n_ref = []
        for thr in thresholds:
            mask = base_ok & np.isfinite(dt_ref) & (pmax_mcp >= thr)
            n_ref.append(int(mask.sum()))
        ax2.plot(thresholds, n_ref, "k--", lw=1.2, alpha=0.5, label="N events")
        ax2.set_ylabel("N events (dashed)", fontsize=11, color="grey")
        ax2.tick_params(axis="y", colors="grey")

    ax.set_xlabel("MCP Threshold [mV]", fontsize=12)
    ax.set_ylabel(r"Time Resolution $\sigma$ [ps]", fontsize=12)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11)
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
            fontsize=12, ha="right", va="bottom")

    plt.tight_layout()
    out = os.path.join(plot_dir, f"mcp_threshold_scan_{file_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → mcp_threshold_scan_{file_tag}.png"
          f"  (MCP {mcp_lo:.0f}–{mcp_hi:.0f} mV, {n_steps} steps)")


def _calc_auto_bins(sub, lo, hi, n_min=8, n_max=None):
    """Freedman-Diaconis 규칙으로 최적 빈 수 계산. n_max 기본값 = DT_HIST_BINS."""
    if n_max is None:
        n_max = DT_HIST_BINS
    sub_in = sub[(sub >= lo) & (sub <= hi)]
    n_in = max(len(sub_in), 1)
    q75, q25 = (np.percentile(sub_in, [75, 25]) if n_in >= 4
                else (float(np.std(sub_in)), 0.0))
    iqr = q75 - q25
    if iqr > 0:
        h_fd = 2.0 * iqr / (n_in ** (1.0 / 3.0))
        n_bins = int(np.ceil((hi - lo) / h_fd))
    else:
        n_bins = int(np.ceil(np.sqrt(n_in)))
    return int(np.clip(n_bins, n_min, n_max))


def plot_noise_rms_cut_scan(data, cut_mask, rms_lo, rms_hi, n_steps,
                             plot_label, file_tag, plot_dir,
                             cfd_keys=(20, 30, 50), stem=""):
    """
    noise_rms 상한 컷을 rms_lo → rms_hi 까지 n_steps 단계로 올려가며
    각 step에서:
      1) σ_dT 프로파일 플롯 (summary)
      2) 각 step × CFD 별 dT 분포 + Gaussian fit 개별 플롯
         → plots/noise_rms_cut_scan/ 서브디렉토리에 저장
    """
    COLORS = {20: "royalblue", 30: "forestgreen", 50: "crimson",
              10: "darkorange", 40: "purple", 60: "grey"}

    thresholds = np.linspace(rms_lo, rms_hi, int(n_steps))
    noise_arr  = data["noise_rms"]
    base_ok    = cut_mask & np.isfinite(noise_arr)

    # 서브디렉토리 생성
    scan_dir = os.path.join(plot_dir, "noise_rms_cut_scan")
    os.makedirs(scan_dir, exist_ok=True)

    # ── 1) summary 프로파일 플롯 ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()

    for cfd_key in cfd_keys:
        dt_key = f"dt_{cfd_key}"
        if dt_key not in data:
            continue
        dt_arr = data[dt_key]
        ok_dt  = np.isfinite(dt_arr)

        sig_vals, sig_errs, n_counts = [], [], []
        for thr in thresholds:
            mask = base_ok & ok_dt & (noise_arr <= thr)
            s, se = fit_sigma_dt_full(dt_arr[mask])
            sig_vals.append(s)
            sig_errs.append(se)
            n_counts.append(int(mask.sum()))

        sig_vals = np.array(sig_vals)
        sig_errs = np.array(sig_errs)
        valid    = np.isfinite(sig_vals) & np.isfinite(sig_errs)

        ax.errorbar(thresholds[valid], sig_vals[valid], yerr=sig_errs[valid],
                    fmt="o-", color=COLORS.get(cfd_key, "grey"), lw=1.8, capsize=4,
                    label=f"CFD {cfd_key}%")

    ref_key = cfd_keys[0]
    dt_ref  = data.get(f"dt_{ref_key}")
    if dt_ref is not None:
        n_ref = [int((base_ok & np.isfinite(dt_ref) & (noise_arr <= t)).sum())
                 for t in thresholds]
        ax2.plot(thresholds, n_ref, "k--", lw=1.2, alpha=0.5, label="N events")
        ax2.set_ylabel("N events (dashed)", fontsize=11, color="grey")
        ax2.tick_params(axis="y", colors="grey")

    ax.set_xlabel("Noise RMS Cut [mV]", fontsize=12)
    ax.set_ylabel(r"Time Resolution $\sigma$ [ps]", fontsize=12)
    ax.set_ylim(*NOISE_SCAN_YLIM)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11)
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
            fontsize=12, ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"noise_rms_cut_scan_{file_tag}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── 1b-zoom) Sr 파일: y 35–45 ps 줌인 플롯 ──────────────────────────
    if stem.startswith("Sr"):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax2 = ax.twinx()
        for cfd_key in cfd_keys:
            dt_key = f"dt_{cfd_key}"
            if dt_key not in data:
                continue
            dt_arr = data[dt_key]
            ok_dt  = np.isfinite(dt_arr)
            sig_vals, sig_errs = [], []
            for thr in thresholds:
                mask = base_ok & ok_dt & (noise_arr <= thr)
                s, se = fit_sigma_dt_full(dt_arr[mask])
                sig_vals.append(s)
                sig_errs.append(se)
            sig_vals = np.array(sig_vals)
            sig_errs = np.array(sig_errs)
            valid    = np.isfinite(sig_vals) & np.isfinite(sig_errs)
            ax.errorbar(thresholds[valid], sig_vals[valid], yerr=sig_errs[valid],
                        fmt="o-", color=COLORS.get(cfd_key, "grey"), lw=1.8, capsize=4,
                        label=f"CFD {cfd_key}%")
        ref_key = cfd_keys[0]
        dt_ref  = data.get(f"dt_{ref_key}")
        if dt_ref is not None:
            n_ref = [int((base_ok & np.isfinite(dt_ref) & (noise_arr <= t)).sum())
                     for t in thresholds]
            ax2.plot(thresholds, n_ref, "k--", lw=1.2, alpha=0.5, label="N events")
            ax2.set_ylabel("N events (dashed)", fontsize=11, color="grey")
            ax2.tick_params(axis="y", colors="grey")
        ax.set_xlabel("Noise RMS Cut [mV]", fontsize=12)
        ax.set_ylabel(r"Time Resolution $\sigma$ [ps]", fontsize=12)
        ax.set_ylim(*NOISE_SCAN_YLIM_SR)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11)
        ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom")
        ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
                fontsize=12, ha="right", va="bottom")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"noise_rms_cut_scan_zoom_{file_tag}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ── 1b) pulse area MPV vs noise RMS cut ─────────────────────────────
    charge_arr = data.get("charge")
    if charge_arr is not None:
        ok_q = base_ok & np.isfinite(charge_arr) & (charge_arr > 0)
        q_mpvs, q_errs, n_q = [], [], []
        for thr in thresholds:
            mask = ok_q & (noise_arr <= thr)
            mpv, mpv_err = fit_charge_mpv(charge_arr[mask])
            q_mpvs.append(mpv)
            q_errs.append(mpv_err)
            n_q.append(int(mask.sum()))

        q_mpvs  = np.array(q_mpvs)
        q_errs  = np.array(q_errs)
        valid_q = np.isfinite(q_mpvs) & np.isfinite(q_errs)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax2q = ax.twinx()
        ax.errorbar(thresholds[valid_q], q_mpvs[valid_q], yerr=q_errs[valid_q],
                    fmt="o-", color="steelblue", lw=1.8, capsize=4,
                    label="Pulse Area MPV (Landau fit) ± err")
        ax2q.plot(thresholds, n_q, "k--", lw=1.2, alpha=0.5, label="N events")
        ax2q.set_ylabel("N events (dashed)", fontsize=11, color="grey")
        ax2q.tick_params(axis="y", colors="grey")
        ax.set_xlabel("Noise RMS Cut [mV]", fontsize=12)
        ax.set_ylabel("Pulse Area MPV [mV·ns]", fontsize=12)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2q.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11)
        ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom")
        ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
                fontsize=12, ha="right", va="bottom")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"noise_rms_cut_scan_charge_{file_tag}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        # zoomed (y: 50–70 mV·ns)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax2q = ax.twinx()
        ax.errorbar(thresholds[valid_q], q_mpvs[valid_q], yerr=q_errs[valid_q],
                    fmt="o-", color="steelblue", lw=1.8, capsize=4,
                    label="Pulse Area MPV (Landau fit) ± err")
        ax2q.plot(thresholds, n_q, "k--", lw=1.2, alpha=0.5, label="N events")
        ax2q.set_ylabel("N events (dashed)", fontsize=11, color="grey")
        ax2q.tick_params(axis="y", colors="grey")
        ax.set_ylim(*CHARGE_SCAN_YLIM_ZOOM)
        ax.set_xlabel("Noise RMS Cut [mV]", fontsize=12)
        ax.set_ylabel("Pulse Area MPV [mV·ns]", fontsize=12)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2q.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11)
        ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom")
        ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
                fontsize=12, ha="right", va="bottom")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"noise_rms_cut_scan_charge_zoom_{file_tag}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ── 2) step × CFD 별 dT 분포 개별 플롯 (고정 비닝 + 자동 비닝 각 1장) ──
    n_saved = 0
    for step_idx, thr in enumerate(thresholds):
        for cfd_key in cfd_keys:
            dt_key = f"dt_{cfd_key}"
            if dt_key not in data:
                continue
            dt_arr = data[dt_key]
            mask   = base_ok & np.isfinite(dt_arr) & (noise_arr <= thr)
            sub    = dt_arr[mask]
            if len(sub) < 10:
                continue

            med    = float(np.median(sub))
            lo_h, hi_h = med - DT_HIST_HALF_RANGE_PS, med + DT_HIST_HALF_RANGE_PS
            color  = COLORS.get(cfd_key, "navy")

            n_bins_auto = _calc_auto_bins(sub, lo_h, hi_h)
            bin_variants = [
                (DT_HIST_BINS,  ""),
                (n_bins_auto,   "_autobin"),
            ]

            for n_bins_plot, fname_suffix in bin_variants:
                counts, bins = np.histogram(sub, bins=n_bins_plot, range=(lo_h, hi_h))
                bc  = (bins[:-1] + bins[1:]) / 2
                bw  = float(bins[1] - bins[0])

                mu_ps = sigma_ps = sigma_err_ps = chi2_val = ndf_val = np.nan
                popt_ok = False
                try:
                    peak_idx = int(np.argmax(counts))
                    p0 = [float(counts[peak_idx]), float(bc[peak_idx]), 100.0]
                    popt, pcov = curve_fit(gauss, bc, counts, p0=p0,
                                           sigma=np.sqrt(np.maximum(counts, 1)),
                                           absolute_sigma=True, maxfev=5000)
                    perr         = np.sqrt(np.diag(pcov))
                    mu_ps        = float(popt[1])
                    sigma_ps     = float(abs(popt[2]))
                    sigma_err_ps = float(perr[2])
                    y_exp        = gauss(bc, *popt)
                    chi2_val     = float(np.sum((counts - y_exp) ** 2
                                                / np.maximum(counts, 1)))
                    ndf_val      = int((counts > 0).sum()) - 3
                    popt_ok      = True
                except Exception:
                    pass

                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(sub, bins=n_bins_plot, range=(lo_h, hi_h),
                        histtype="stepfilled", lw=1.5, color=color, alpha=0.4,
                        label=f"$\\Delta T$  CFD {cfd_key}%")
                ax.hist(sub, bins=n_bins_plot, range=(lo_h, hi_h),
                        histtype="step", lw=1.5, color=color)
                if popt_ok:
                    x_fine = np.linspace(lo_h, hi_h, 600)
                    ax.plot(x_fine, gauss(x_fine, *popt), "r-", lw=2)

                bin_label = ("auto (F-D)" if fname_suffix else f"fixed ({DT_HIST_BINS})")
                stats = [f"N = {len(sub)}",
                         f"Mean = {float(np.mean(sub)):.1f} ps",
                         f"Std  = {float(np.std(sub)):.1f} ps"]
                if popt_ok:
                    stats += [rf"Fit $\mu$ = {mu_ps:.1f} ps",
                              rf"Fit $\sigma$ = {sigma_ps:.1f} ± {sigma_err_ps:.1f} ps"]
                    if ndf_val > 0:
                        stats.append(f"$\\chi^2$/ndf = {chi2_val:.1f}/{ndf_val}"
                                     f" = {chi2_val/ndf_val:.2f}")
                stats.append(f"Bins = {n_bins_plot} ({bin_label}),  width = {bw:.1f} ps")
                ax.text(0.97, 0.97, "\n".join(stats),
                        transform=ax.transAxes, fontsize=9, va="top", ha="right",
                        bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                  ec="grey", alpha=0.85))

                ax.set_xlabel(r"$\Delta T$ [ps]")
                ax.set_ylabel(f"Entries / {bw:.1f} ps")
                ax.set_title(f"noise RMS ≤ {thr:.2f} mV  |  CFD {cfd_key}%  |  "
                             f"step {step_idx + 1}/{int(n_steps)}"
                             + (f"  [bins={n_bins_plot}, F-D]" if fname_suffix else ""),
                             fontsize=11)
                ax.legend(fontsize=10)
                ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
                        fontsize=11, fontweight="bold", va="bottom")
                ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
                        fontsize=10, ha="right", va="bottom")
                plt.tight_layout()

                fname = (f"step{step_idx + 1:02d}_noise{thr:.2f}mV"
                         f"_cfd{cfd_key}{fname_suffix}_{file_tag}.png")
                plt.savefig(os.path.join(scan_dir, fname), dpi=120, bbox_inches="tight")
                plt.close()
                n_saved += 1

    print(f"  → noise_rms_cut_scan_{file_tag}.png  (time resolution)")
    if stem.startswith("Sr"):
        print(f"  → noise_rms_cut_scan_zoom_{file_tag}.png  (y: 35–45 ps)")
    print(f"  → noise_rms_cut_scan_charge_{file_tag}.png  (collected charge)")
    print(f"  → noise_rms_cut_scan/ ({n_saved} dt plots, 고정+자동 비닝 각 1장)"
          f"  [noise RMS {rms_lo:.1f}–{rms_hi:.1f} mV, {int(n_steps)} steps]")


def plot_snr_cut_scan(data, cut_mask, snr_lo, snr_hi, n_steps,
                      plot_label, file_tag, plot_dir,
                      cfd_keys=(20, 30)):
    """
    SNR(=pmax/noise_rms) 하한 컷을 snr_lo → snr_hi 까지 n_steps 단계로 스캔.
    각 step에서:
      1) σ_dT 프로파일 + N events 오버레이 (summary)
      2) Pulse area MPV vs SNR cut
      3) 각 step × CFD 별 dT 분포 + Gaussian fit 개별 플롯
         → plots/snr_cut_scan/ 서브디렉토리에 저장
    """
    COLORS = {20: "royalblue", 30: "forestgreen", 50: "crimson",
              10: "darkorange", 40: "purple", 60: "grey"}

    thresholds = np.linspace(snr_lo, snr_hi, int(n_steps))
    noise_arr  = data["noise_rms"]
    pmax_arr   = data["pmax"]
    ok_snr_base = np.isfinite(noise_arr) & (noise_arr > 0) & np.isfinite(pmax_arr)
    snr_arr    = np.where(ok_snr_base, pmax_arr / noise_arr, 0.0)
    base_ok    = cut_mask & ok_snr_base

    scan_dir = os.path.join(plot_dir, "snr_cut_scan")
    os.makedirs(scan_dir, exist_ok=True)

    # ── 1) σ_dT summary ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()

    for cfd_key in cfd_keys:
        dt_key = f"dt_{cfd_key}"
        if dt_key not in data:
            continue
        dt_arr = data[dt_key]
        ok_dt  = np.isfinite(dt_arr)

        sig_vals, sig_errs, n_counts = [], [], []
        for thr in thresholds:
            mask = base_ok & ok_dt & (snr_arr >= thr)
            s, se = fit_sigma_dt_full(dt_arr[mask])
            sig_vals.append(s)
            sig_errs.append(se)
            n_counts.append(int(mask.sum()))

        sig_vals = np.array(sig_vals)
        sig_errs = np.array(sig_errs)
        valid    = np.isfinite(sig_vals) & np.isfinite(sig_errs)

        ax.errorbar(thresholds[valid], sig_vals[valid], yerr=sig_errs[valid],
                    fmt="o-", color=COLORS.get(cfd_key, "grey"), lw=1.8, capsize=4,
                    label=f"CFD {cfd_key}%")

    ref_key = cfd_keys[0]
    dt_ref  = data.get(f"dt_{ref_key}")
    if dt_ref is not None:
        n_ref = [int((base_ok & np.isfinite(dt_ref) & (snr_arr >= t)).sum())
                 for t in thresholds]
        ax2.plot(thresholds, n_ref, "k--", lw=1.2, alpha=0.5, label="N events")
        ax2.set_ylabel("N events (dashed)", fontsize=11, color="grey")
        ax2.tick_params(axis="y", colors="grey")

    # SNR 분포 mean ± 1σ 수직선 + 음영
    snr_valid  = snr_arr[base_ok]
    snr_mean   = float(np.mean(snr_valid))
    snr_std    = float(np.std(snr_valid))
    ax.axvline(snr_mean, color="black", lw=1.5, ls="-",
               label=f"SNR mean = {snr_mean:.1f}")
    ax.axvline(snr_mean - snr_std, color="black", lw=1.0, ls="--",
               label=f"±1σ = [{snr_mean - snr_std:.1f}, {snr_mean + snr_std:.1f}]")
    ax.axvline(snr_mean + snr_std, color="black", lw=1.0, ls="--")
    ax.axvspan(snr_mean - snr_std, snr_mean + snr_std,
               alpha=0.08, color="black")

    ax.set_xlabel("SNR Cut (pmax / noise_rms)", fontsize=12)
    ax.set_ylabel(r"Time Resolution $\sigma$ [ps]", fontsize=12)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11)
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
            fontsize=12, ha="right", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"snr_cut_scan_{file_tag}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2) Pulse area MPV vs SNR cut ─────────────────────────────────────────
    charge_arr = data.get("charge")
    if charge_arr is not None:
        ok_q = base_ok & np.isfinite(charge_arr) & (charge_arr > 0)
        q_mpvs, q_errs, n_q = [], [], []
        for thr in thresholds:
            mask = ok_q & (snr_arr >= thr)
            mpv, mpv_err = fit_charge_mpv(charge_arr[mask])
            q_mpvs.append(mpv)
            q_errs.append(mpv_err)
            n_q.append(int(mask.sum()))

        q_mpvs  = np.array(q_mpvs)
        q_errs  = np.array(q_errs)
        valid_q = np.isfinite(q_mpvs) & np.isfinite(q_errs)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax2q = ax.twinx()
        ax.errorbar(thresholds[valid_q], q_mpvs[valid_q], yerr=q_errs[valid_q],
                    fmt="o-", color="steelblue", lw=1.8, capsize=4,
                    label="Pulse Area MPV (Landau fit) ± err")
        ax2q.plot(thresholds, n_q, "k--", lw=1.2, alpha=0.5, label="N events")
        ax2q.set_ylabel("N events (dashed)", fontsize=11, color="grey")
        ax2q.tick_params(axis="y", colors="grey")
        ax.set_xlabel("SNR Cut (pmax / noise_rms)", fontsize=12)
        ax.set_ylabel("Pulse Area MPV [mV·ns]", fontsize=12)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2q.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11)
        ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="bottom")
        ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
                fontsize=12, ha="right", va="bottom")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"snr_cut_scan_charge_{file_tag}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    # ── 3) step × CFD 별 dT 분포 개별 플롯 ──────────────────────────────────
    n_saved = 0
    for step_idx, thr in enumerate(thresholds):
        for cfd_key in cfd_keys:
            dt_key = f"dt_{cfd_key}"
            if dt_key not in data:
                continue
            dt_arr = data[dt_key]
            mask   = base_ok & np.isfinite(dt_arr) & (snr_arr >= thr)
            sub    = dt_arr[mask]
            if len(sub) < 10:
                continue

            med  = float(np.median(sub))
            lo_h, hi_h = med - 1000.0, med + 1000.0
            counts, bins = np.histogram(sub, bins=200, range=(lo_h, hi_h))
            bc   = (bins[:-1] + bins[1:]) / 2
            bw   = float(bins[1] - bins[0])

            mu_ps = sigma_ps = sigma_err_ps = chi2_val = ndf_val = np.nan
            popt_ok = False
            try:
                peak_idx = int(np.argmax(counts))
                p0 = [float(counts[peak_idx]), float(bc[peak_idx]), 100.0]
                popt, pcov = curve_fit(gauss, bc, counts, p0=p0,
                                       sigma=np.sqrt(np.maximum(counts, 1)),
                                       absolute_sigma=True, maxfev=5000)
                perr         = np.sqrt(np.diag(pcov))
                mu_ps        = float(popt[1])
                sigma_ps     = float(abs(popt[2]))
                sigma_err_ps = float(perr[2])
                y_exp        = gauss(bc, *popt)
                chi2_val     = float(np.sum((counts - y_exp) ** 2
                                            / np.maximum(counts, 1)))
                ndf_val      = int((counts > 0).sum()) - 3
                popt_ok      = True
            except Exception:
                pass

            color = COLORS.get(cfd_key, "navy")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(sub, bins=200, range=(lo_h, hi_h),
                    histtype="stepfilled", lw=1.5, color=color, alpha=0.4,
                    label=f"$\\Delta T$  CFD {cfd_key}%")
            ax.hist(sub, bins=200, range=(lo_h, hi_h),
                    histtype="step", lw=1.5, color=color)
            if popt_ok:
                x_fine = np.linspace(lo_h, hi_h, 600)
                ax.plot(x_fine, gauss(x_fine, *popt), "r-", lw=2)

            stats = [f"N = {len(sub)}",
                     f"Mean = {float(np.mean(sub)):.1f} ps",
                     f"Std  = {float(np.std(sub)):.1f} ps"]
            if popt_ok:
                stats += [rf"Fit $\mu$ = {mu_ps:.1f} ps",
                          rf"Fit $\sigma$ = {sigma_ps:.1f} ± {sigma_err_ps:.1f} ps"]
                if ndf_val > 0:
                    stats.append(f"$\\chi^2$/ndf = {chi2_val:.1f}/{ndf_val}"
                                 f" = {chi2_val/ndf_val:.2f}")
            stats.append(f"Bins = 200,  width = {bw:.1f} ps")
            ax.text(0.97, 0.97, "\n".join(stats),
                    transform=ax.transAxes, fontsize=9, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="grey", alpha=0.85))

            ax.set_xlabel(r"$\Delta T$ [ps]")
            ax.set_ylabel(f"Entries / {bw:.1f} ps")
            ax.set_title(f"SNR ≥ {thr:.1f}  |  CFD {cfd_key}%  |  "
                         f"step {step_idx + 1}/{int(n_steps)}",
                         fontsize=11)
            ax.legend(fontsize=10)
            ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="bottom")
            ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
                    fontsize=10, ha="right", va="bottom")
            plt.tight_layout()

            fname = (f"step{step_idx + 1:02d}_snr{thr:.1f}"
                     f"_cfd{cfd_key}_{file_tag}.png")
            plt.savefig(os.path.join(scan_dir, fname), dpi=120, bbox_inches="tight")
            plt.close()
            n_saved += 1

    print(f"  → snr_cut_scan_{file_tag}.png  (time resolution)")
    print(f"  → snr_cut_scan_charge_{file_tag}.png  (collected charge)")
    print(f"  → snr_cut_scan/ ({n_saved} dt plots)"
          f"  [SNR {snr_lo:.1f}–{snr_hi:.1f}, {int(n_steps)} steps]")


def plot_noise_sigma_cut_scan(data, cut_mask, sigma_values,
                              plot_label, file_tag, plot_dir,
                              cfd_keys=(20, 30, 50)):
    """
    noise_rms 시그마 컷을 지정된 N 값 리스트로 스캔.
    각 step에서 |noise_rms - mean| <= N×σ 마스크 적용:
      1) σ_dT vs N 프로파일 플롯 (summary)
      2) step × CFD 별 dT 분포 개별 플롯 → plots/noise_sigma_cut_scan/
    mean, σ 는 cut_mask 통과 이벤트의 noise_rms 분포에서 계산.
    """
    COLORS = {20: "royalblue", 30: "forestgreen", 50: "crimson",
              10: "darkorange", 40: "purple", 60: "grey"}

    noise_arr  = data["noise_rms"]
    base_ok    = cut_mask & np.isfinite(noise_arr)

    # mean, std 계산 (cut_mask 통과 + finite 이벤트 기준)
    noise_base = noise_arr[base_ok]
    if len(noise_base) < 10:
        print("[WARNING] plot_noise_sigma_cut_scan: 이벤트 부족, 스킵")
        return
    mean_rms = float(np.mean(noise_base))
    std_rms  = float(np.std(noise_base))
    print(f"  [sigma scan] noise RMS: mean={mean_rms:.4f} mV, σ={std_rms:.4f} mV")

    sigmas     = np.array(sorted(sigma_values))
    n_steps    = len(sigmas)

    # 사용 가능한 CFD dt 키 목록 (출력용)
    avail_cfd = [k for k in cfd_keys if f"dt_{k}" in data]
    charge_arr = data.get("charge")

    # 헤더
    cfd_hdr = "  ".join(f"σ_dT cfd{k}[ps]" for k in avail_cfd)
    print(f"  {'N':>5}  {'lo[mV]':>8}  {'hi[mV]':>8}  {'N_pass':>7}  "
          f"{'Area_MPV[mV·ns]':>15}  {cfd_hdr}")
    print(f"  {'-' * (55 + 18 * len(avail_cfd))}")

    # 각 sigma 값에 대한 컷 범위, 통과 이벤트 수, charge, time resolution 출력
    ok_any = base_ok & np.isfinite(noise_arr)
    for nsig in sigmas:
        lo_v    = mean_rms - nsig * std_rms
        hi_v    = mean_rms + nsig * std_rms
        sig_mask = ok_any & (noise_arr >= lo_v) & (noise_arr <= hi_v)
        n_pass  = int(sig_mask.sum())

        # pulse area: Landau MPV
        q_med_str = f"{'—':>13}"
        if charge_arr is not None:
            q_sub = charge_arr[sig_mask & np.isfinite(charge_arr) & (charge_arr > 0)]
            if len(q_sub) > 10:
                mpv, mpv_err = fit_charge_mpv(q_sub)
                q_med_str = (f"{mpv:>10.3f}±{mpv_err:<5.3f}"
                             if np.isfinite(mpv) else f"{'—':>13}")

        # time resolution per CFD
        tr_parts = []
        for k in avail_cfd:
            dt_sub = data[f"dt_{k}"][sig_mask & np.isfinite(data[f"dt_{k}"])]
            s, se  = fit_sigma_dt_full(dt_sub) if len(dt_sub) > 10 else (np.nan, np.nan)
            tr_parts.append(f"{s:>14.1f}±{se:<5.1f}" if np.isfinite(s) else f"{'—':>20}")
        tr_str = "  ".join(tr_parts)

        print(f"  {nsig:>5.1f}  {lo_v:>8.4f}  {hi_v:>8.4f}  {n_pass:>7d}  "
              f"{q_med_str}  {tr_str}")
    print()

    scan_dir = os.path.join(plot_dir, "noise_sigma_cut_scan")
    os.makedirs(scan_dir, exist_ok=True)

    # ── 1) summary 프로파일 플롯 ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()

    for cfd_key in cfd_keys:
        dt_key = f"dt_{cfd_key}"
        if dt_key not in data:
            continue
        dt_arr = data[dt_key]
        ok_dt  = np.isfinite(dt_arr)

        sig_vals, sig_errs = [], []
        for nsig in sigmas:
            sig_lo_v = mean_rms - nsig * std_rms
            sig_hi_v = mean_rms + nsig * std_rms
            mask = base_ok & ok_dt & (noise_arr >= sig_lo_v) & (noise_arr <= sig_hi_v)
            s, se = fit_sigma_dt_full(dt_arr[mask])
            sig_vals.append(s)
            sig_errs.append(se)

        sig_vals = np.array(sig_vals)
        sig_errs = np.array(sig_errs)
        valid    = np.isfinite(sig_vals) & np.isfinite(sig_errs)

        ax.errorbar(sigmas[valid], sig_vals[valid], yerr=sig_errs[valid],
                    fmt="o-", color=COLORS.get(cfd_key, "grey"), lw=1.8, capsize=4,
                    label=f"CFD {cfd_key}%")

    ref_key = cfd_keys[0]
    dt_ref  = data.get(f"dt_{ref_key}")
    if dt_ref is not None:
        ok_dt_ref = np.isfinite(dt_ref)
        n_ref = []
        for nsig in sigmas:
            sig_lo_v = mean_rms - nsig * std_rms
            sig_hi_v = mean_rms + nsig * std_rms
            mask = base_ok & ok_dt_ref & (noise_arr >= sig_lo_v) & (noise_arr <= sig_hi_v)
            n_ref.append(int(mask.sum()))
        ax2.plot(sigmas, n_ref, "k--", lw=1.2, alpha=0.5, label="N events")
        ax2.set_ylabel("N events (dashed)", fontsize=11, color="grey")
        ax2.tick_params(axis="y", colors="grey")

    ax.set_xlabel(r"Noise RMS $\sigma$-cut  (N,  range = mean $\pm$ N$\sigma$)", fontsize=12)
    ax.set_ylabel(r"Time Resolution $\sigma$ [ps]", fontsize=12)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=11)
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
            fontsize=12, ha="right", va="bottom")
    # 범례 하단에 mean/std 표시
    ax.text(0.02, 0.03,
            f"noise RMS:  mean = {mean_rms:.3f} mV,  σ = {std_rms:.3f} mV",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"noise_sigma_cut_scan_{file_tag}.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # ── 2) step × CFD 별 dT 분포 개별 플롯 ──────────────────────────────
    n_saved = 0
    for step_idx, nsig in enumerate(sigmas):
        sig_lo_v = mean_rms - nsig * std_rms
        sig_hi_v = mean_rms + nsig * std_rms
        for cfd_key in cfd_keys:
            dt_key = f"dt_{cfd_key}"
            if dt_key not in data:
                continue
            dt_arr = data[dt_key]
            mask   = base_ok & np.isfinite(dt_arr) & (noise_arr >= sig_lo_v) & (noise_arr <= sig_hi_v)
            sub    = dt_arr[mask]
            if len(sub) < 10:
                continue

            med  = float(np.median(sub))
            lo_h, hi_h = med - 1000.0, med + 1000.0
            counts, bins = np.histogram(sub, bins=200, range=(lo_h, hi_h))
            bc   = (bins[:-1] + bins[1:]) / 2
            bw   = float(bins[1] - bins[0])

            mu_ps = sigma_ps = sigma_err_ps = chi2_val = ndf_val = np.nan
            popt_ok = False
            try:
                peak_idx = int(np.argmax(counts))
                p0 = [float(counts[peak_idx]), float(bc[peak_idx]), 100.0]
                popt, pcov = curve_fit(gauss, bc, counts, p0=p0,
                                       sigma=np.sqrt(np.maximum(counts, 1)),
                                       absolute_sigma=True, maxfev=5000)
                perr         = np.sqrt(np.diag(pcov))
                mu_ps        = float(popt[1])
                sigma_ps     = float(abs(popt[2]))
                sigma_err_ps = float(perr[2])
                y_exp        = gauss(bc, *popt)
                chi2_val     = float(np.sum((counts - y_exp) ** 2
                                             / np.maximum(counts, 1)))
                ndf_val      = int((counts > 0).sum()) - 3
                popt_ok      = True
            except Exception:
                pass

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(sub, bins=100, range=(lo_h, hi_h),
                    histtype="step", lw=1.5, color="navy",
                    label=f"$\\Delta T$  CFD {cfd_key}%")
            if popt_ok:
                x_fine = np.linspace(lo_h, hi_h, 600)
                ax.plot(x_fine, gauss(x_fine, *popt), "r-", lw=2)

            stats = [f"N = {len(sub)}",
                     f"Mean = {float(np.mean(sub)):.1f} ps",
                     f"Std  = {float(np.std(sub)):.1f} ps"]
            if popt_ok:
                stats += [rf"Fit $\mu$ = {mu_ps:.1f} ps",
                          rf"Fit $\sigma$ = {sigma_ps:.1f} ± {sigma_err_ps:.1f} ps"]
                if ndf_val > 0:
                    stats.append(f"$\\chi^2$/ndf = {chi2_val:.1f}/{ndf_val}"
                                 f" = {chi2_val/ndf_val:.2f}")
            stats.append(f"Bins = 100,  width = {bw:.1f} ps")
            ax.text(0.97, 0.97, "\n".join(stats),
                    transform=ax.transAxes, fontsize=9, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="grey", alpha=0.85))

            ax.set_xlabel(r"$\Delta T$ [ps]")
            ax.set_ylabel(f"Entries / {bw:.1f} ps")
            ax.set_title(f"noise RMS ±{nsig:.2f}σ  [{sig_lo_v:.3f}, {sig_hi_v:.3f}] mV"
                         f"  |  CFD {cfd_key}%  |  step {step_idx + 1}/{n_steps}",
                         fontsize=10)
            ax.legend(fontsize=10)
            ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="bottom")
            ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
                    fontsize=10, ha="right", va="bottom")
            plt.tight_layout()

            fname = (f"step{step_idx + 1:02d}_nsig{nsig:.2f}"
                     f"_cfd{cfd_key}_{file_tag}.png")
            plt.savefig(os.path.join(scan_dir, fname), dpi=120, bbox_inches="tight")
            plt.close()
            n_saved += 1

    print(f"  → noise_sigma_cut_scan_{file_tag}.png"
          f"  (N = {sigmas[0]:.1f}–{sigmas[-1]:.1f}σ, {n_steps} steps)")
    print(f"  → noise_sigma_cut_scan/ ({n_saved} dt plots)")


def plot_avg_noise_waveform(file_path, time_arr, pmax_thr_mV, plot_label,
                             file_tag, plot_dir,
                             ch1_negative=False, chunk_size=1000, fmt='knu'):
    """
    pmax < pmax_thr_mV 이벤트의 평균 파형 + ±1σ 엔벨로프.
    순수 노이즈 형태 진단용.
    """
    pmax_thr_V = pmax_thr_mV / 1e3
    wfm_sum    = None
    wfm_sum2   = None
    n_evts     = 0
    ref_len    = None

    tree_key = "Events" if fmt == 'knu' else "wfm"
    ch_key   = "ch1"    if fmt == 'knu' else "w1"

    for batch in uproot.iterate(f"{file_path}:{tree_key}", [ch_key],
                                 step_size=chunk_size, library="np"):
        raw = batch[ch_key]

        if fmt == 'knu':
            ch1 = raw.astype(np.float64)
        else:
            try:
                ch1 = np.array([np.asarray(x) for x in raw], dtype=np.float64)
            except ValueError:
                # jagged: truncate to shortest
                min_l = min(len(x) for x in raw)
                ch1 = np.array([np.asarray(x)[:min_l] for x in raw], dtype=np.float64)

        ped1 = ch1[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        v1   = -(ch1 - ped1) if ch1_negative else (ch1 - ped1)
        pmax1 = v1.max(axis=1)

        sel = v1[pmax1 < pmax_thr_V]
        if len(sel) == 0:
            continue

        # 길이 통일
        if ref_len is None:
            ref_len = sel.shape[1]
        sel = sel[:, :ref_len]

        if wfm_sum is None:
            wfm_sum  = sel.sum(axis=0)
            wfm_sum2 = (sel ** 2).sum(axis=0)
        else:
            L = min(len(wfm_sum), sel.shape[1])
            wfm_sum  = wfm_sum[:L]  + sel[:, :L].sum(axis=0)
            wfm_sum2 = wfm_sum2[:L] + (sel[:, :L] ** 2).sum(axis=0)
            ref_len = L
        n_evts += len(sel)

    if n_evts == 0 or wfm_sum is None:
        print(f"[WARNING] plot_avg_noise_waveform: pmax < {pmax_thr_mV} mV 이벤트 없음, 스킵")
        return

    wfm_mean = wfm_sum  / n_evts * 1e3   # → mV
    wfm_var  = wfm_sum2 / n_evts * 1e6 - wfm_mean ** 2
    wfm_std  = np.sqrt(np.maximum(wfm_var, 0))

    t_ns = time_arr[:ref_len] * 1e9

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t_ns, wfm_mean, color="steelblue", lw=1.8,
            label=f"Mean  (N = {n_evts:,})")
    ax.fill_between(t_ns, wfm_mean - wfm_std, wfm_mean + wfm_std,
                    color="steelblue", alpha=0.25, label=r"$\pm 1\sigma$ envelope")
    ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)

    rms_baseline = float(np.std(wfm_mean[:BASELINE_SAMPLES]))
    ax.text(0.97, 0.95,
            f"Baseline RMS = {rms_baseline:.2f} mV\nN = {n_evts:,} events",
            transform=ax.transAxes, fontsize=12, ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(f"Average Noise Waveform  (pmax < {pmax_thr_mV:.0f} mV)")
    ax.legend(fontsize=12)
    ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")
    ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
            fontsize=12, ha="right", va="bottom")
    plt.tight_layout()
    out = os.path.join(plot_dir, f"avg_noise_waveform_{file_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → avg_noise_waveform_{file_tag}.png  (N={n_evts:,})")


def plot_noise_psd(file_path, time_arr, pmax_thr_mV, plot_label, file_tag, plot_dir,
                   ch1_negative=False, chunk_size=1000, fmt='knu', max_events=5000):
    """
    pmax < pmax_thr_mV 이벤트의 평균 Power Spectral Density (one-sided, Hann window).

    주파수 해상도 = 1 / (N_samples × dt)
    Nyquist      = 1 / (2 × dt)
    최대 max_events 개만 사용 (추정 안정성 + 속도).

    출력: log/linear 두 패널, 상위 피크 자동 탐지 및 주파수 표기.
    """
    pmax_thr_V = pmax_thr_mV / 1e3
    dt  = float(time_arr[1] - time_arr[0])   # seconds
    fs  = 1.0 / dt                            # Hz

    psd_sum  = None
    psd_sum2 = None
    n_evts   = 0
    ref_len  = None

    tree_key = "Events" if fmt == 'knu' else "wfm"
    ch_key   = "ch1"    if fmt == 'knu' else "w1"

    for batch in uproot.iterate(f"{file_path}:{tree_key}", [ch_key],
                                step_size=chunk_size, library="np"):
        if n_evts >= max_events:
            break

        raw = batch[ch_key]
        if fmt == 'knu':
            ch1 = raw.astype(np.float64)
        else:
            try:
                ch1 = np.array([np.asarray(x) for x in raw], dtype=np.float64)
            except ValueError:
                min_l = min(len(x) for x in raw)
                ch1 = np.array([np.asarray(x)[:min_l] for x in raw], dtype=np.float64)

        ped1  = ch1[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        v1    = -(ch1 - ped1) if ch1_negative else (ch1 - ped1)
        pmax1 = v1.max(axis=1)

        sel = v1[pmax1 < pmax_thr_V]
        if len(sel) == 0:
            continue

        if ref_len is None:
            ref_len = sel.shape[1]
        sel = sel[:, :ref_len]

        # Hann window: spectral leakage 억제
        window   = np.hanning(ref_len)
        win_corr = float(np.sum(window ** 2))   # power normalization

        # DC 제거 후 window 적용
        sel_dc = sel - sel.mean(axis=1, keepdims=True)
        sel_w  = sel_dc * window[np.newaxis, :]

        # One-sided FFT
        fft_vals = np.fft.rfft(sel_w, axis=1)   # (n_sel, ref_len//2+1)
        power    = np.abs(fft_vals) ** 2 / win_corr   # V²

        # DC(k=0) 와 Nyquist 제외한 모든 bin × 2 (one-sided 보정)
        nf = power.shape[1]
        last = nf - (1 if ref_len % 2 == 0 else 0)
        power[:, 1:last] *= 2

        # PSD [V²/Hz] → [mV²/GHz]
        df  = fs / ref_len                # Hz per bin
        psd = power / df * 1e6 * 1e9     # mV²/GHz

        if psd_sum is None:
            psd_sum  = psd.sum(axis=0)
            psd_sum2 = (psd ** 2).sum(axis=0)
        else:
            L = min(len(psd_sum), psd.shape[1])
            psd_sum  = psd_sum[:L]  + psd[:, :L].sum(axis=0)
            psd_sum2 = psd_sum2[:L] + (psd[:, :L] ** 2).sum(axis=0)
        n_evts += len(sel)

    if n_evts == 0 or psd_sum is None:
        print(f"[WARNING] plot_noise_psd: pmax < {pmax_thr_mV} mV 이벤트 없음, 스킵")
        return

    psd_mean = psd_sum  / n_evts
    # standard error of the mean PSD
    psd_sem  = np.sqrt(np.maximum(psd_sum2 / n_evts - psd_mean ** 2, 0) / n_evts)

    freqs_GHz = np.fft.rfftfreq(ref_len, d=dt) * 1e-9   # GHz
    df_MHz    = float(freqs_GHz[1]) * 1e3                # MHz per bin

    # 상위 스펙트럼 피크 탐지 (DC 제외, 90th percentile 이상)
    dc_skip  = max(1, int(0.05e9 / (fs / ref_len)))   # skip bins below 50 MHz
    body     = psd_mean[dc_skip:]
    thr      = np.percentile(body, 90)
    peaks, _ = signal.find_peaks(body, height=thr, distance=4)
    peaks    += dc_skip   # 원래 인덱스로 복원

    # 상위 5개 피크 (전력 기준)
    top5 = peaks[np.argsort(psd_mean[peaks])[::-1][:5]] if len(peaks) > 0 else []

    # ── 플롯: log / linear 두 패널 ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, yscale in zip(axes, ("log", "linear")):
        # DC bin(k=0) 제외하고 표시
        ax.plot(freqs_GHz[1:], psd_mean[1:],
                color="steelblue", lw=1.0, label=f"Mean PSD  (N={n_evts:,})")
        ax.fill_between(freqs_GHz[1:],
                        np.maximum(psd_mean[1:] - psd_sem[1:], psd_mean[1:] * 1e-4),
                        psd_mean[1:] + psd_sem[1:],
                        color="steelblue", alpha=0.2, label="±SEM")

        for pk in top5:
            ax.axvline(freqs_GHz[pk], color="tomato", ls="--", lw=1.0, alpha=0.85)
            ax.annotate(
                f"{freqs_GHz[pk]:.2f} GHz",
                xy=(freqs_GHz[pk], psd_mean[pk]),
                xytext=(4, 6), textcoords="offset points",
                fontsize=8, color="tomato", rotation=65, va="bottom"
            )

        ax.set_xlabel("Frequency [GHz]")
        ax.set_ylabel(r"PSD [mV$^2$/GHz]")
        ax.set_xlim(0, freqs_GHz[-1])
        ax.set_yscale(yscale)
        ax.legend(fontsize=12)
        ax.text(0.02, 1.01, plot_label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="bottom")
        ax.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax.transAxes,
                fontsize=11, ha="right", va="bottom")
        ax.text(0.98, 0.02,
                f"Δf = {df_MHz:.0f} MHz/bin\n"
                f"f$_{{Nyq}}$ = {freqs_GHz[-1]:.1f} GHz\n"
                f"N = {n_evts:,} events",
                transform=ax.transAxes, fontsize=10, ha="right", va="bottom",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    plt.tight_layout()
    out = os.path.join(plot_dir, f"noise_psd_{file_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()

    peak_str = "  ".join(f"{freqs_GHz[p]:.3f}" for p in top5)
    print(f"  → noise_psd_{file_tag}.png  "
          f"(N={n_evts:,}, Δf={df_MHz:.0f} MHz, Nyq={freqs_GHz[-1]:.1f} GHz)")
    if peak_str:
        print(f"     주요 피크 [GHz]: {peak_str}")


# =============================================================================
# pmax cut 이상 / 이하 waveform overlay (2×2 패널)
# =============================================================================

def plot_waveform_pmax_overlay(file_path, time_arr, pmax_cut_mV,
                                plot_label, file_tag, plot_dir,
                                ch1_negative=False, pmax_cut_ch2_mV=0.0,
                                adc_rails=None, noise_max=20000, xlim=(0, 50),
                                chunk_size=1000, fmt='knu'):
    """
    pmax cut 기준 3그룹 × 2채널 오버레이 플롯 (2행 × 3열).
      col 0: Noise    — pmax < cut          (최대 noise_max 개)
      col 1: Signal   — pmax ≥ cut          (전체)
      col 2: Clipping — ADC rail 도달 이벤트 (전체)

    Parameters
    ----------
    adc_rails : (ch1_max, ch1_min, ch2_max, ch2_min) [V]  None 이면 자동 탐색
    noise_max : int   noise 그룹 최대 수집 이벤트 수
    xlim      : (lo, hi) [ns]  x축 표시 범위
    """
    pmax_cut_V = pmax_cut_mV / 1e3
    mcp_cut_V  = pmax_cut_ch2_mV / 1e3
    tree_key   = "Events" if fmt == "knu" else "wfm"
    ch1_key    = "ch1"    if fmt == "knu" else "w1"
    ch2_key    = "ch2"    if fmt == "knu" else "w2"

    if adc_rails is None:
        adc_rails = find_adc_rails(file_path, chunk_size, fmt=fmt)
    ch1_rmax, ch1_rmin, ch2_rmax, ch2_rmin = adc_rails

    noise_ch1, noise_ch2 = [], []   # pmax < cut           (최대 noise_max)
    sig_ch1,   sig_ch2   = [], []   # pmax ≥ cut, 비클리핑  (전체)
    clip_ch1,  clip_ch2  = [], []   # ADC rail 도달          (전체)
    ref_len = None

    for batch in uproot.iterate(f"{file_path}:{tree_key}",
                                [ch1_key, ch2_key],
                                step_size=chunk_size, library="np"):
        if fmt == "torino":
            raw1 = np.array([np.asarray(x) for x in batch[ch1_key]], dtype=np.float64)
            raw2 = np.array([np.asarray(x) for x in batch[ch2_key]], dtype=np.float64)
        else:
            raw1 = batch[ch1_key].astype(np.float64)
            raw2 = batch[ch2_key].astype(np.float64)

        if ref_len is None:
            ref_len = raw1.shape[1]

        clipped = (batch_clip_mask(raw1, ch1_rmax, ch1_rmin) |
                   batch_clip_mask(raw2, ch2_rmax, ch2_rmin))

        ped1 = raw1[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        ped2 = raw2[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        v1   = (-(raw1 - ped1) if ch1_negative else (raw1 - ped1)) * 1e3   # → mV
        v2   = -(raw2 - ped2) * 1e3                                          # → mV

        pmax1 = v1.max(axis=1)
        pmax2 = v2.max(axis=1)
        mcp_ok = (pmax2 >= mcp_cut_V * 1e3) if pmax_cut_ch2_mV > 0 else np.ones(len(v1), dtype=bool)

        for i in np.where(clipped)[0]:
            clip_ch1.append(v1[i])
            clip_ch2.append(v2[i])

        not_clipped = ~clipped
        sig_mask   = not_clipped & (pmax1 >= pmax_cut_V * 1e3) & mcp_ok
        noise_mask = not_clipped & (pmax1 <  pmax_cut_V * 1e3)

        for i in np.where(sig_mask)[0]:
            sig_ch1.append(v1[i])
            sig_ch2.append(v2[i])

        for i in np.where(noise_mask)[0]:
            if len(noise_ch1) >= noise_max:
                break
            noise_ch1.append(v1[i])
            noise_ch2.append(v2[i])

    n_noise = len(noise_ch1)
    n_sig   = len(sig_ch1)
    n_clip  = len(clip_ch1)

    if n_noise == 0 and n_sig == 0 and n_clip == 0:
        print("[WARNING] plot_waveform_pmax_overlay: 이벤트 없음, 스킵")
        return

    t_ns = time_arr[:ref_len] * 1e9

    def _draw(ax, waves, color, mean_color, alpha=0.08, lw=0.6, mean_lw=1.8):
        if not waves:
            ax.text(0.5, 0.5, "No events", transform=ax.transAxes,
                    ha="center", va="center", fontsize=11, color="gray")
            return
        arr = np.array(waves)[:, :ref_len]
        for w in arr:
            ax.plot(t_ns, w, color=color, alpha=alpha, lw=lw)
        ax.plot(t_ns, arr.mean(axis=0), color=mean_color, lw=mean_lw,
                label=f"mean  (N={len(arr):,})")

    fig, axes = plt.subplots(2, 3, figsize=(21, 9), sharex=True)
    fig.subplots_adjust(hspace=0.08, wspace=0.25)

    # row 0: CH1 (LGAD)
    _draw(axes[0, 0], noise_ch1, color="steelblue",  mean_color="midnightblue")
    _draw(axes[0, 1], sig_ch1,   color="royalblue",  mean_color="navy")
    _draw(axes[0, 2], clip_ch1,  color="mediumpurple", mean_color="indigo")

    # row 1: CH2 (MCP)
    _draw(axes[1, 0], noise_ch2, color="salmon",   mean_color="firebrick")
    _draw(axes[1, 1], sig_ch2,   color="tomato",   mean_color="darkred")
    _draw(axes[1, 2], clip_ch2,  color="orchid",   mean_color="purple")

    noise_title = f"Noise  (pmax < {pmax_cut_mV:.0f} mV)\nmax {noise_max:,} drawn"
    sig_title   = f"Signal  (pmax ≥ {pmax_cut_mV:.0f} mV)\nall events"
    clip_title  = "Clipping  (ADC rail)\nall events"
    axes[0, 0].set_title(noise_title, fontsize=12)
    axes[0, 1].set_title(sig_title,   fontsize=12)
    axes[0, 2].set_title(clip_title,  fontsize=12)

    axes[0, 0].set_ylabel("LGAD CH1 [mV]")
    axes[1, 0].set_ylabel("MCP CH2 [mV]")
    for col in range(3):
        axes[1, col].set_xlabel("Time [ns]")

    for ax in axes.flat:
        ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.6)
        ax.set_xlim(xlim)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=10)

    fig.suptitle(
        f"Waveform overlay  |  pmax cut = {pmax_cut_mV:.0f} mV  |  {plot_label}",
        fontsize=13
    )
    axes[0, 2].text(1.0, 1.08, r"$\beta$-Test @KNU", transform=axes[0, 2].transAxes,
                    fontsize=12, ha="right", va="bottom")

    plt.tight_layout()
    out = os.path.join(plot_dir, f"waveform_pmax_overlay_{file_tag}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → waveform_pmax_overlay_{file_tag}.png"
          f"  (noise N={n_noise:,}, signal N={n_sig:,}, clip N={n_clip:,})")


# =============================================================================
# 신호 이벤트별 파형 플롯 (CFD crossing + Gaussian peak fit)
# =============================================================================

def plot_signal_waveforms_cfd(file_path, time_arr, pmax_cut_mV, tmax_lo_ns, tmax_hi_ns,
                               plot_label, file_tag, plot_dir,
                               ch1_negative=False, pmax_cut_ch2_mV=0.0,
                               chunk_size=500, max_plots=200,
                               cfd_fracs=(0.20, 0.30)):
    """
    컷 통과 신호 이벤트별로 LGAD + MCP 파형, Gaussian peak fit, CFD crossing 마커를
    cfd_fracs 각 분율 디렉토리(time_resolution_cfd{N}/)에 저장.
    """
    dt      = float(time_arr[1] - time_arr[0])
    t_ns    = time_arr * 1e9
    pcut1_V = pmax_cut_mV    / 1e3
    pcut2_V = pmax_cut_ch2_mV / 1e3

    # 저장 디렉토리 준비
    cfd_dirs = {}
    for frac in cfd_fracs:
        key = int(round(frac * 100))
        d = os.path.join(plot_dir, f"time_resolution_cfd{key}")
        os.makedirs(d, exist_ok=True)
        cfd_dirs[frac] = (key, d)

    n_saved  = {frac: 0 for frac in cfd_fracs}
    evt_idx  = 0     # 전체 이벤트 일련번호 (파일명용)

    CFD_COLORS = {0.20: ("royalblue",   "steelblue"),    # (LGAD marker, MCP marker)
                  0.30: ("darkorange",  "saddlebrown")}

    def _gauss_popt(v, imax):
        """피크 근방 7포인트 Gaussian fit → popt or None."""
        lo = max(0, imax - 3)
        hi = min(len(v), imax + 4)
        if hi - lo < 5:
            return None
        try:
            v_raw = float(v[imax])
            t_raw = float(time_arr[imax])
            p0 = [v_raw, t_raw, 3 * dt]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    gaussian, time_arr[lo:hi], v[lo:hi], p0=p0, maxfev=500,
                    bounds=([0, t_raw - 10*dt, dt*0.5],
                            [v_raw * 3, t_raw + 10*dt, 20*dt]))
            return popt if popt[0] > 0 else None
        except Exception:
            return None

    for batch in uproot.iterate(f"{file_path}:Events", ["ch1", "ch2"],
                                 step_size=chunk_size, library="np"):
        if all(n_saved[f] >= max_plots for f in cfd_fracs):
            break

        ch1 = batch["ch1"].astype(np.float64)
        ch2 = batch["ch2"].astype(np.float64)

        ped1   = ch1[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        ped2   = ch2[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        v1_all = -(ch1 - ped1) if ch1_negative else (ch1 - ped1)
        v2_all = -(ch2 - ped2)

        pmax1_all    = v1_all.max(axis=1)
        pmax2_all    = v2_all.max(axis=1)
        imax1_all    = v1_all.argmax(axis=1)
        tmax1_ns_all = time_arr[imax1_all] * 1e9

        sel = ((pmax1_all    >= LOOSE_THR_CH1) &
               (pmax1_all    >= pcut1_V) &
               (pmax2_all    >= pcut2_V) &
               (tmax1_ns_all >= tmax_lo_ns) &
               (tmax1_ns_all <= tmax_hi_ns))

        for i in np.where(sel)[0]:
            evt_idx += 1
            if all(n_saved[f] >= max_plots for f in cfd_fracs):
                break

            v1    = v1_all[i]
            v2    = v2_all[i]
            imax1 = int(imax1_all[i])
            imax2 = int(v2.argmax())

            # Gaussian peak fit (전체 popt 필요)
            popt1 = _gauss_popt(v1, imax1)
            popt2 = _gauss_popt(v2, imax2)
            pmax1_fit = float(popt1[0]) if popt1 is not None else float(v1[imax1])
            pmax2_fit = float(popt2[0]) if popt2 is not None else float(v2[imax2])

            # CFD crossing (모든 분율 한 번에 계산)
            toa1, toa2 = {}, {}
            for frac in cfd_fracs:
                t_c1, _ = get_rising_cfd(v1, time_arr, pmax1_fit, frac)
                t_c2, _ = get_rising_cfd(v2, time_arr, pmax2_fit, frac)
                toa1[frac] = t_c1
                toa2[frac] = t_c2

            # 표시 시간 범위: LGAD 피크 중심 ±4 ns
            t_center = float(time_arr[imax1]) * 1e9
            idx_lo   = max(0, int(np.searchsorted(t_ns, t_center - 4.0)))
            idx_hi   = min(len(t_ns), int(np.searchsorted(t_ns, t_center + 6.0)))
            t_plot   = t_ns[idx_lo:idx_hi]

            # Gaussian fit 곡선용 fine grid (피크 근방 7포인트 범위 기준)
            lo1 = max(0, imax1 - 3);  hi1 = min(len(v1), imax1 + 4)
            lo2 = max(0, imax2 - 3);  hi2 = min(len(v2), imax2 + 4)
            t_fine1 = np.linspace(time_arr[lo1], time_arr[hi1 - 1], 200)
            t_fine2 = np.linspace(time_arr[lo2], time_arr[hi2 - 1], 200)

            for frac in cfd_fracs:
                if n_saved[frac] >= max_plots:
                    continue
                key, cfd_dir = cfd_dirs[frac]
                col_lgad, col_mcp = CFD_COLORS.get(frac, ("purple", "darkviolet"))

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                               sharex=True, gridspec_kw={"hspace": 0.08})

                # ── LGAD 패널 ────────────────────────────────────────────
                ax1.plot(t_plot, v1[idx_lo:idx_hi] * 1e3,
                         color="steelblue", lw=1.5, label="LGAD (CH1)")
                if popt1 is not None:
                    ax1.plot(t_fine1 * 1e9, gaussian(t_fine1, *popt1) * 1e3,
                             color="red", lw=2, ls="--",
                             label=f"Gauss fit  pmax = {pmax1_fit*1e3:.1f} mV")
                t1c = toa1[frac]
                if np.isfinite(t1c):
                    lvl1 = frac * pmax1_fit * 1e3
                    ax1.axvline(t1c * 1e9, color=col_lgad, ls="--", lw=2,
                                label=f"CFD {key}% = {t1c*1e9:.4f} ns")
                    ax1.axhline(lvl1, color=col_lgad, ls=":", lw=1.2, alpha=0.7)
                    ax1.annotate(f"{key}%", xy=(t1c * 1e9, lvl1),
                                 xytext=(4, 4), textcoords="offset points",
                                 fontsize=9, color=col_lgad)
                ax1.set_ylabel("Amplitude [mV]", fontsize=12)
                ax1.legend(fontsize=10, loc="upper right")
                ax1.text(0.02, 1.01, plot_label, transform=ax1.transAxes,
                         fontsize=11, fontweight="bold", va="bottom")
                ax1.text(1.0, 1.01, r"$\beta$-Test @KNU", transform=ax1.transAxes,
                         fontsize=10, ha="right", va="bottom")

                # ── MCP 패널 ─────────────────────────────────────────────
                ax2.plot(t_plot, v2[idx_lo:idx_hi] * 1e3,
                         color="tomato", lw=1.5, label="MCP (CH2)")
                if popt2 is not None:
                    ax2.plot(t_fine2 * 1e9, gaussian(t_fine2, *popt2) * 1e3,
                             color="darkred", lw=2, ls="--",
                             label=f"Gauss fit  pmax = {pmax2_fit*1e3:.1f} mV")
                t2c = toa2[frac]
                if np.isfinite(t2c):
                    lvl2 = frac * pmax2_fit * 1e3
                    ax2.axvline(t2c * 1e9, color=col_mcp, ls="--", lw=2,
                                label=f"CFD {key}% = {t2c*1e9:.4f} ns")
                    ax2.axhline(lvl2, color=col_mcp, ls=":", lw=1.2, alpha=0.7)
                    ax2.annotate(f"{key}%", xy=(t2c * 1e9, lvl2),
                                 xytext=(4, 4), textcoords="offset points",
                                 fontsize=9, color=col_mcp)
                ax2.set_xlabel("Time [ns]", fontsize=12)
                ax2.set_ylabel("Amplitude [mV]", fontsize=12)
                ax2.legend(fontsize=10, loc="upper right")

                # suptitle: ΔT
                dt_ps = (t2c - t1c) * 1e12 if (np.isfinite(t1c) and np.isfinite(t2c)) else float("nan")
                dt_str = f"ΔT = {dt_ps:.1f} ps" if np.isfinite(dt_ps) else "ΔT = N/A"
                fig.suptitle(f"Event #{evt_idx:05d}  |  CFD {key}%  |  {dt_str}",
                             fontsize=13, y=1.01)

                plt.savefig(os.path.join(cfd_dir, f"evt{evt_idx:05d}_{file_tag}.png"),
                            dpi=100, bbox_inches="tight")
                plt.close()
                n_saved[frac] += 1

    for frac in cfd_fracs:
        key, _ = cfd_dirs[frac]
        print(f"  → time_resolution_cfd{key}/ ({n_saved[frac]} event waveforms)")


# =============================================================================
# Torino wfm 포맷 지원
# =============================================================================

def detect_format(file_path):
    """ROOT 파일 포맷 감지: 'knu' (Events 트리) 또는 'torino' (wfm 트리)."""
    with uproot.open(file_path) as f:
        tree_names = {k.split(';')[0] for k in f.keys()}
    if 'Events' in tree_names:
        return 'knu'
    if 'wfm' in tree_names:
        return 'torino'
    raise ValueError(f"알 수 없는 ROOT 포맷. 트리 목록: {tree_names}")


def scan_events_torino(file_path, chunk_size=1000, ch1_negative=False,
                       mcp_cut_mV=0.0, tmax_lo_ns=None, tmax_hi_ns=None):
    """
    Torino wfm 포맷용 Pass 1 스캔.
    각 이벤트마다 개별 t1/t2 배열이 있으나 일정하다고 가정해 첫 이벤트를 기준 시간축으로 사용.
    반환: (time1_arr, time2_arr, pmax_all [mV], tmax_all [ns], pmax2_all [mV])
    """
    with uproot.open(file_path) as f:
        first = f["wfm"].arrays(["t1", "t2"], entry_stop=1, library="np")
        time1_arr = np.asarray(first["t1"][0], dtype=np.float64)
        time2_arr = np.asarray(first["t2"][0], dtype=np.float64)

    print(f"[Pass 1] 스캔 시작 (Torino format): {file_path}")
    print(f"         t1 범위: [{time1_arr[0]*1e9:.2f} ~ {time1_arr[-1]*1e9:.2f}] ns, "
          f"샘플 수: {len(time1_arr)}, 간격: {(time1_arr[1]-time1_arr[0])*1e12:.1f} ps")

    pmax_list, tmax_list, pmax2_list = [], [], []
    n_total = 0
    mcp_cut_V = mcp_cut_mV / 1e3

    for batch in uproot.iterate(f"{file_path}:wfm", ["w1", "w2"],
                                step_size=chunk_size, library="np"):
        # 고정 길이 waveform 가정 (동일 오실로스코프 세팅)
        w1 = np.array([np.asarray(x) for x in batch["w1"]], dtype=np.float64)
        w2 = np.array([np.asarray(x) for x in batch["w2"]], dtype=np.float64)
        n_total += len(w1)

        ped1 = w1[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        ped2 = w2[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        v1 = -(w1 - ped1) if ch1_negative else (w1 - ped1)
        v2 = -(w2 - ped2)

        pmax1    = v1.max(axis=1)
        pmax2    = v2.max(axis=1)
        imax1    = v1.argmax(axis=1)
        tmax1_ns = time1_arr[imax1] * 1e9

        loose = (pmax1 >= LOOSE_THR_CH1) & (pmax2 >= LOOSE_THR_CH2)
        if mcp_cut_V > 0:
            loose &= (pmax2 >= mcp_cut_V)
        if tmax_lo_ns is not None and tmax_hi_ns is not None:
            loose &= (tmax1_ns >= tmax_lo_ns) & (tmax1_ns <= tmax_hi_ns)

        if loose.sum() > 0:
            pmax_list.append(pmax1[loose] * 1e3)
            tmax_list.append(tmax1_ns[loose])
            pmax2_list.append(pmax2[loose] * 1e3)

    pmax_all  = np.concatenate(pmax_list)  if pmax_list  else np.array([])
    tmax_all  = np.concatenate(tmax_list)  if tmax_list  else np.array([])
    pmax2_all = np.concatenate(pmax2_list) if pmax2_list else np.array([])

    print(f"[Pass 1] 완료: {n_total} 이벤트 스캔, loose 통과: {len(pmax_all)}")
    return time1_arr, time2_arr, pmax_all, tmax_all, pmax2_all


def process_events_torino(file_path, time1_arr, time2_arr,
                           pmax_cut_mV, tmax_lo_ns, tmax_hi_ns,
                           chunk_size=1000, ch1_negative=False,
                           pmax_cut_ch2_mV=0.0):
    """
    Torino wfm 포맷용 Pass 2 정밀 분석.
    CH1(LGAD): time1_arr, CH2(MCP): time2_arr 사용.
    noise_rms 컷은 main()에서 mask로 적용 (비교 플롯 생성을 위해 모든 이벤트 저장).
    """
    dt1 = float(time1_arr[1] - time1_arr[0])
    dt2 = float(time2_arr[1] - time2_arr[0])
    pmax_cut_V     = pmax_cut_mV     / 1e3
    pmax_cut_ch2_V = pmax_cut_ch2_mV / 1e3
    records = []
    n_total = 0

    print(f"[Pass 2] 정밀 분석 시작 (Torino format)")
    print(f"         컷: pmax >= {pmax_cut_mV:.0f} mV, tmax {tmax_lo_ns:.3f}~{tmax_hi_ns:.3f} ns")

    for batch in uproot.iterate(f"{file_path}:wfm", ["w1", "w2"],
                                step_size=chunk_size, library="np"):
        w1 = np.array([np.asarray(x) for x in batch["w1"]], dtype=np.float64)
        w2 = np.array([np.asarray(x) for x in batch["w2"]], dtype=np.float64)
        n_total += len(w1)

        ped1 = w1[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        ped2 = w2[:, :BASELINE_SAMPLES].mean(axis=1, keepdims=True)
        v1_all = -(w1 - ped1) if ch1_negative else (w1 - ped1)
        v2_all = -(w2 - ped2)

        pmax1_all    = v1_all.max(axis=1)
        pmax2_all    = v2_all.max(axis=1)
        imax1_all    = v1_all.argmax(axis=1)
        tmax1_ns_all = time1_arr[imax1_all] * 1e9

        cut = ((pmax1_all    >= LOOSE_THR_CH1) &
               (pmax2_all    >= LOOSE_THR_CH2) &
               (pmax1_all    >= pmax_cut_V) &
               (pmax2_all    >= pmax_cut_ch2_V) &
               (tmax1_ns_all >= tmax_lo_ns) &
               (tmax1_ns_all <= tmax_hi_ns))

        for i in np.where(cut)[0]:
            v1 = v1_all[i]
            v2 = v2_all[i]
            noise1_rms = float(np.std(v1[:BASELINE_SAMPLES]))
            noise2_rms = float(np.std(v2[:BASELINE_SAMPLES]))

            imax1 = int(imax1_all[i])
            pmax1, tmax1 = gaussian_peak_fit(v1, time1_arr, imax1, dt1)
            if pmax1 <= 0:
                continue

            toa_rise1  = {}
            dvdt_rise1 = {}
            for frac in CFD_FRACS + [0.9]:
                t_c, dv_c = get_rising_cfd(v1, time1_arr, pmax1, frac)
                toa_rise1[frac]  = t_c
                dvdt_rise1[frac] = dv_c

            t10 = toa_rise1.get(0.1, np.nan)
            t90 = toa_rise1.get(0.9, np.nan)
            rise_time = (t90 - t10) * 1e12 if (np.isfinite(t10) and np.isfinite(t90)) else np.nan

            toa_fall1 = {}
            for frac in CFD_FRACS:
                toa_fall1[frac] = get_falling_cfd(v1, time1_arr, pmax1, frac)

            charge_vs = compute_area_new(v1, time1_arr, imax1, pmax1)

            imax2 = int(np.argmax(v2))
            pmax2, _ = gaussian_peak_fit(v2, time2_arr, imax2, dt2)
            if pmax2 <= 0:
                continue

            toa_rise2 = {}
            for frac in CFD_FRACS:
                t_c2, _ = get_rising_cfd(v2, time2_arr, pmax2, frac)
                toa_rise2[frac] = t_c2

            rec = {
                "pmax"      : pmax1 * 1e3,
                "pmax_mcp"     : pmax2 * 1e3,
                "tmax"         : tmax1 * 1e9,
                "noise_rms"    : noise1_rms * 1e3,
                "noise_rms_mcp": noise2_rms * 1e3,
                "charge"    : charge_vs * 1e12 if np.isfinite(charge_vs) else np.nan,
                "rise_time" : rise_time,
            }
            for frac in CFD_FRACS:
                key = int(frac * 100)
                t1c = toa_rise1[frac]
                t2c = toa_rise2[frac]
                dv  = dvdt_rise1[frac]
                tf  = toa_fall1[frac]
                rec[f"toa_{key}"]         = t1c * 1e9 if np.isfinite(t1c) else np.nan
                rec[f"dt_{key}"]          = (t2c - t1c) * 1e12 if (np.isfinite(t1c) and np.isfinite(t2c)) else np.nan
                rec[f"dvdt_{key}"]        = dv  / 1e6 if np.isfinite(dv)  else np.nan
                rec[f"pulse_width_{key}"] = (tf - t1c) * 1e12 if (np.isfinite(tf) and np.isfinite(t1c)) else np.nan
            records.append(rec)

    print(f"[Pass 2] 완료: {len(records)} / {n_total} 이벤트 분석")
    return records


# =============================================================================
# main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="KNU LGAD improved analysis")
    parser.add_argument("--file",         required=True,       help="ROOT 파일 경로")
    parser.add_argument("--voltage",      required=True,       help="바이어스 전압 (레이블용, 예: 330)")
    parser.add_argument("--outdir",       default=".",         help="출력 루트 디렉토리 (기본: CWD)")
    parser.add_argument("--pmax-cut",     type=float,          default=None,
                        help="LGAD pmax 컷 하한 [mV] (미지정 시 자동)")
    parser.add_argument("--pmax-cut-hi",  type=float,          default=0.0,
                        help="LGAD pmax 컷 상한 [mV] (기본: 0, 컷 없음). 예: --pmax-cut-hi 500")
    parser.add_argument("--pmax-bins",    type=float, nargs=3, default=None,
                        metavar=("N", "LO", "HI"),
                        help="pmax 히스토그램: bin 수, 최솟값, 최댓값 [mV] (기본: 100개, 자동범위)")
    parser.add_argument("--tmax-window",  type=float, nargs=2, default=[25.0, 25.0],
                        metavar=("CENTER", "HALFWIDTH"),
                        help="tmax 컷 [ns]: center ± halfwidth (기본: 25.0 25.0 → 0~50 ns)")
    parser.add_argument("--chunk-size",   type=int,            default=1000,
                        help="uproot.iterate 청크 크기 (기본: 1000)")
    parser.add_argument("--ch1-negative", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="CH1 음극성 반전 (기본: on). 비활성화: --no-ch1-negative")
    parser.add_argument("--mcp-cut",      type=float,          default=0.0,
                        help="MCP(CH2) pmax 컷 하한 [mV] (기본: 0, 컷 없음)")
    parser.add_argument("--mcp-cut-hi",   type=float,          default=0.0,
                        help="MCP(CH2) pmax 컷 상한 [mV] (기본: 0, 컷 없음). 예: --mcp-cut-hi 300")
    parser.add_argument("--amp",          default="",
                        help="앰프 표기 (예: 20dB, 40dB). 미지정 시 레이블에 포함하지 않음")
    parser.add_argument("--noise-cut",       type=float, default=0.0,
                        help="noise_rms 절댓값 상한 컷 [mV] (기본: 0, 컷 없음). 예: --noise-cut 10")
    parser.add_argument("--noise-sigma-cut", type=float, default=0.0,
                        help="noise_rms 시그마 컷: |noise_rms - mean| <= N×σ (기본: 0, 컷 없음)"
                             ". 예: --noise-sigma-cut 3")
    parser.add_argument("--noise-upper-sigma-cut", type=float, default=0.0,
                        help="noise_rms 상한 자동 컷: noise_rms <= mean + N×σ (기본: 0, 컷 없음)."
                             " 낮은 noise 이벤트는 보존, 상위 꼬리만 제거. 예: --noise-upper-sigma-cut 1")
    parser.add_argument("--noise-psd-thr", type=float,         default=0.0,
                        help="noise waveform/PSD 플롯용 pmax 상한 [mV]"
                             " (기본: auto = max(20, pmax_cut×0.5)). 예: --noise-psd-thr 200")
    parser.add_argument("--mcp-cut-scan", type=float, nargs=3,
                        metavar=("N_STEPS", "LO_MV", "HI_MV"),
                        help="MCP threshold scan: --mcp-cut-scan 30 60 120"
                             " → 60–120 mV 를 30 단계로 스캔해 σ_dT vs MCP 플롯 생성."
                             " 지정 시 --mcp-cut 은 Pass2 hard cut 에서 제외됨.")
    parser.add_argument("--noise-rms-cut-scan", type=float, nargs=3,
                        metavar=("N_STEPS", "LO_MV", "HI_MV"),
                        help="noise_rms 상한 컷 스캔: --noise-rms-cut-scan 10 2 3"
                             " → 2–3 mV 를 10 단계로 스캔해 σ_dT vs noise RMS cut 플롯 생성.")
    parser.add_argument("--noise-sigma-cut-scan", type=float, nargs='+',
                        metavar="SIGMA",
                        help="noise_rms 시그마 컷 스캔: 특정 N 값 리스트 지정."
                             " 예: --noise-sigma-cut-scan 1 2 3 4 5"
                             " → 각 N 에 대해 |noise_rms - mean| <= N×σ 컷 적용 후 σ_dT 플롯.")
    parser.add_argument("--snr-cut",         type=float, default=0.0,
                        help="SNR(=pmax/noise_rms) 하한 컷 (기본: 0, 컷 없음). 예: --snr-cut 10")
    parser.add_argument("--snr-cut-scan",    type=float, nargs=3,
                        metavar=("N_STEPS", "LO", "HI"),
                        help="SNR 하한 컷 스캔: --snr-cut-scan 20 5 30"
                             " → SNR 5–30 을 20 단계로 스캔해 σ_dT vs SNR 플롯 생성.")
    parser.add_argument("--no-clip-cut",  action="store_true", default=False,
                        help="클리핑 이벤트 제거 비활성화 (기본: 활성). 양 채널 raw ADC 연속 동일 샘플 검출.")
    parser.add_argument("--no-waveform-overlay", action="store_true", default=False,
                        help="pmax cut 기준 waveform overlay 플롯 생성 비활성화.")
    parser.add_argument("--plot-waveforms", action="store_true",
                        help="신호 이벤트별 파형 플롯 생성 (time_resolution_cfd20/30/). 느림.")
    parser.add_argument("--max-waveform-plots", type=int, default=200,
                        help="--plot-waveforms 사용 시 CFD 디렉토리당 최대 이벤트 수 (기본: 200)")
    args = parser.parse_args()

    # 출력 디렉토리
    stem    = os.path.splitext(os.path.basename(args.file))[0]
    base    = os.path.join(args.outdir, stem)
    res_dir = os.path.join(base, "results")
    plt_dir = os.path.join(base, "plots")
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(plt_dir, exist_ok=True)

    voltage = args.voltage

    fmt = detect_format(args.file)
    print(f"[INFO] 파일 포맷: {fmt.upper()}")

    amp_tag    = f"_amp{args.amp}"                   if args.amp            else ""
    mcp_tag    = f"_mcp{int(args.mcp_cut)}"          if args.mcp_cut  > 0  else ""
    if args.mcp_cut_hi > 0:
        mcp_tag += f"-{int(args.mcp_cut_hi)}"
    if args.pmax_cut_hi > 0:
        mcp_tag += f"_pmaxhi{int(args.pmax_cut_hi)}"
    noise_tag  = ""
    if args.noise_cut > 0:
        noise_tag += f"_noise{int(args.noise_cut)}mV"
    if args.noise_sigma_cut > 0:
        noise_tag += f"_nsig{args.noise_sigma_cut:.1f}s"
    if args.snr_cut > 0:
        noise_tag += f"_snr{args.snr_cut:.0f}"
    file_tag   = f"{voltage}V{amp_tag}{mcp_tag}{noise_tag}"   # 예: 350V_amp20dB_mcp100_noise10mV

    amp_str    = f"  Amp {args.amp}"                         if args.amp            else ""
    if args.mcp_cut > 0 and args.mcp_cut_hi > 0:
        mcp_str = f"  MCP [{int(args.mcp_cut)}, {int(args.mcp_cut_hi)}] mV"
    elif args.mcp_cut > 0:
        mcp_str = f"  MCP > {int(args.mcp_cut)} mV"
    else:
        mcp_str = ""
    noise_str  = ""
    if args.noise_cut > 0:
        noise_str += f"  Noise<{int(args.noise_cut)}mV"
    if args.noise_sigma_cut > 0:
        noise_str += f"  Noise±{args.noise_sigma_cut:.1f}σ"
    if args.snr_cut > 0:
        noise_str += f"  SNR>{args.snr_cut:.0f}"
    plot_label = f"LGAD BV-{voltage}V{amp_str}{mcp_str}{noise_str}"

    # -------------------------------------------------------------------------
    # Pass 1 전에 수동 컷 값 미리 계산 (scan_events에 전달)
    # -------------------------------------------------------------------------
    if args.tmax_window is not None:
        center, hw   = args.tmax_window
        tmax_lo_p1   = center - hw
        tmax_hi_p1   = center + hw
    else:
        tmax_lo_p1 = tmax_hi_p1 = None

    # -------------------------------------------------------------------------
    # Pass 1: 빠른 스캔 → raw pmax·tmax 수집
    # MCP 컷과 수동 tmax 윈도우는 Pass 1에서 적용
    # -------------------------------------------------------------------------
    clip_cut = not args.no_clip_cut
    adc_rails = None
    if clip_cut and fmt != "torino":
        print(f"[INFO] 클리핑 컷 활성 — ADC 레일 탐색 중...")
        adc_rails = find_adc_rails(args.file, args.chunk_size)
        ch1_rmax, ch1_rmin, ch2_rmax, ch2_rmin = adc_rails
        ch1_range = (ch1_rmax - ch1_rmin) * 1e3
        ch2_range = (ch2_rmax - ch2_rmin) * 1e3
        ch1_vdiv  = ch1_range / 10
        ch2_vdiv  = ch2_range / 10
        ch1_lsb   = ch1_range / 256
        ch2_lsb   = ch2_range / 256
        print(f"[INFO] ADC 레일: CH1 [{ch1_rmin*1e3:.1f}, {ch1_rmax*1e3:.1f}] mV"
              f"  range {ch1_range:.1f} mV  {ch1_vdiv:.1f} mV/div  {ch1_lsb:.3f} mV/count")
        print(f"[INFO] ADC 레일: CH2 [{ch2_rmin*1e3:.1f}, {ch2_rmax*1e3:.1f}] mV"
              f"  range {ch2_range:.1f} mV  {ch2_vdiv:.1f} mV/div  {ch2_lsb:.3f} mV/count")

    if fmt == "torino":
        time_arr, time2_arr, pmax_all, tmax_all, pmax2_all = scan_events_torino(
            args.file, args.chunk_size,
            ch1_negative=args.ch1_negative,
            mcp_cut_mV=args.mcp_cut,
            tmax_lo_ns=tmax_lo_p1,
            tmax_hi_ns=tmax_hi_p1,
        )
        pmax1_clipped_all = np.array([])
        pmax2_clipped_all = np.array([])
    else:
        time_arr, pmax_all, tmax_all, pmax2_all, pmax1_clipped_all, pmax2_clipped_all = scan_events(
            args.file, args.chunk_size,
            ch1_negative=args.ch1_negative,
            mcp_cut_mV=args.mcp_cut,
            tmax_lo_ns=tmax_lo_p1,
            tmax_hi_ns=tmax_hi_p1,
            clip_cut=clip_cut,
            adc_rails=adc_rails,
        )
        time2_arr = time_arr

    if len(pmax_all) == 0:
        print("[ERROR] Pass 1: 유효 이벤트 없음. 종료.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # 컷 결정
    # -------------------------------------------------------------------------
    if args.pmax_cut is not None:
        cut_low_p  = args.pmax_cut
        p_success  = False
        p_label    = "수동"
    else:
        cut_low_p, p_success = auto_pmax_cut(pmax_all)
        p_label = "자동 성공" if p_success else "자동 fallback"

    if args.tmax_window is not None:
        cut_lo_t, cut_hi_t = tmax_lo_p1, tmax_hi_p1
        t_success = False
        t_label   = "수동"
    else:
        cut_lo_t, cut_hi_t, t_success = auto_tmax_cut(tmax_all)
        t_label = "자동 성공" if t_success else "자동 fallback"

    pmax_cut_hi  = args.pmax_cut_hi   # LGAD pmax 상한 [mV]
    mcp_cut_hi   = args.mcp_cut_hi    # MCP  pmax 상한 [mV]

    print(f"\n[컷 결정]")
    pmax_hi_str = f" ~ {pmax_cut_hi:.1f}" if pmax_cut_hi > 0 else "+"
    print(f"  pmax:      [{cut_low_p:.1f}{pmax_hi_str}] mV  ({p_label})")
    mcp_lo_str  = f">= {args.mcp_cut:.1f}" if args.mcp_cut > 0 else "없음"
    mcp_hi_str  = f", <= {mcp_cut_hi:.1f}" if mcp_cut_hi > 0 else ""
    print(f"  mcp pmax:  {mcp_lo_str}{mcp_hi_str} mV")
    print(f"  tmax:      {cut_lo_t:.3f} ~ {cut_hi_t:.3f} ns  ({t_label})")
    if args.noise_cut > 0:
        print(f"  noise_rms: <= {args.noise_cut:.1f} mV  (절댓값 컷)")
    if args.noise_sigma_cut > 0:
        print(f"  noise_rms: ±{args.noise_sigma_cut:.1f}σ  (시그마 컷, 범위는 Pass2 후 결정)")
    if args.snr_cut > 0:
        print(f"  SNR:       >= {args.snr_cut:.1f}  (pmax/noise_rms 하한 컷)")

    # -------------------------------------------------------------------------
    # Pass 2: 정밀 분석
    # MCP pmax cut 은 항상 post-hoc으로 적용하여 no-selection 비교 플롯 생성.
    # (--mcp-cut-scan 지정 시에도 동일하게 MCP hard cut 없이 진행)
    # -------------------------------------------------------------------------
    if fmt == "torino":
        records = process_events_torino(args.file, time_arr, time2_arr,
                                         cut_low_p, cut_lo_t, cut_hi_t,
                                         args.chunk_size, ch1_negative=args.ch1_negative,
                                         pmax_cut_ch2_mV=0.0)
    else:
        records = process_events(args.file, time_arr,
                                 cut_low_p, cut_lo_t, cut_hi_t,
                                 args.chunk_size, ch1_negative=args.ch1_negative,
                                 pmax_cut_ch2_mV=0.0,
                                 clip_cut=clip_cut,
                                 adc_rails=adc_rails,
                                 pmax_cut_hi_mV=pmax_cut_hi,
                                 pmax_cut_ch2_hi_mV=0.0)

    if not records:
        print("[ERROR] Pass 2: 유효 이벤트 없음. 종료.")
        sys.exit(1)

    data = records_to_arrays(records)
    data["pmax_ch1"] = data["pmax"]   # compare_benchmark.py 호환

    # LGAD / MCP noise RMS 요약 출력
    lgad_noise = data["noise_rms"][np.isfinite(data["noise_rms"])]
    if len(lgad_noise) > 0:
        med_n = float(np.median(lgad_noise))
        p84_n = float(np.percentile(lgad_noise, 84))
        print(f"\n[LGAD noise RMS] median = {med_n:.3f} mV,  84th pct = {p84_n:.3f} mV")
        print(f"  권장 threshold:  5× = {5*med_n:.1f} mV  |  "
              f"10× = {10*med_n:.1f} mV  |  "
              f"20× = {20*med_n:.1f} mV")

    if "noise_rms_mcp" in data:
        mcp_noise = data["noise_rms_mcp"][np.isfinite(data["noise_rms_mcp"])]
        if len(mcp_noise) > 0:
            med_n = float(np.median(mcp_noise))
            p84_n = float(np.percentile(mcp_noise, 84))
            print(f"\n[MCP noise RMS]  median = {med_n:.3f} mV,  84th pct = {p84_n:.3f} mV")
            print(f"  권장 threshold:  5× = {5*med_n:.1f} mV  |  "
                  f"10× = {10*med_n:.1f} mV  |  "
                  f"20× = {20*med_n:.1f} mV")

    # -------------------------------------------------------------------------
    # 앰프 이득 보정: charge [mV·ns] ÷ amp_gain → 실제 검출기 신호 기준
    # 20dB = ×10, 40dB = ×122 (측정된 effective gain)
    # -------------------------------------------------------------------------
    amp_gain = AMP_GAIN_MAP.get(args.amp, 1.0)
    if amp_gain != 1.0:
        data["charge"] = data["charge"] / amp_gain
        print(f"\n[AMP 보정] charge ÷ {amp_gain:.0f}  (--amp {args.amp},"
              f" effective gain = {amp_gain:.0f}×)")

    # -------------------------------------------------------------------------
    # noise_rms 컷 → post-hoc mask (비교 플롯을 위해 모든 이벤트 보존)
    # 두 종류 컷 AND 조합:
    #   1) 절댓값 컷: noise_rms <= noise_cut
    #   2) 시그마 컷: |noise_rms - mean| <= N × σ
    # -------------------------------------------------------------------------
    _noise_arr   = data["noise_rms"]
    _noise_finite = _noise_arr[np.isfinite(_noise_arr)]

    # 1) 절댓값 컷
    if args.noise_cut > 0:
        abs_mask   = np.isfinite(_noise_arr) & (_noise_arr <= args.noise_cut)
        n_pass_abs = int(abs_mask.sum())
        print(f"\n[Noise 절댓값 컷] {n_pass_abs}/{len(abs_mask)} 통과"
              f" ({100*n_pass_abs/len(abs_mask):.1f}%)"
              f"  (noise_rms <= {args.noise_cut:.1f} mV)")
    else:
        abs_mask = np.ones(len(_noise_arr), dtype=bool)

    # 2) 시그마 컷
    noise_sigma_bounds = None
    if args.noise_sigma_cut > 0 and len(_noise_finite) > 1:
        _mean = float(np.mean(_noise_finite))
        _std  = float(np.std(_noise_finite))
        _sig_lo = _mean - args.noise_sigma_cut * _std
        _sig_hi = _mean + args.noise_sigma_cut * _std
        noise_sigma_bounds = (_sig_lo, _sig_hi)
        sig_mask   = np.isfinite(_noise_arr) & (_noise_arr >= _sig_lo) & (_noise_arr <= _sig_hi)
        n_pass_sig = int(sig_mask.sum())
        print(f"\n[Noise 시그마 컷] {n_pass_sig}/{len(sig_mask)} 통과"
              f" ({100*n_pass_sig/len(sig_mask):.1f}%)"
              f"  (mean={_mean:.2f} ± {args.noise_sigma_cut:.1f}×{_std:.2f}"
              f" → [{_sig_lo:.2f}, {_sig_hi:.2f}] mV)")
    else:
        sig_mask = np.ones(len(_noise_arr), dtype=bool)

    # 3) 상한 자동 컷: noise_rms <= mean + N×σ
    if args.noise_upper_sigma_cut > 0 and len(_noise_finite) > 1:
        _mean = float(np.mean(_noise_finite))
        _std  = float(np.std(_noise_finite))
        _upper = _mean + args.noise_upper_sigma_cut * _std
        upper_mask = np.isfinite(_noise_arr) & (_noise_arr <= _upper)
        n_pass_upper = int(upper_mask.sum())
        print(f"\n[Noise 상한 자동 컷] {n_pass_upper}/{len(upper_mask)} 통과"
              f" ({100*n_pass_upper/len(upper_mask):.1f}%)"
              f"  (mean={_mean:.3f} + {args.noise_upper_sigma_cut:.1f}×{_std:.3f}"
              f" = {_upper:.3f} mV)")
    else:
        upper_mask = np.ones(len(_noise_arr), dtype=bool)

    # 4) SNR 하한 컷: pmax / noise_rms >= snr_cut
    if args.snr_cut > 0:
        _snr_arr  = _noise_arr.copy()
        _pmax_arr = data["pmax"]
        _ok       = np.isfinite(_noise_arr) & (_noise_arr > 0)
        _snr_arr  = np.where(_ok, _pmax_arr / _noise_arr, 0.0)
        snr_mask  = _snr_arr >= args.snr_cut
        n_pass_snr = int(snr_mask.sum())
        print(f"\n[SNR 컷] {n_pass_snr}/{len(snr_mask)} 통과"
              f" ({100*n_pass_snr/len(snr_mask):.1f}%)"
              f"  (pmax/noise_rms >= {args.snr_cut:.1f})")
    else:
        snr_mask = np.ones(len(_noise_arr), dtype=bool)

    noise_mask = abs_mask & sig_mask & upper_mask & snr_mask
    if args.noise_cut > 0 or args.noise_sigma_cut > 0 or args.noise_upper_sigma_cut > 0 or args.snr_cut > 0:
        n_pass = int(noise_mask.sum())
        n_all  = len(noise_mask)
        print(f"[Noise/SNR 컷 통합] {n_pass}/{n_all} 통과 ({100*n_pass/n_all:.1f}%)")

    # Post-hoc MCP pmax 컷 (Pass 2 에서 제거 → no-selection 비교 플롯 가능)
    if args.mcp_cut > 0 and "pmax_mcp" in data:
        mcp_lo_mask = np.isfinite(data["pmax_mcp"]) & (data["pmax_mcp"] >= args.mcp_cut)
        if mcp_cut_hi > 0:
            mcp_lo_mask &= (data["pmax_mcp"] <= mcp_cut_hi)
        n_mcp      = int(mcp_lo_mask.sum())
        n_total_ev = len(mcp_lo_mask)
        mcp_range_str = f">= {args.mcp_cut:.0f}"
        if mcp_cut_hi > 0:
            mcp_range_str += f", <= {mcp_cut_hi:.0f}"
        print(f"\n[MCP pmax 컷 (post-hoc)] {n_mcp}/{n_total_ev} 통과"
              f" ({100*n_mcp/n_total_ev:.1f}%)"
              f"  (pmax_mcp {mcp_range_str} mV)")
    else:
        mcp_lo_mask = np.ones(len(data["pmax"]), dtype=bool)

    cut_mask = noise_mask & mcp_lo_mask

    # -------------------------------------------------------------------------
    # .npy 저장 (noise cut 전 전체 이벤트 저장 → downstream에서 직접 필터 가능)
    # -------------------------------------------------------------------------
    mcp_tag_npy = f"_mcp{int(args.mcp_cut)}mV" if args.mcp_cut > 0 else ""
    npy_path = os.path.join(res_dir, f"stats_{stem}_pmax{int(cut_low_p)}mV{mcp_tag_npy}.npy")
    np.save(npy_path, data)
    print(f"\n[저장] {npy_path}  (전체 {len(data['pmax'])}개 이벤트, noise_rms 필드 포함)")

    # -------------------------------------------------------------------------
    # 플랏 생성
    # -------------------------------------------------------------------------
    print("[INFO] 플랏 생성 중 ...")
    if args.pmax_bins is not None:
        _pb_n = int(args.pmax_bins[0])
        _pb_range = (float(args.pmax_bins[1]), float(args.pmax_bins[2]))
    else:
        _pb_n, _pb_range = 100, None
    plot_pmax_tmax_dist(pmax_all, tmax_all, pmax2_all, cut_low_p, cut_lo_t, cut_hi_t, args.mcp_cut,
                        plot_label, file_tag, plt_dir, pmax_bins=_pb_n, pmax_range=_pb_range,
                        cut_hi_p=pmax_cut_hi, mcp_cut_hi=mcp_cut_hi,
                        pmax1_clipped=pmax1_clipped_all)
    plot_mcp_pmax_dist(pmax2_all, args.mcp_cut, mcp_cut_hi, plot_label, file_tag, plt_dir,
                       pmax1_all=pmax_all, lgad_pmax_cut=cut_low_p,
                       pmax1_clipped=pmax1_clipped_all, pmax2_clipped=pmax2_clipped_all)
    plot_charge_dist(data, cut_mask, plot_label, file_tag, plt_dir)
    sigmas = plot_time_resolution(data, cut_mask, plot_label, file_tag, plt_dir)
    plot_summary_dist(data, cut_mask, plot_label, file_tag, plt_dir, sigmas)
    plot_noise_rms(data, cut_mask, voltage, plt_dir, pmax_min=cut_low_p)
    plot_dvdt(data, cut_mask, voltage, plt_dir)
    plot_snr(data, cut_mask, voltage, plt_dir)
    plot_toa_dist(data, cut_mask, plot_label, file_tag, plt_dir)

    n_plots = 9
    if args.noise_cut > 0 or args.noise_sigma_cut > 0:
        plot_noise_cut_comparison(data, noise_mask, args.noise_cut, plot_label, file_tag, plt_dir,
                                  noise_sigma_bounds=noise_sigma_bounds)
        n_plots += 1
    if args.mcp_cut > 0:
        plot_mcp_cut_comparison(data, noise_mask, mcp_lo_mask, args.mcp_cut, mcp_cut_hi,
                                plot_label, file_tag, plt_dir)
        n_plots += 1

    # noise waveform/PSD 용 pmax 상한: 미지정 시 pmax_cut × 0.5 (최소 20 mV)
    if args.noise_psd_thr > 0:
        noise_psd_thr = args.noise_psd_thr
    else:
        noise_psd_thr = max(20.0, cut_low_p * 0.5)
    print(f"[INFO] noise waveform/PSD pmax 상한: {noise_psd_thr:.0f} mV"
          f"  ({'수동' if args.noise_psd_thr > 0 else 'auto'})")

    # 진단 플롯 4종 (항상 생성)
    plot_noise_rms_vs_pmax(data, noise_mask, args.noise_cut, plot_label, file_tag, plt_dir,
                           noise_sigma_bounds=noise_sigma_bounds)
    plot_noise_rms_vs_time_resolution(data, noise_mask, args.noise_cut, plot_label, file_tag, plt_dir,
                                      noise_sigma_bounds=noise_sigma_bounds)
    plot_avg_noise_waveform(args.file, time_arr, noise_psd_thr, plot_label, file_tag, plt_dir,
                            ch1_negative=args.ch1_negative,
                            chunk_size=args.chunk_size, fmt=fmt)
    plot_noise_psd(args.file, time_arr, noise_psd_thr, plot_label, file_tag, plt_dir,
                   ch1_negative=args.ch1_negative,
                   chunk_size=args.chunk_size, fmt=fmt)
    if not args.no_waveform_overlay:
        plot_waveform_pmax_overlay(args.file, time_arr, cut_low_p,
                                   plot_label, file_tag, plt_dir,
                                   ch1_negative=args.ch1_negative,
                                   pmax_cut_ch2_mV=args.mcp_cut,
                                   adc_rails=adc_rails,
                                   chunk_size=args.chunk_size, fmt=fmt)
        n_plots += 1
    n_plots += 4

    # 신호 이벤트별 파형 플롯 (CFD20 / CFD30 디렉토리) — --plot-waveforms 로 활성화
    if args.plot_waveforms and fmt == "knu":
        print(f"\n[INFO] 신호 이벤트 파형 플롯 (CFD 20/30%, 최대 {args.max_waveform_plots}개) ...")
        plot_signal_waveforms_cfd(
            args.file, time_arr, cut_low_p, cut_lo_t, cut_hi_t,
            plot_label, file_tag, plt_dir,
            ch1_negative=args.ch1_negative,
            pmax_cut_ch2_mV=args.mcp_cut,
            chunk_size=args.chunk_size,
            max_plots=args.max_waveform_plots,
            cfd_fracs=(0.20, 0.30))
        n_plots += 1

    # MCP threshold scan
    if args.mcp_cut_scan is not None:
        n_steps, mcp_lo, mcp_hi = args.mcp_cut_scan
        print(f"\n[INFO] MCP threshold scan: {mcp_lo:.0f}–{mcp_hi:.0f} mV, {int(n_steps)} steps ...")
        plot_mcp_threshold_scan(data, cut_mask,
                                mcp_lo, mcp_hi, int(n_steps),
                                plot_label, file_tag, plt_dir)
        n_plots += 1

    # noise RMS cut scan
    if args.noise_rms_cut_scan is not None:
        n_steps, rms_lo, rms_hi = args.noise_rms_cut_scan
        print(f"\n[INFO] noise RMS cut scan: {rms_lo:.1f}–{rms_hi:.1f} mV, {int(n_steps)} steps ...")
        plot_noise_rms_cut_scan(data, cut_mask,
                                rms_lo, rms_hi, int(n_steps),
                                plot_label, file_tag, plt_dir,
                                stem=stem)
        n_plots += 1

    # SNR cut scan
    if args.snr_cut_scan is not None:
        n_steps, snr_lo, snr_hi = args.snr_cut_scan
        print(f"\n[INFO] SNR cut scan: {snr_lo:.1f}–{snr_hi:.1f}, {int(n_steps)} steps ...")
        plot_snr_cut_scan(data, cut_mask,
                          snr_lo, snr_hi, int(n_steps),
                          plot_label, file_tag, plt_dir)
        n_plots += 1

    # noise sigma cut scan
    if args.noise_sigma_cut_scan is not None:
        _svals = args.noise_sigma_cut_scan
        print(f"\n[INFO] noise sigma cut scan: N = {_svals}  ({len(_svals)} steps) ...")
        plot_noise_sigma_cut_scan(data, cut_mask,
                                  _svals,
                                  plot_label, file_tag, plt_dir)
        n_plots += 1

    print(f"\n[완료] 플랏 {n_plots}종 → {plt_dir}/")
    print(f"        결과 파일 → {npy_path}")


if __name__ == "__main__":
    main()
