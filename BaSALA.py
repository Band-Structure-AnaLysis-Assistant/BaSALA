import sys
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, find_peaks
from io import StringIO
from dataclasses import dataclass

# ==========================================
# 0. リソースパス管理
# ==========================================
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ==========================================
# 1. アプリケーション設定・定数管理
# ==========================================
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

@dataclass(frozen=True)
class AppConfig:
    # --- アプリ全体設定 ---
    APP_NAME: str = "BaSALA - Band Structure AnaLysis Assistant"
    VERSION: str = "v1.1.0"
    WINDOW_SIZE: str = "1280x900"
    SIDEBAR_WIDTH: int = 280
    
    # --- グラフ描画・UIカラー ---
    COLOR_RAW: str = "#4a90e2"      # 生データ (Raw Spectrum)
    COLOR_BG: str = "gray"          # Shirley BG
    COLOR_CORRECTED: str = "#4a90e2"# 補正後データ
    
    COLOR_FIT_BASE: str = "blue"    # フィッティング線 (Base)
    COLOR_FIT_SLOPE: str = "red"    # フィッティング線 (Slope)
    COLOR_FIT_SLOPE_UPS: str = "cyan" # UPS Cutoff用 (Slope)
    
    COLOR_MARKER_VBM: str = "green" # VBMマーカー
    COLOR_MARKER_BG: str = "blue"   # BandGapマーカー
    COLOR_MARKER_MAIN_PEAK: str = "green" # BandGap Main Peak
    
    # 選択範囲の塗りつぶし色マップ（LEIPSとLEETを追加）
    SELECTOR_COLORS = {
        'vbm_base': 'blue', 'vbm_slope': 'red', 'vbm_single': 'orange',
        'bg_peak': 'green', 'bg_base': 'blue', 'bg_slope': 'red', 'bg_single': 'orange',
        'ups_cutoff_base': 'blue', 'ups_cutoff_slope': 'cyan',
        'ups_fermi_base': 'blue', 'ups_fermi_slope': 'red',
        'leips_base': 'blue', 'leips_slope': 'red',  # LEIPS用
        'leet_base': 'blue', 'leet_slope': 'red'     # LEET用
    }

    # --- 解析パラメータのデフォルト値 ---
    DEFAULT_HV: float = 21.22  # He I
    
    # データロード時の自動範囲設定用オフセット (eV)
    # { 'キー': (minからのオフセット, maxからのオフセット) }
    RANGE_DEFAULTS = {
        'vbm_base': (0.0, 1.0),
        'vbm_slope': (1.5, 2.5),
        'vbm_single': (0.0, 3.0),
        'bg_peak': (0.0, 0.8),
        'bg_base': (4.0, 5.0),
        'bg_slope': (5.5, 6.5),
        'bg_single': (4.0, 7.0),
        'ups_cutoff_base': (15.0, 16.0), 
        'ups_cutoff_slope': (16.0, 17.0),
        'ups_fermi_base': (-1.0, 0.0),
        'ups_fermi_slope': (0.0, 1.0),
        # LEIPS/LEET用（仮のオフセット値。データに合わせて後で調整可能）
        'leips_base': (0.0, 1.0),
        'leips_slope': (1.5, 2.5),
        'leet_base': (0.0, 1.0),
        'leet_slope': (1.5, 2.5)
    }

# ==========================================
# 2. 数学・物理計算用 関数群
# ==========================================
def linear_func(x, a, b):
    return a * x + b

def gaussian_func(x, a, mu, sigma, c):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

def calculate_shirley_bg(x, y, tol=1e-5, max_iters=50):
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    n = len(y)
    I_start = np.mean(y_sorted[:5])
    I_end = np.mean(y_sorted[-5:])
    bg = np.full(n, I_start)

    for _ in range(max_iters):
        signal = y_sorted - bg
        signal[signal < 0] = 0
        cum_area = np.zeros(n)
        cum_area[1:] = np.cumsum((signal[:-1] + signal[1:]) / 2 * np.diff(x_sorted))
        total_area = cum_area[-1]
        if total_area == 0: k = 0
        else: k = (I_end - I_start) / total_area
        bg_new = I_start + k * cum_area
        if np.max(np.abs(bg_new - bg)) < tol:
            bg = bg_new
            break
        bg = bg_new
        
    bg_original_order = np.zeros(n)
    bg_original_order[sorted_indices] = bg
    return bg_original_order

# ==========================================
# 3. GUI コンポーネント
# ==========================================
class DataSelectionDialog(ctk.CTkToplevel):
    def __init__(self, parent, data_blocks, callback):
        super().__init__(parent)
        self.title("Select Data")
        self.geometry("350x250")
        self.data_blocks = data_blocks
        self.callback = callback
        self.grab_set(); self.focus_set()

        ctk.CTkLabel(self, text="1. Select Region:", font=("Roboto", 12, "bold")).pack(pady=(20, 5), padx=20, anchor="w")
        self.combo_region = ctk.CTkComboBox(self, values=list(data_blocks.keys()), command=self.on_region_change)
        self.combo_region.pack(pady=5, padx=20, fill="x")
        
        ctk.CTkLabel(self, text="2. Select File #:", font=("Roboto", 12, "bold")).pack(pady=(10, 5), padx=20, anchor="w")
        self.combo_file = ctk.CTkComboBox(self, values=[])
        self.combo_file.pack(pady=5, padx=20, fill="x")
        
        self.btn_ok = ctk.CTkButton(self, text="Load Data", command=self.on_ok, fg_color="#2d8d2d")
        self.btn_ok.pack(pady=20, padx=20, fill="x")
        
        if data_blocks:
            first_region = list(data_blocks.keys())[0]
            self.combo_region.set(first_region)
            self.on_region_change(first_region)

    def on_region_change(self, region):
        df = self.data_blocks[region]
        cols = [str(c) for c in df.columns[1:]] 
        self.combo_file.configure(values=cols)
        if cols: self.combo_file.set(cols[0])

    def on_ok(self):
        region = self.combo_region.get()
        file_col = self.combo_file.get()
        df = self.data_blocks[region]
        try:
            energy = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
            intensity = pd.to_numeric(df[file_col], errors='coerce').values
            mask = np.isfinite(energy) & np.isfinite(intensity) & ((energy != 0) | (intensity != 0))
            self.callback(energy[mask], intensity[mask], f"{region} ({file_col})")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to extract data: {e}")

# ==========================================
# 4. GUI アプリケーションクラス
# ==========================================
class BaSALA_App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title(f"{AppConfig.APP_NAME} ({AppConfig.VERSION})")
        self.geometry(AppConfig.WINDOW_SIZE)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        try: self.iconbitmap(resource_path("app_icon.ico"))
        except: pass

        # Data variables
        self.energy = None
        self.intensity = None
        self.intensity_corrected = None
        self.bg_data = None
        self.span = None
        self.selection_mode = None
        
        # Contexts
        self.vbm_candidates = []; self.vbm_context = {}
        self.bg_candidates = []; self.bg_context = {}

        self._create_sidebar()
        self._create_main_area()

    # -------------------------------------------------------------------------
    # UI Creation Helpers (共通化)
    # -------------------------------------------------------------------------
    def _create_range_selector(self, parent, label_text, selector_mode, min_val="", max_val=""):
        """ラベル + [Entry - Entry] + Selectボタン のセットを作成"""
        ctk.CTkLabel(parent, text=label_text, font=("Roboto", 11)).pack(anchor="w", padx=2, pady=(5, 0))
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", padx=2)
        
        e_min = ctk.CTkEntry(frame, width=65); e_min.pack(side="left")
        e_min.insert(0, str(min_val))
        ctk.CTkLabel(frame, text="-").pack(side="left", padx=2)
        e_max = ctk.CTkEntry(frame, width=65); e_max.pack(side="left")
        e_max.insert(0, str(max_val))
        
        ctk.CTkButton(frame, text="Select", width=50, fg_color="gray", 
                      command=lambda: self.activate_selector(selector_mode)).pack(side="right")
        return e_min, e_max

    # -------------------------------------------------------------------------
    # Main UI Structure
    # -------------------------------------------------------------------------
    def _create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=AppConfig.SIDEBAR_WIDTH, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar.pack_propagate(False)

        self.common_frame = ctk.CTkFrame(self.sidebar)
        self.common_frame.pack(padx=10, pady=(20, 5), fill="x") 
        
        self.load_btn = ctk.CTkButton(self.common_frame, text="Open CSV / Text", command=self.load_csv, fg_color="#1f538d", height=30)
        self.load_btn.pack(padx=5, pady=5, fill="x")
        
        self.sep_option = ctk.CTkComboBox(self.common_frame, values=[", (Comma)", "\\t (Tab)", "Space"], height=24)
        self.sep_option.set(", (Comma)")
        self.sep_option.pack(padx=5, pady=(0, 5))
        
        self.chk_shirley_var = ctk.BooleanVar(value=False)
        self.chk_shirley = ctk.CTkCheckBox(self.common_frame, text="Apply Shirley BG", variable=self.chk_shirley_var, command=self.on_shirley_toggle)
        self.chk_shirley.pack(padx=10, pady=10, anchor="w")

        self.tabview = ctk.CTkTabview(self.sidebar, width=AppConfig.SIDEBAR_WIDTH - 20)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)
        
        # --- タブの追加（LEIPSとLEETを追加） ---
        self.tab_bg = self.tabview.add("Band Gap")
        self.tab_vbm = self.tabview.add("VBM")
        self.tab_ups = self.tabview.add("UPS")
        self.tab_leips = self.tabview.add("LEIPS")
        self.tab_leet = self.tabview.add("LEET")

        # 各タブのUI初期化メソッドを呼び出し
        self._init_bandgap_tab()
        self._init_vbm_tab()
        self._init_ups_tab()
        self._init_leips_tab()
        self._init_leet_tab()

    def _init_bandgap_tab(self):
        frame = ctk.CTkFrame(self.tab_bg, fg_color="transparent")
        frame.pack(fill="both", expand=True)

        self.bg_mode_var = ctk.StringVar(value="Linear")
        self.seg_bg_mode = ctk.CTkSegmentedButton(frame, values=["Linear", "Deriv", "Hybrid"], variable=self.bg_mode_var, command=self.update_bg_ui)
        self.seg_bg_mode.pack(pady=10, padx=5, fill="x")
        self.seg_bg_mode.set("Linear")

        self.bg_peak_min, self.bg_peak_max = self._create_range_selector(frame, "1. Main Peak (Eg Reference):", "bg_peak")
        
        self.bg_input_container = ctk.CTkFrame(frame, fg_color="transparent")
        self.bg_input_container.pack(fill="x", pady=5)
        
        self.frame_bg_linear = ctk.CTkFrame(self.bg_input_container, fg_color="transparent")
        self.bg_base_min, self.bg_base_max = self._create_range_selector(self.frame_bg_linear, "2. Loss Base (Flat):", "bg_base")
        self.bg_slope_min, self.bg_slope_max = self._create_range_selector(self.frame_bg_linear, "3. Loss Slope (Rise):", "bg_slope")
        self.frame_bg_linear.pack(fill="x", pady=5)

        self.frame_bg_single = ctk.CTkFrame(self.bg_input_container, fg_color="transparent")
        self.bg_single_min, self.bg_single_max = self._create_range_selector(self.frame_bg_single, "2. Onset Search Region:", "bg_single")
        
        ctk.CTkButton(frame, text="Stop Selection", fg_color="transparent", border_width=1, height=24, command=self.deactivate_selector).pack(pady=10)
        self.calc_bg_btn = ctk.CTkButton(frame, text="Calculate Band Gap", command=self.calculate_bandgap, fg_color="#E07A5F", state="disabled")
        self.calc_bg_btn.pack(pady=5, fill="x")
        
        self.frame_bg_res_container = ctk.CTkFrame(frame, fg_color="transparent")
        self.frame_bg_res_container.pack(pady=5, fill="x")
        self.lbl_res_gap = ctk.CTkLabel(self.frame_bg_res_container, text="Eg: --- eV", font=ctk.CTkFont(size=18, weight="bold"), text_color="#E07A5F")
        self.lbl_res_gap.pack()
        
        self.frame_bg_cand_container = ctk.CTkFrame(frame, fg_color="transparent")
        self.frame_bg_cand_container.pack(pady=5, fill="x")
        self.combo_bg_candidates = ctk.CTkComboBox(self.frame_bg_cand_container, width=240, values=["Candidates (Curvature Order)"], command=self.on_bg_candidate_selected)
        
        self.update_bg_ui("Linear")

    def _init_vbm_tab(self):
        frame = ctk.CTkFrame(self.tab_vbm, fg_color="transparent")
        frame.pack(fill="both", expand=True)
        
        self.vbm_mode_var = ctk.StringVar(value="Linear")
        self.seg_vbm_mode = ctk.CTkSegmentedButton(frame, values=["Linear", "Deriv", "Hybrid"], variable=self.vbm_mode_var, command=self.update_vbm_ui)
        self.seg_vbm_mode.pack(pady=10, padx=5, fill="x")
        self.seg_vbm_mode.set("Linear")
        
        self.vbm_input_container = ctk.CTkFrame(frame, fg_color="transparent")
        self.vbm_input_container.pack(fill="x", pady=5)
        
        self.frame_vbm_linear = ctk.CTkFrame(self.vbm_input_container, fg_color="transparent")
        self.entry_vbm_base_min, self.entry_vbm_base_max = self._create_range_selector(self.frame_vbm_linear, "1. Base Range (Flat):", "vbm_base")
        self.entry_vbm_slope_min, self.entry_vbm_slope_max = self._create_range_selector(self.frame_vbm_linear, "2. Slope Range (Edge):", "vbm_slope")
        self.frame_vbm_linear.pack(fill="x", pady=5)
        
        self.frame_vbm_single = ctk.CTkFrame(self.vbm_input_container, fg_color="transparent")
        self.entry_vbm_single_min, self.entry_vbm_single_max = self._create_range_selector(self.frame_vbm_single, "1. Search Region:", "vbm_single")
        
        self.btn_reset_mode_vbm = ctk.CTkButton(frame, text="Stop Selection", fg_color="transparent", border_width=1, height=24, command=self.deactivate_selector)
        self.btn_reset_mode_vbm.pack(pady=10)
        
        self.calc_vbm_btn = ctk.CTkButton(frame, text="Calculate VBM", command=self.calculate_vbm, fg_color="#2d8d2d", state="disabled")
        self.calc_vbm_btn.pack(padx=5, pady=15, fill="x")
        
        self.frame_vbm_res_container = ctk.CTkFrame(frame, fg_color="transparent")
        self.frame_vbm_res_container.pack(pady=5, fill="x")
        self.vbm_label = ctk.CTkLabel(self.frame_vbm_res_container, text="VBM: --- eV", font=ctk.CTkFont(size=18, weight="bold"), text_color="#4db6ac")
        self.vbm_label.pack()
        
        self.frame_vbm_cand_container = ctk.CTkFrame(frame, fg_color="transparent")
        self.frame_vbm_cand_container.pack(pady=5, fill="x")
        self.combo_vbm_candidates = ctk.CTkComboBox(self.frame_vbm_cand_container, width=240, values=["Candidates (Curvature Order)"], command=self.on_vbm_candidate_selected)
        
        self.update_vbm_ui("Linear")

    def _init_ups_tab(self):
        frame = ctk.CTkFrame(self.tab_ups, fg_color="transparent")
        frame.pack(fill="both", expand=True)

        param_frame = ctk.CTkFrame(frame, fg_color="transparent")
        param_frame.pack(fill="x", pady=(5, 10))
        ctk.CTkLabel(param_frame, text="Photon Energy (eV):", font=("Roboto", 12)).pack(side="left", padx=2)
        self.entry_ups_hv = ctk.CTkEntry(param_frame, width=70)
        self.entry_ups_hv.pack(side="right", padx=5)
        self.entry_ups_hv.insert(0, str(AppConfig.DEFAULT_HV)) 

        ctk.CTkLabel(frame, text="1. SE Cutoff (High BE):", font=("Roboto", 12, "bold"), text_color="#ffa726").pack(anchor="w", pady=(5, 0))
        self.frame_ups_cutoff = ctk.CTkFrame(frame, fg_color="transparent")
        self.frame_ups_cutoff.pack(fill="x", pady=2)
        self.entry_ups_cutoff_base_min, self.entry_ups_cutoff_base_max = self._create_range_selector(self.frame_ups_cutoff, "Base Range:", "ups_cutoff_base")
        self.entry_ups_cutoff_slope_min, self.entry_ups_cutoff_slope_max = self._create_range_selector(self.frame_ups_cutoff, "Slope Range:", "ups_cutoff_slope")

        ctk.CTkLabel(frame, text="2. Fermi / HOMO (Low BE):", font=("Roboto", 12, "bold"), text_color="#4db6ac").pack(anchor="w", pady=(15, 0))
        self.frame_ups_fermi = ctk.CTkFrame(frame, fg_color="transparent")
        self.frame_ups_fermi.pack(fill="x", pady=2)
        self.entry_ups_fermi_base_min, self.entry_ups_fermi_base_max = self._create_range_selector(self.frame_ups_fermi, "Base Range:", "ups_fermi_base")
        self.entry_ups_fermi_slope_min, self.entry_ups_fermi_slope_max = self._create_range_selector(self.frame_ups_fermi, "Slope Range:", "ups_fermi_slope")

        ctk.CTkButton(frame, text="Stop Selection", fg_color="transparent", border_width=1, height=24, command=self.deactivate_selector).pack(pady=10)
        self.calc_ups_btn = ctk.CTkButton(frame, text="Calculate WF & IP", command=self.calculate_ups, fg_color="#e65100", state="disabled")
        self.calc_ups_btn.pack(pady=5, fill="x")
        
        self.frame_ups_res = ctk.CTkFrame(frame, fg_color="transparent")
        self.frame_ups_res.pack(pady=5, fill="x")
        self.lbl_res_wf = ctk.CTkLabel(self.frame_ups_res, text="WF (Φ): --- eV", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_res_wf.pack(anchor="center", padx=10)
        self.lbl_res_ip = ctk.CTkLabel(self.frame_ups_res, text="IP: --- eV", font=ctk.CTkFont(size=16, weight="bold"))
        self.lbl_res_ip.pack(anchor="center", padx=10)

    # --- 新規追加：LEIPSタブ ---
    def _init_leips_tab(self):
        """LEIPSデータからLUMO（伝導帯下端）を直線交点で算出するUI"""
        frame = ctk.CTkFrame(self.tab_leips, fg_color="transparent")
        frame.pack(fill="both", expand=True)

        ctk.CTkLabel(frame, text="LUMO Analysis (Linear Intersection):", font=("Roboto", 12, "bold"), text_color="#4db6ac").pack(anchor="w", pady=(5, 0))
        
        self.frame_leips = ctk.CTkFrame(frame, fg_color="transparent")
        self.frame_leips.pack(fill="x", pady=5)
        
        self.entry_leips_base_min, self.entry_leips_base_max = self._create_range_selector(self.frame_leips, "1. Base Range (Gap):", "leips_base")
        self.entry_leips_slope_min, self.entry_leips_slope_max = self._create_range_selector(self.frame_leips, "2. Slope Range (LUMO Edge):", "leips_slope")

        ctk.CTkButton(frame, text="Stop Selection", fg_color="transparent", border_width=1, height=24, command=self.deactivate_selector).pack(pady=10)
        self.calc_leips_btn = ctk.CTkButton(frame, text="Calculate LUMO", command=self.calculate_leips, fg_color="#2d8d2d", state="disabled")
        self.calc_leips_btn.pack(pady=5, fill="x")
        
        self.frame_leips_res = ctk.CTkFrame(frame, fg_color="transparent")
        self.frame_leips_res.pack(pady=5, fill="x")
        self.lbl_res_lumo = ctk.CTkLabel(self.frame_leips_res, text="LUMO: --- eV", font=ctk.CTkFont(size=18, weight="bold"), text_color="#4db6ac")
        self.lbl_res_lumo.pack()

    # --- 新規追加：LEETタブ ---
    def _init_leet_tab(self):
        """LEETデータから真空準位（Vacuum Level）を直線交点で算出するUI"""
        frame = ctk.CTkFrame(self.tab_leet, fg_color="transparent")
        frame.pack(fill="both", expand=True)

        ctk.CTkLabel(frame, text="Vacuum Level Analysis:", font=("Roboto", 12, "bold"), text_color="#ffa726").pack(anchor="w", pady=(5, 0))
        
        self.frame_leet = ctk.CTkFrame(frame, fg_color="transparent")
        self.frame_leet.pack(fill="x", pady=5)
        
        self.entry_leet_base_min, self.entry_leet_base_max = self._create_range_selector(self.frame_leet, "1. Base Range (Flat):", "leet_base")
        self.entry_leet_slope_min, self.entry_leet_slope_max = self._create_range_selector(self.frame_leet, "2. Slope Range (Target Current):", "leet_slope")

        ctk.CTkButton(frame, text="Stop Selection", fg_color="transparent", border_width=1, height=24, command=self.deactivate_selector).pack(pady=10)
        self.calc_leet_btn = ctk.CTkButton(frame, text="Calculate Vacuum Level", command=self.calculate_leet, fg_color="#e65100", state="disabled")
        self.calc_leet_btn.pack(pady=5, fill="x")
        
        self.frame_leet_res = ctk.CTkFrame(frame, fg_color="transparent")
        self.frame_leet_res.pack(pady=5, fill="x")
        self.lbl_res_vl = ctk.CTkLabel(self.frame_leet_res, text="Vacuum Level: --- eV", font=ctk.CTkFont(size=18, weight="bold"), text_color="#ffa726")
        self.lbl_res_vl.pack()

    def _create_main_area(self):
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.ax.set_xlabel("Binding Energy (eV)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.invert_xaxis()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.main_frame)
        self.toolbar.update()

    def update_bg_ui(self, value):
        if value in ["Linear", "Hybrid"]:
            self.frame_bg_single.pack_forget(); self.frame_bg_linear.pack(fill="x", pady=5)
        else:
            self.frame_bg_linear.pack_forget(); self.frame_bg_single.pack(fill="x", pady=5)
        if value == "Linear": self.combo_bg_candidates.pack_forget()
        else: self.combo_bg_candidates.pack()

    def update_vbm_ui(self, value):
        if value in ["Linear", "Hybrid"]:
            self.frame_vbm_single.pack_forget(); self.frame_vbm_linear.pack(fill="x", pady=5)
        else:
            self.frame_vbm_linear.pack_forget(); self.frame_vbm_single.pack(fill="x", pady=5)
        if value == "Linear": self.combo_vbm_candidates.pack_forget()
        else: self.combo_vbm_candidates.pack()

    # ==========================================
    # 5. 操作ロジック
    # ==========================================
    def get_current_intensity(self):
        if self.chk_shirley_var.get() and self.intensity_corrected is not None:
            return self.intensity_corrected
        return self.intensity

    def plot_base_graph(self):
        self.ax.clear()
        if self.chk_shirley_var.get() and self.intensity_corrected is not None:
            self.ax.plot(self.energy, self.intensity, color='gray', alpha=0.3, label='Raw Data')
            self.ax.plot(self.energy, self.bg_data, color=AppConfig.COLOR_BG, linestyle='--', alpha=0.5, label='Shirley BG')
            self.ax.plot(self.energy, self.intensity_corrected, color=AppConfig.COLOR_CORRECTED, linewidth=1.5, label='Corrected')
        else:
            self.ax.plot(self.energy, self.intensity, color=AppConfig.COLOR_RAW, linewidth=1.5, label='Raw Spectrum')
        self.ax.legend(loc='upper left'); self.ax.grid(True); self.ax.invert_xaxis()
        self.auto_scale_y()
        self.ax.set_xlabel("Binding Energy (eV)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.canvas.draw()

    def auto_scale_y(self):
        if self.intensity is None: return
        y_max_raw = np.max(self.intensity)
        y_current = self.get_current_intensity()
        y_min_curr = np.min(y_current)
        amp = y_max_raw - y_min_curr
        margin = amp * 0.05 if amp > 0 else 10.0
        self.ax.set_ylim(y_min_curr - margin, y_max_raw + margin)

    def activate_selector(self, mode):
        if self.toolbar.mode:
            if 'zoom' in self.toolbar.mode: self.toolbar.zoom()
            elif 'pan' in self.toolbar.mode: self.toolbar.pan()

        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        self.plot_base_graph()
        self.ax.set_xlim(current_xlim)
        self.ax.set_ylim(current_ylim)

        self.selection_mode = mode
        if self.span: self.span.set_visible(False); self.span = None
        
        color = AppConfig.SELECTOR_COLORS.get(mode, 'gray')
        self.span = SpanSelector(self.ax, self.on_select, 'horizontal', useblit=True, 
                                 props=dict(alpha=0.2, facecolor=color), interactive=True, drag_from_anywhere=True)
        self.canvas.draw()

    def deactivate_selector(self):
        if self.span: self.span.set_visible(False); self.span = None
        self.selection_mode = None
        self.canvas.draw()

    def on_select(self, vmin, vmax):
        min_val, max_val = sorted([vmin, vmax])
        v_min, v_max = f"{min_val:.2f}", f"{max_val:.2f}"
        
        # --- 選択範囲の登録先を追加（LEIPS / LEET対応） ---
        entries = {
            'vbm_base': (self.entry_vbm_base_min, self.entry_vbm_base_max),
            'vbm_slope': (self.entry_vbm_slope_min, self.entry_vbm_slope_max),
            'vbm_single': (self.entry_vbm_single_min, self.entry_vbm_single_max),
            'bg_peak': (self.bg_peak_min, self.bg_peak_max),
            'bg_base': (self.bg_base_min, self.bg_base_max), 
            'bg_slope': (self.bg_slope_min, self.bg_slope_max),
            'bg_single': (self.bg_single_min, self.bg_single_max),
            'ups_cutoff_base': (self.entry_ups_cutoff_base_min, self.entry_ups_cutoff_base_max),
            'ups_cutoff_slope': (self.entry_ups_cutoff_slope_min, self.entry_ups_cutoff_slope_max),
            'ups_fermi_base': (self.entry_ups_fermi_base_min, self.entry_ups_fermi_base_max),
            'ups_fermi_slope': (self.entry_ups_fermi_slope_min, self.entry_ups_fermi_slope_max),
            'leips_base': (self.entry_leips_base_min, self.entry_leips_base_max),
            'leips_slope': (self.entry_leips_slope_min, self.entry_leips_slope_max),
            'leet_base': (self.entry_leet_base_min, self.entry_leet_base_max),
            'leet_slope': (self.entry_leet_slope_min, self.entry_leet_slope_max)
        }
        
        if self.selection_mode in entries:
            e1, e2 = entries[self.selection_mode]
            e1.delete(0, tk.END); e1.insert(0, v_min)
            e2.delete(0, tk.END); e2.insert(0, v_max)

    def on_shirley_toggle(self):
        if self.energy is None: return
        if self.chk_shirley_var.get():
            try:
                self.bg_data = calculate_shirley_bg(self.energy, self.intensity)
                self.intensity_corrected = self.intensity - self.bg_data
            except: 
                messagebox.showerror("Error", "Shirley calculation failed.")
                self.chk_shirley_var.set(False)
        else: 
            self.bg_data = None; self.intensity_corrected = None
        self.plot_base_graph()

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("Data Files", "*.csv *.txt *.dat"), ("All Files", "*.*")])
        if not file_path: return
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                head_lines = [next(f) for _ in range(50)]
            is_multipak = any("file#" in line.lower() for line in head_lines)
            if is_multipak: self._load_multipak(file_path)
            else: self._load_normal_csv(file_path)
        except Exception as e: messagebox.showerror("Error", f"Failed to open file: {e}")

    def _load_multipak(self, file_path):
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: lines = f.readlines()
        data_blocks = {}
        header_indices = [i for i, line in enumerate(lines) if "file#" in line.lower()]
        for i, start_idx in enumerate(header_indices):
            region_name = lines[start_idx-2].strip() if start_idx >= 2 else f"Region {i+1}"
            if not region_name or region_name == "no area description": region_name = f"Region {i+1}"
            end_idx = max(start_idx, header_indices[i+1]-4) if i < len(header_indices)-1 else len(lines)
            try:
                df = pd.read_csv(StringIO("".join(lines[start_idx:end_idx])))
                original_name = region_name; counter = 2
                while region_name in data_blocks: region_name = f"{original_name}_{counter}"; counter += 1
                data_blocks[region_name] = df
            except: pass
        if not data_blocks: return messagebox.showerror("Error", "Could not parse MultiPak format.")
        DataSelectionDialog(self, data_blocks, self._on_data_loaded)

    def _load_normal_csv(self, file_path):
        sep = {", (Comma)": ",", "\\t (Tab)": "\t", "Space": r"\s+"}[self.sep_option.get()]
        df = pd.read_csv(file_path, sep=sep, header=None, engine='python')
        if df.shape[1] < 2: return messagebox.showerror("Error", "Invalid data format.")
        energy = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
        intensity = pd.to_numeric(df.iloc[:, 1], errors='coerce').values
        mask = np.isfinite(energy) & np.isfinite(intensity) & ((energy != 0) | (intensity != 0))
        if np.sum(mask) == 0: return messagebox.showerror("Error", "No valid numeric data found.")
        self._on_data_loaded(energy[mask], intensity[mask], "Raw Spectrum")

    def _on_data_loaded(self, energy, intensity, label_text):
        self.energy = energy
        self.intensity = intensity
        min_e = np.min(self.energy)
        
        def set_default(e_min, e_max, key):
            off_min, off_max = AppConfig.RANGE_DEFAULTS[key]
            e_min.delete(0, tk.END); e_min.insert(0, f"{min_e + off_min:.1f}")
            e_max.delete(0, tk.END); e_max.insert(0, f"{min_e + off_max:.1f}")

        # VBM
        set_default(self.entry_vbm_base_min, self.entry_vbm_base_max, 'vbm_base')
        set_default(self.entry_vbm_slope_min, self.entry_vbm_slope_max, 'vbm_slope')
        set_default(self.entry_vbm_single_min, self.entry_vbm_single_max, 'vbm_single')
        # BandGap
        set_default(self.bg_peak_min, self.bg_peak_max, 'bg_peak')
        set_default(self.bg_base_min, self.bg_base_max, 'bg_base')
        set_default(self.bg_slope_min, self.bg_slope_max, 'bg_slope')
        set_default(self.bg_single_min, self.bg_single_max, 'bg_single')
        # UPS 
        set_default(self.entry_ups_cutoff_base_min, self.entry_ups_cutoff_base_max, 'ups_cutoff_base')
        set_default(self.entry_ups_cutoff_slope_min, self.entry_ups_cutoff_slope_max, 'ups_cutoff_slope')
        set_default(self.entry_ups_fermi_base_min, self.entry_ups_fermi_base_max, 'ups_fermi_base')
        set_default(self.entry_ups_fermi_slope_min, self.entry_ups_fermi_slope_max, 'ups_fermi_slope')
        # LEIPS & LEET --- 新規追加 ---
        set_default(self.entry_leips_base_min, self.entry_leips_base_max, 'leips_base')
        set_default(self.entry_leips_slope_min, self.entry_leips_slope_max, 'leips_slope')
        set_default(self.entry_leet_base_min, self.entry_leet_base_max, 'leet_base')
        set_default(self.entry_leet_slope_min, self.entry_leet_slope_max, 'leet_slope')

        self.chk_shirley_var.set(False); self.intensity_corrected = None; self.bg_data = None
        self.plot_base_graph()
        
        # 全ての計算ボタンを有効化
        self.calc_vbm_btn.configure(state="normal")
        self.calc_bg_btn.configure(state="normal")
        self.calc_ups_btn.configure(state="normal")
        self.calc_leips_btn.configure(state="normal")
        self.calc_leet_btn.configure(state="normal")

    def _find_candidates(self, search_min, search_max, y_data):
        target_window = 21 
        w_len = min(target_window, len(y_data))
        if w_len % 2 == 0: w_len -= 1
        if w_len < 3: w_len = 3
        y_smooth_all = savgol_filter(y_data, window_length=w_len, polyorder=2)
        mask_search = (self.energy >= search_min) & (self.energy <= search_max)
        x_s = self.energy[mask_search]
        y_s_smooth = y_smooth_all[mask_search]
        if len(x_s) < 5: return [], None, None
        d2y = np.gradient(np.gradient(y_s_smooth, x_s), x_s)
        peaks, properties = find_peaks(d2y, height=0)
        cand_list = []
        if len(peaks) > 0:
            for p_idx in peaks:
                cx = x_s[p_idx]; cy = y_s_smooth[p_idx]
                raw_score = properties['peak_heights'][list(peaks).index(p_idx)]
                cand_list.append((cx, cy, raw_score))
            cand_list.sort(key=lambda x: x[2], reverse=True)
            cand_list = cand_list[:5]
        else:
            mid_idx = len(x_s) // 2
            cand_list = [(x_s[mid_idx], y_s_smooth[mid_idx], 0)]
        return cand_list, x_s, y_s_smooth

    # -------------------------------------------------------------------------
    # Calculation Logic Helpers (共通化)
    # -------------------------------------------------------------------------
    def _fit_and_intersect(self, y_data, range_bg, range_sl):
        """2つの範囲(Background, Slope)で直線フィッティングし、交点を求める"""
        mask_bg = (self.energy >= range_bg[0]) & (self.energy <= range_bg[1])
        popt_bg, _ = curve_fit(linear_func, self.energy[mask_bg], y_data[mask_bg])
        
        mask_sl = (self.energy >= range_sl[0]) & (self.energy <= range_sl[1])
        popt_sl, _ = curve_fit(linear_func, self.energy[mask_sl], y_data[mask_sl])
        
        x_int = (popt_sl[1] - popt_bg[1]) / (popt_bg[0] - popt_sl[0])
        y_int = linear_func(x_int, *popt_bg)
        
        return x_int, y_int, popt_bg, popt_sl

    def calculate_bandgap(self):
        if self.energy is None: return
        try:
            y_data = self.get_current_intensity()
            mode = self.bg_mode_var.get()
            pk_r = (float(self.bg_peak_min.get()), float(self.bg_peak_max.get()))
            mask_pk = (self.energy >= pk_r[0]) & (self.energy <= pk_r[1])
            if not np.any(mask_pk): raise ValueError("Peak range empty")
            x_pk = self.energy[mask_pk]; y_pk = y_data[mask_pk]
            p0 = [np.max(y_pk)-np.min(y_pk), x_pk[np.argmax(y_pk)], (np.max(x_pk)-np.min(x_pk))/4, np.min(y_pk)]
            try:
                popt_g, _ = curve_fit(gaussian_func, x_pk, y_pk, p0=p0, maxfev=2000)
                peak_x = popt_g[1]; peak_y = gaussian_func(peak_x, *popt_g)
            except:
                idx_m = np.argmax(y_pk); peak_x = x_pk[idx_m]; peak_y = y_pk[idx_m]; popt_g = None
            self.bg_context = {'peak_x': peak_x, 'peak_y': peak_y, 'popt_g': popt_g, 'mode': mode, 'pk_r': pk_r}
            self.plot_base_graph()
            
            if popt_g is not None:
                xf = np.linspace(pk_r[0], pk_r[1], 100)
                self.ax.plot(xf, gaussian_func(xf, *popt_g), 'lime', linestyle='--', linewidth=1.5, label='Gauss Fit')
            self.ax.plot(peak_x, peak_y, color=AppConfig.COLOR_MARKER_MAIN_PEAK, marker='*', markersize=15, zorder=5, label='Main Peak')
            self.ax.axvline(peak_x, color='green', linestyle=':', alpha=0.6)
            self.ax.axvspan(pk_r[0], pk_r[1], color='green', alpha=0.1)

            if mode == "Linear":
                bg_r = (float(self.bg_base_min.get()), float(self.bg_base_max.get()))
                sl_r = (float(self.bg_slope_min.get()), float(self.bg_slope_max.get()))
                
                onset_x, onset_y, popt_bg, popt_sl = self._fit_and_intersect(y_data, bg_r, sl_r)
                
                self.bg_context.update({'popt_bg': popt_bg, 'popt_sl': popt_sl, 'bg_r': bg_r, 'sl_r': sl_r})
                xp = np.linspace(min(self.energy), max(self.energy), 200)
                self.ax.plot(xp, linear_func(xp, *popt_bg), color=AppConfig.COLOR_FIT_BASE, linestyle='--', alpha=0.5, label='Base Fit')
                self.ax.plot(xp, linear_func(xp, *popt_sl), color=AppConfig.COLOR_FIT_SLOPE, linestyle='--', alpha=0.5, label='Slope Fit')
                self.ax.axvspan(bg_r[0], bg_r[1], color='blue', alpha=0.1)
                self.ax.axvspan(sl_r[0], sl_r[1], color='red', alpha=0.1)
                self.draw_bg_result(onset_x, onset_y, abs(onset_x - peak_x), "Linear Onset")

            elif mode in ["Hybrid", "Deriv"]:
                if mode == "Hybrid":
                    bg_r = (float(self.bg_base_min.get()), float(self.bg_base_max.get()))
                    sl_r = (float(self.bg_slope_min.get()), float(self.bg_slope_max.get()))
                    lin_onset_x, _, popt_bg, popt_sl = self._fit_and_intersect(y_data, bg_r, sl_r)
                    self.bg_context.update({'popt_bg': popt_bg, 'popt_sl': popt_sl, 'bg_r': bg_r, 'sl_r': sl_r})
                    xp = np.linspace(min(self.energy), max(self.energy), 200)
                    self.ax.plot(xp, linear_func(xp, *popt_bg), color=AppConfig.COLOR_FIT_BASE, linestyle='--', alpha=0.3, label='Base Fit')
                    self.ax.plot(xp, linear_func(xp, *popt_sl), color=AppConfig.COLOR_FIT_SLOPE, linestyle='--', alpha=0.3, label='Slope Fit')
                    self.ax.axvspan(bg_r[0], bg_r[1], color='blue', alpha=0.1)
                    self.ax.axvspan(sl_r[0], sl_r[1], color='red', alpha=0.1)
                    s_min, s_max = lin_onset_x - 1.5, lin_onset_x + 1.5
                else: # Deriv
                    s_min = float(self.bg_single_min.get())
                    s_max = float(self.bg_single_max.get())
                
                cands, xs, ys = self._find_candidates(s_min, s_max, y_data)
                self.bg_candidates = cands
                self.bg_context.update({'x_smooth': xs, 'y_smooth': ys})
                self.ax.axvspan(s_min, s_max, color='orange', alpha=0.1, label='Search Region')
                if xs is not None: self.ax.plot(xs, ys, color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Smoothed')
                self.update_bg_candidates_dropdown()

            self.ax.legend(loc='upper left'); self.canvas.draw()
        except Exception as e: messagebox.showerror("Calc Error", str(e))

    def update_bg_candidates_dropdown(self):
        peak_x = self.bg_context['peak_x']
        vals = ["Candidates (Curvature Order)"]
        for i, (cx, cy, _) in enumerate(self.bg_candidates):
            vals.append(f"{i+1}. Eg={abs(cx - peak_x):.3f} eV")
        self.combo_bg_candidates.configure(values=vals)
        self.combo_bg_candidates.set(vals[1] if len(vals)>1 else vals[0])
        if self.bg_candidates:
            bx, by, _ = self.bg_candidates[0]
            self.draw_bg_result(bx, by, abs(bx - peak_x), "Selected Onset")

    def on_bg_candidate_selected(self, choice):
        if choice.startswith("Candidates") or not self.bg_candidates: return
        try:
            idx = int(choice.split(".")[0]) - 1
            cx, cy, _ = self.bg_candidates[idx]
            self.plot_base_graph()
            
            if self.bg_context.get('popt_g') is not None:
                pk_r = self.bg_context['pk_r']
                xf = np.linspace(pk_r[0], pk_r[1], 100)
                self.ax.plot(xf, gaussian_func(xf, *self.bg_context['popt_g']), 'lime', linestyle='--', linewidth=1.5, label='Gauss Fit')
            
            self.ax.plot(self.bg_context['peak_x'], self.bg_context['peak_y'], 
                         color=AppConfig.COLOR_MARKER_MAIN_PEAK, marker='*', markersize=15, zorder=5, label='Main Peak')
            self.ax.axvline(self.bg_context['peak_x'], color='green', linestyle=':', alpha=0.6)
            self.ax.axvspan(self.bg_context['pk_r'][0], self.bg_context['pk_r'][1], color='green', alpha=0.1)

            if 'popt_bg' in self.bg_context:
                xp = np.linspace(min(self.energy), max(self.energy), 200)
                self.ax.plot(xp, linear_func(xp, *self.bg_context['popt_bg']), color=AppConfig.COLOR_FIT_BASE, linestyle='--', alpha=0.5, label='Base Fit')
                self.ax.plot(xp, linear_func(xp, *self.bg_context['popt_sl']), color=AppConfig.COLOR_FIT_SLOPE, linestyle='--', alpha=0.5, label='Slope Fit')
                self.ax.axvspan(self.bg_context['bg_r'][0], self.bg_context['bg_r'][1], color='blue', alpha=0.1)
                self.ax.axvspan(self.bg_context['sl_r'][0], self.bg_context['sl_r'][1], color='red', alpha=0.1)
            
            if 'x_smooth' in self.bg_context:
                xs = self.bg_context['x_smooth']
                self.ax.axvspan(xs[0], xs[-1], color='orange', alpha=0.1, label='Search Region')
                self.ax.plot(xs, self.bg_context['y_smooth'], color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Smoothed')
                
            self.draw_bg_result(cx, cy, abs(cx - self.bg_context['peak_x']), "Selected Onset")
            self.ax.legend(loc='upper left'); self.canvas.draw()
        except Exception as e: print(f"Error in selection: {e}")

    def draw_bg_result(self, x, y, gap, label):
        pk_x = self.bg_context['peak_x']; pk_y = self.bg_context['peak_y']
        self.ax.plot(x, y, color=AppConfig.COLOR_MARKER_BG, marker='x', markersize=10, markeredgewidth=3, zorder=6, label=label)
        self.ax.axvline(x, color='orange', linestyle='--', alpha=0.8)
        ay = (pk_y + y) / 2
        self.ax.annotate(f'Eg = {gap:.2f} eV', xy=(pk_x, ay), xytext=(x, ay),
                         arrowprops=dict(arrowstyle='<->', color='purple', lw=2),
                         ha='center', va='bottom', fontsize=12, color='purple', fontweight='bold')
        self.lbl_res_gap.configure(text=f"Eg: {gap:.3f} eV")

    def calculate_vbm(self):
        if self.energy is None: return
        try:
            y_data = self.get_current_intensity()
            mode = self.vbm_mode_var.get()
            self.vbm_context = {'mode': mode, 'y_data': y_data}
            self.plot_base_graph()
            
            if mode == "Linear":
                bg_r = (float(self.entry_vbm_base_min.get()), float(self.entry_vbm_base_max.get()))
                sl_r = (float(self.entry_vbm_slope_min.get()), float(self.entry_vbm_slope_max.get()))
                
                vbm_x, vbm_y, popt_bg, popt_sl = self._fit_and_intersect(y_data, bg_r, sl_r)
                
                self.vbm_context.update({'popt_bg': popt_bg, 'popt_sl': popt_sl, 'bg_r': bg_r, 'sl_r': sl_r})
                xp = np.linspace(min(self.energy), max(self.energy), 200)
                self.ax.plot(xp, linear_func(xp, *popt_bg), color=AppConfig.COLOR_FIT_BASE, linestyle='--', alpha=0.8, label='Base Fit')
                self.ax.plot(xp, linear_func(xp, *popt_sl), color=AppConfig.COLOR_FIT_SLOPE, linestyle='--', alpha=0.8, label='Slope Fit')
                self.ax.axvspan(bg_r[0], bg_r[1], color='blue', alpha=0.1)
                self.ax.axvspan(sl_r[0], sl_r[1], color='red', alpha=0.1)
                self.draw_vbm_result(vbm_x, vbm_y, "VBM (Linear)")

            elif mode in ["Hybrid", "Deriv"]:
                if mode == "Hybrid":
                    bg_r = (float(self.entry_vbm_base_min.get()), float(self.entry_vbm_base_max.get()))
                    sl_r = (float(self.entry_vbm_slope_min.get()), float(self.entry_vbm_slope_max.get()))
                    linear_vbm_x, _, popt_bg, popt_sl = self._fit_and_intersect(y_data, bg_r, sl_r)
                    self.vbm_context.update({'popt_bg': popt_bg, 'popt_sl': popt_sl, 'bg_r': bg_r, 'sl_r': sl_r})
                    xp = np.linspace(min(self.energy), max(self.energy), 200)
                    self.ax.plot(xp, linear_func(xp, *popt_bg), color=AppConfig.COLOR_FIT_BASE, linestyle='--', alpha=0.3, label='Base Fit')
                    self.ax.plot(xp, linear_func(xp, *popt_sl), color=AppConfig.COLOR_FIT_SLOPE, linestyle='--', alpha=0.3, label='Slope Fit')
                    self.ax.axvspan(bg_r[0], bg_r[1], color='blue', alpha=0.1)
                    self.ax.axvspan(sl_r[0], sl_r[1], color='red', alpha=0.1)
                    search_min, search_max = linear_vbm_x - 1.5, linear_vbm_x + 1.5
                else: # Deriv
                    search_min = float(self.entry_vbm_single_min.get())
                    search_max = float(self.entry_vbm_single_max.get())
                
                cands, xs, ys = self._find_candidates(search_min, search_max, y_data)
                self.vbm_candidates = cands
                self.vbm_context.update({'x_smooth': xs, 'y_smooth': ys})
                self.ax.axvspan(search_min, search_max, color='orange', alpha=0.1, label='Search Region')
                if xs is not None: self.ax.plot(xs, ys, color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Smoothed')
                self.update_vbm_candidates_dropdown()

            self.ax.legend(loc='upper left'); self.canvas.draw()
        except Exception as e: messagebox.showerror("Calc Error", str(e))

    def update_vbm_candidates_dropdown(self):
        vals = ["Candidates (Curvature Order)"]
        for i, (cx, cy, _) in enumerate(self.vbm_candidates): vals.append(f"{i+1}. VBM={cx:.3f} eV")
        self.combo_vbm_candidates.configure(values=vals)
        self.combo_vbm_candidates.set(vals[1] if len(vals)>1 else vals[0])
        if self.vbm_candidates:
            bx, by, _ = self.vbm_candidates[0]
            self.draw_vbm_result(bx, by, "Selected Onset")

    def on_vbm_candidate_selected(self, choice):
        if choice.startswith("Candidates") or not self.vbm_candidates: return
        try:
            idx = int(choice.split(".")[0]) - 1
            cx, cy, _ = self.vbm_candidates[idx]
            self.plot_base_graph()
            
            if 'popt_bg' in self.vbm_context:
                xp = np.linspace(min(self.energy), max(self.energy), 200)
                self.ax.plot(xp, linear_func(xp, *self.vbm_context['popt_bg']), color=AppConfig.COLOR_FIT_BASE, linestyle='--', alpha=0.8, label='Base Fit')
                self.ax.plot(xp, linear_func(xp, *self.vbm_context['popt_sl']), color=AppConfig.COLOR_FIT_SLOPE, linestyle='--', alpha=0.8, label='Slope Fit')
                self.ax.axvspan(self.vbm_context['bg_r'][0], self.vbm_context['bg_r'][1], color='blue', alpha=0.1)
                self.ax.axvspan(self.vbm_context['sl_r'][0], self.vbm_context['sl_r'][1], color='red', alpha=0.1)
            
            if 'x_smooth' in self.vbm_context:
                xs = self.vbm_context['x_smooth']
                self.ax.axvspan(xs[0], xs[-1], color='orange', alpha=0.1, label='Search Region')
                self.ax.plot(xs, self.vbm_context['y_smooth'], color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Smoothed')
            
            self.draw_vbm_result(cx, cy, "Selected Onset")
            self.ax.legend(loc='upper left'); self.canvas.draw()
        except Exception as e: print(f"Error in selection: {e}")

    def draw_vbm_result(self, x, y, label):
        self.ax.plot(x, y, color=AppConfig.COLOR_MARKER_VBM, marker='o', markersize=10, zorder=6, label=label)
        self.ax.axvline(x, color='green', linestyle=':', alpha=0.8)
        self.vbm_label.configure(text=f"VBM: {x:.3f} eV")

    def calculate_ups(self):
        if self.energy is None: return
        try:
            y_data = self.get_current_intensity()
            hv = float(self.entry_ups_hv.get())
            self.plot_base_graph()

            # A. Cutoff (High BE)
            c_bg = (float(self.entry_ups_cutoff_base_min.get()), float(self.entry_ups_cutoff_base_max.get()))
            c_sl = (float(self.entry_ups_cutoff_slope_min.get()), float(self.entry_ups_cutoff_slope_max.get()))
            e_cutoff, y_cutoff, popt_c_bg, popt_c_sl = self._fit_and_intersect(y_data, c_bg, c_sl)
            
            x_plot = np.linspace(min(self.energy), max(self.energy), 200)
            self.ax.plot(x_plot, linear_func(x_plot, *popt_c_bg), color=AppConfig.COLOR_FIT_BASE, linestyle='--', alpha=0.5)
            self.ax.plot(x_plot, linear_func(x_plot, *popt_c_sl), color=AppConfig.COLOR_FIT_SLOPE_UPS, linestyle='--', alpha=0.5)
            self.ax.axvspan(c_bg[0], c_bg[1], color='blue', alpha=0.1)
            self.ax.axvspan(c_sl[0], c_sl[1], color='cyan', alpha=0.1)
            self.draw_ups_result(e_cutoff, y_cutoff, "E_cutoff", "blue", align="right")

            # B. Fermi / HOMO (Low BE)
            f_bg = (float(self.entry_ups_fermi_base_min.get()), float(self.entry_ups_fermi_base_max.get()))
            f_sl = (float(self.entry_ups_fermi_slope_min.get()), float(self.entry_ups_fermi_slope_max.get()))
            e_onset, y_onset, popt_f_bg, popt_f_sl = self._fit_and_intersect(y_data, f_bg, f_sl)
            
            self.ax.plot(x_plot, linear_func(x_plot, *popt_f_bg), color=AppConfig.COLOR_FIT_BASE, linestyle='--', alpha=0.5)
            self.ax.plot(x_plot, linear_func(x_plot, *popt_f_sl), color=AppConfig.COLOR_FIT_SLOPE, linestyle='--', alpha=0.5)
            self.ax.axvspan(f_bg[0], f_bg[1], color='blue', alpha=0.1)
            self.ax.axvspan(f_sl[0], f_sl[1], color='red', alpha=0.1)
            self.draw_ups_result(e_onset, y_onset, "E_onset", "red", align="left")

            # C. Physics
            wf = hv - e_cutoff
            ip = hv - abs(e_cutoff - e_onset)
            self.lbl_res_wf.configure(text=f"WF (Φ): {wf:.3f} eV")
            self.lbl_res_ip.configure(text=f"IP: {ip:.3f} eV")
            self.ax.legend(loc='upper left'); self.canvas.draw()
        except Exception as e: messagebox.showerror("Calc Error", str(e))

    def draw_ups_result(self, x, y, label, color, align="right"):
        self.ax.plot(x, y, color=color, marker='x', markersize=10, markeredgewidth=2, zorder=6, label=label)
        offset_x = 1.0 if align == "right" else -1.0
        self.ax.annotate(f'{label}\n{x:.2f} eV', xy=(x, y), xytext=(x+offset_x, y), 
                         arrowprops=dict(arrowstyle='->', color=color), color=color, fontweight='bold')

    # --- 新規追加：LEIPS解析ロジック ---
    def calculate_leips(self):
        """VBMと同じく2直線の交点からLUMOを求める"""
        if self.energy is None: return
        try:
            y_data = self.get_current_intensity()
            self.plot_base_graph()

            l_bg = (float(self.entry_leips_base_min.get()), float(self.entry_leips_base_max.get()))
            l_sl = (float(self.entry_leips_slope_min.get()), float(self.entry_leips_slope_max.get()))
            
            lumo_x, lumo_y, popt_bg, popt_sl = self._fit_and_intersect(y_data, l_bg, l_sl)
            
            x_plot = np.linspace(min(self.energy), max(self.energy), 200)
            self.ax.plot(x_plot, linear_func(x_plot, *popt_bg), color=AppConfig.COLOR_FIT_BASE, linestyle='--', alpha=0.8, label='Base Fit')
            self.ax.plot(x_plot, linear_func(x_plot, *popt_sl), color=AppConfig.COLOR_FIT_SLOPE, linestyle='--', alpha=0.8, label='Slope Fit')
            self.ax.axvspan(l_bg[0], l_bg[1], color='blue', alpha=0.1)
            self.ax.axvspan(l_sl[0], l_sl[1], color='red', alpha=0.1)
            
            # 結果を描画（色はVBMと区別してシアンにしています）
            self.ax.plot(lumo_x, lumo_y, color='cyan', marker='o', markersize=10, zorder=6, label="LUMO (Intersection)")
            self.ax.axvline(lumo_x, color='cyan', linestyle=':', alpha=0.8)
            self.lbl_res_lumo.configure(text=f"LUMO: {lumo_x:.3f} eV")

            self.ax.legend(loc='upper left')
            self.canvas.draw()
        except Exception as e: messagebox.showerror("Calc Error", str(e))

    # --- 新規追加：LEET解析ロジック ---
    def calculate_leet(self):
        """ターゲット電流の立ち上がりから真空準位（Vacuum Level）を求める"""
        if self.energy is None: return
        try:
            y_data = self.get_current_intensity()
            self.plot_base_graph()

            v_bg = (float(self.entry_leet_base_min.get()), float(self.entry_leet_base_max.get()))
            v_sl = (float(self.entry_leet_slope_min.get()), float(self.entry_leet_slope_max.get()))
            
            vl_x, vl_y, popt_bg, popt_sl = self._fit_and_intersect(y_data, v_bg, v_sl)
            
            x_plot = np.linspace(min(self.energy), max(self.energy), 200)
            self.ax.plot(x_plot, linear_func(x_plot, *popt_bg), color=AppConfig.COLOR_FIT_BASE, linestyle='--', alpha=0.8, label='Base Fit')
            self.ax.plot(x_plot, linear_func(x_plot, *popt_sl), color=AppConfig.COLOR_FIT_SLOPE, linestyle='--', alpha=0.8, label='Slope Fit')
            self.ax.axvspan(v_bg[0], v_bg[1], color='blue', alpha=0.1)
            self.ax.axvspan(v_sl[0], v_sl[1], color='red', alpha=0.1)
            
            # 結果を描画（オレンジ色で目立たせます）
            self.ax.plot(vl_x, vl_y, color='orange', marker='D', markersize=10, zorder=6, label="Vacuum Level")
            self.ax.axvline(vl_x, color='orange', linestyle=':', alpha=0.8)
            self.lbl_res_vl.configure(text=f"Vacuum Level: {vl_x:.3f} eV")

            self.ax.legend(loc='upper left')
            self.canvas.draw()
        except Exception as e: messagebox.showerror("Calc Error", str(e))

    def on_closing(self):
        plt.close('all'); self.quit(); self.destroy()

if __name__ == "__main__":
    app = BaSALA_App()
    app.mainloop()