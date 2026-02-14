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
import os

# ==========================================
# 1. アプリケーション設定エリア
# ==========================================
ctk.set_appearance_mode("Dark")       # 外観モード: Dark, Light, System
ctk.set_default_color_theme("dark-blue") # テーマカラー

# ==========================================
# 2. 数学・物理計算用 関数群
# ==========================================

def linear_func(x, a, b):
    """線形近似用関数 (y = ax + b)"""
    return a * x + b

def gaussian_func(x, a, mu, sigma, c):
    """ガウス関数 (ピークフィッティング用)"""
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

def calculate_shirley_bg(x, y, tol=1e-5, max_iters=50):
    """
    Shirley法によるバックグラウンド（BG）計算関数
    """
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    n = len(y)
    I_start = y_sorted[0]
    I_end = y_sorted[-1]
    bg = np.full(n, I_start)
    for _ in range(max_iters):
        signal = y_sorted - I_start
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
# 3. GUI アプリケーションクラス
# ==========================================

class BaSALA_App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # ★ ウィンドウタイトル設定 (v2.14)
        self.title("BaSALA - Band Structure AnaLysis Assistant (v2.14)")
        self.geometry("1280x900")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # データ保持用変数
        self.file_path = None
        self.df = None
        self.energy = None
        self.intensity = None
        self.intensity_corrected = None
        self.bg_data = None
        self.span = None
        self.selection_mode = None
        
        # 候補選択用 (Hybrid & Derivative)
        self.candidates = []
        self.calc_context = {}

        self._create_sidebar()
        self._create_main_area()

    def _create_sidebar(self):
        """サイドバー（操作パネル）の作成"""
        # サイドバー幅 (280px)
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        
        # UI固定: 中身が増減しても幅が変わらないように固定
        self.sidebar.pack_propagate(False)

        # --- 共通操作エリア ---
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

        # --- 機能タブ ---
        self.tabview = ctk.CTkTabview(self.sidebar, width=260)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.tab_vbm = self.tabview.add("VBM")
        self.tab_bg = self.tabview.add("Band Gap")
        
        self._init_vbm_tab()
        self._init_bandgap_tab()

    def _init_vbm_tab(self):
        """Tab 1: VBM解析"""
        frame = ctk.CTkFrame(self.tab_vbm, fg_color="transparent")
        frame.pack(fill="both", expand=True)
        
        ctk.CTkLabel(frame, text="Determine VBM by Intersection", font=("Roboto", 12, "bold")).pack(pady=5)

        # 1. Background Range
        ctk.CTkLabel(frame, text="1. Background Range:", font=("Roboto", 11)).pack(anchor="w", padx=2)
        bg_frame = ctk.CTkFrame(frame, fg_color="transparent"); bg_frame.pack(fill="x", padx=2)
        self.entry_bg_min = ctk.CTkEntry(bg_frame, width=65); self.entry_bg_min.pack(side="left")
        ctk.CTkLabel(bg_frame, text="-").pack(side="left", padx=2)
        self.entry_bg_max = ctk.CTkEntry(bg_frame, width=65); self.entry_bg_max.pack(side="left")
        ctk.CTkButton(bg_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg")).pack(side="right")

        # 2. Slope Range
        ctk.CTkLabel(frame, text="2. VB Slope Range:", font=("Roboto", 11)).pack(anchor="w", padx=2, pady=(10,0))
        slope_frame = ctk.CTkFrame(frame, fg_color="transparent"); slope_frame.pack(fill="x", padx=2)
        self.entry_slope_min = ctk.CTkEntry(slope_frame, width=65); self.entry_slope_min.pack(side="left")
        ctk.CTkLabel(slope_frame, text="-").pack(side="left", padx=2)
        self.entry_slope_max = ctk.CTkEntry(slope_frame, width=65); self.entry_slope_max.pack(side="left")
        ctk.CTkButton(slope_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("slope")).pack(side="right")

        # Buttons
        self.btn_reset_mode = ctk.CTkButton(frame, text="Stop Selection", fg_color="transparent", border_width=1, height=24, command=self.deactivate_selector)
        self.btn_reset_mode.pack(pady=10)
        
        self.calc_btn = ctk.CTkButton(frame, text="Calculate VBM", command=self.calculate, fg_color="#2d8d2d", state="disabled")
        self.calc_btn.pack(padx=5, pady=15, fill="x")
        
        self.vbm_label = ctk.CTkLabel(frame, text="VBM: --- eV", font=ctk.CTkFont(size=18, weight="bold"), text_color="#4db6ac")
        self.vbm_label.pack(pady=5)

    def _init_bandgap_tab(self):
        """Tab 2: Band Gap解析"""
        self.bg_tab_frame = ctk.CTkFrame(self.tab_bg, fg_color="transparent")
        self.bg_tab_frame.pack(fill="both", expand=True)

        self.bg_mode_var = ctk.StringVar(value="Hybrid")
        self.seg_bg_mode = ctk.CTkSegmentedButton(self.bg_tab_frame, 
                                                  values=["Hybrid", "Linear", "Deriv"], 
                                                  variable=self.bg_mode_var, command=self.update_bg_ui)
        self.seg_bg_mode.pack(pady=10, padx=5, fill="x")
        self.seg_bg_mode.set("Hybrid")

        # 1. Main Peak
        ctk.CTkLabel(self.bg_tab_frame, text="1. Main Peak (Eg Reference):", font=("Roboto", 11)).pack(anchor="w", padx=2)
        self.p_frame = ctk.CTkFrame(self.bg_tab_frame, fg_color="transparent")
        self.p_frame.pack(fill="x", padx=2)
        self.bg_peak_min = ctk.CTkEntry(self.p_frame, width=65); self.bg_peak_min.pack(side="left")
        ctk.CTkLabel(self.p_frame, text="-").pack(side="left", padx=2)
        self.bg_peak_max = ctk.CTkEntry(self.p_frame, width=65); self.bg_peak_max.pack(side="left")
        ctk.CTkButton(self.p_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_peak")).pack(side="right")

        # Input Container
        self.bg_input_container = ctk.CTkFrame(self.bg_tab_frame, fg_color="transparent")
        self.bg_input_container.pack(fill="x", pady=5)

        # --- A. Linear / Hybrid UI Inputs ---
        self.frame_linear = ctk.CTkFrame(self.bg_input_container, fg_color="transparent")
        ctk.CTkLabel(self.frame_linear, text="2. Loss Base (Flat):", font=("Roboto", 11)).pack(anchor="w", padx=2, pady=(5,0))
        self.b_frame = ctk.CTkFrame(self.frame_linear, fg_color="transparent"); self.b_frame.pack(fill="x", padx=2)
        self.bg_base_min = ctk.CTkEntry(self.b_frame, width=65); self.bg_base_min.pack(side="left")
        ctk.CTkLabel(self.b_frame, text="-").pack(side="left", padx=2)
        self.bg_base_max = ctk.CTkEntry(self.b_frame, width=65); self.bg_base_max.pack(side="left")
        ctk.CTkButton(self.b_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_base")).pack(side="right")
        
        ctk.CTkLabel(self.frame_linear, text="3. Loss Slope (Rise):", font=("Roboto", 11)).pack(anchor="w", padx=2, pady=(5,0))
        self.s_frame = ctk.CTkFrame(self.frame_linear, fg_color="transparent"); self.s_frame.pack(fill="x", padx=2)
        self.bg_slope_min = ctk.CTkEntry(self.s_frame, width=65); self.bg_slope_min.pack(side="left")
        ctk.CTkLabel(self.s_frame, text="-").pack(side="left", padx=2)
        self.bg_slope_max = ctk.CTkEntry(self.s_frame, width=65); self.bg_slope_max.pack(side="left")
        ctk.CTkButton(self.s_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_slope")).pack(side="right")

        # --- B. Derivative UI Inputs ---
        self.frame_single = ctk.CTkFrame(self.bg_input_container, fg_color="transparent")
        ctk.CTkLabel(self.frame_single, text="2. Onset Search Region:", font=("Roboto", 11)).pack(anchor="w", padx=2, pady=(5,0))
        ctk.CTkLabel(self.frame_single, text="(Cover both Background & Rising edge)", font=("Roboto", 10), text_color="gray").pack(anchor="w", padx=2)
        self.d_frame = ctk.CTkFrame(self.frame_single, fg_color="transparent"); self.d_frame.pack(fill="x", padx=2)
        self.bg_single_min = ctk.CTkEntry(self.d_frame, width=65); self.bg_single_min.pack(side="left")
        ctk.CTkLabel(self.d_frame, text="-").pack(side="left", padx=2)
        self.bg_single_max = ctk.CTkEntry(self.d_frame, width=65); self.bg_single_max.pack(side="left")
        ctk.CTkButton(self.d_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_single")).pack(side="right")

        # 初期表示
        self.frame_linear.pack(fill="x", pady=5)

        # 共通ボタン
        ctk.CTkButton(self.bg_tab_frame, text="Stop Selection", fg_color="transparent", border_width=1, height=24, command=self.deactivate_selector).pack(pady=10)
        self.calc_bg_btn = ctk.CTkButton(self.bg_tab_frame, text="Calculate Band Gap", command=self.calculate_bandgap, fg_color="#E07A5F", state="disabled")
        self.calc_bg_btn.pack(pady=5, fill="x")
        
        # --- UIレイアウトの固定化 (ここが修正ポイント) ---
        # 1. 候補エリア用フレーム（常にここに存在し、中身だけ入れ替わる）
        self.frame_candidates_container = ctk.CTkFrame(self.bg_tab_frame, fg_color="transparent")
        self.frame_candidates_container.pack(pady=5, fill="x")
        
        # 2. 結果エリア用フレーム（常に候補の下に存在）
        self.frame_result_container = ctk.CTkFrame(self.bg_tab_frame, fg_color="transparent")
        self.frame_result_container.pack(pady=5, fill="x")

        # コンボボックス (containerの中に配置)
        self.combo_candidates = ctk.CTkComboBox(self.frame_candidates_container, width=240, 
                                                values=["Candidates (Curvature Order)"], 
                                                command=self.on_candidate_selected)
        self.combo_candidates.pack()
        self.combo_candidates.set("Candidates (Curvature Order)")
        
        # 結果ラベル (containerの中に配置)
        self.lbl_res_gap = ctk.CTkLabel(self.frame_result_container, text="Eg: --- eV", font=ctk.CTkFont(size=18, weight="bold"), text_color="#E07A5F")
        self.lbl_res_gap.pack()

    def update_bg_ui(self, value):
        """モード切替時のUI更新"""
        # 入力項目の切り替え
        if value in ["Linear", "Hybrid"]:
            self.frame_single.pack_forget()
            self.frame_linear.pack(fill="x", pady=5)
        else:
            self.frame_linear.pack_forget()
            self.frame_single.pack(fill="x", pady=5)
            
        # 候補ドロップダウンの表示/非表示 (コンテナは動かさず、中身だけ制御)
        if value == "Linear":
            self.combo_candidates.pack_forget() # 隠す
        else:
            self.combo_candidates.pack() # 表示する (コンテナ内での位置は変わらない)

    def _create_main_area(self):
        """メイングラフエリア"""
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

    # ==========================================
    # 4. 操作ロジック
    # ==========================================

    def auto_scale_y(self):
        if self.intensity is None: return
        y_max_raw = np.max(self.intensity)
        y_current = self.get_current_intensity()
        y_min_curr = np.min(y_current)
        amp = y_max_raw - y_min_curr
        margin = amp * 0.05 if amp > 0 else 10.0
        self.ax.set_ylim(y_min_curr - margin, y_max_raw + margin)

    def activate_selector(self, mode):
        self.selection_mode = mode
        if self.span: self.span.set_visible(False); self.span = None
        
        colors = {
            'bg': 'blue', 'slope': 'red',             
            'bg_peak': 'green',                       
            'bg_base': 'blue', 'bg_slope': 'red',     
            'bg_single': 'orange'                     
        }
        color = colors.get(mode, 'gray')
        
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
        
        entries = {
            'bg': (self.entry_bg_min, self.entry_bg_max), 
            'slope': (self.entry_slope_min, self.entry_slope_max),
            'bg_peak': (self.bg_peak_min, self.bg_peak_max),
            'bg_base': (self.bg_base_min, self.bg_base_max), 
            'bg_slope': (self.bg_slope_min, self.bg_slope_max),
            'bg_single': (self.bg_single_min, self.bg_single_max)
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
        
        sep = {", (Comma)": ",", "\\t (Tab)": "\t", "Space": r"\s+"}[self.sep_option.get()]
        
        try:
            self.df = pd.read_csv(file_path, sep=sep, header=None, engine='python')
            if self.df.shape[1] < 2: return
            
            self.energy = pd.to_numeric(self.df.iloc[:, 0], errors='coerce').values
            self.intensity = pd.to_numeric(self.df.iloc[:, 1], errors='coerce').values
            mask = ~np.isnan(self.energy) & ~np.isnan(self.intensity)
            self.energy = self.energy[mask]; self.intensity = self.intensity[mask]
            if len(self.energy) == 0: return

            min_e, max_e = np.min(self.energy), np.max(self.energy)
            
            # 初期範囲設定
            # VBM Tab
            self.entry_bg_min.delete(0, tk.END); self.entry_bg_min.insert(0, f"{min_e:.1f}")
            self.entry_bg_max.delete(0, tk.END); self.entry_bg_max.insert(0, f"{min_e+1.0:.1f}") 
            self.entry_slope_min.delete(0, tk.END); self.entry_slope_min.insert(0, f"{min_e+1.5:.1f}")
            self.entry_slope_max.delete(0, tk.END); self.entry_slope_max.insert(0, f"{min_e+2.5:.1f}") 
            
            # Band Gap Tab
            self.bg_peak_min.delete(0, tk.END); self.bg_peak_min.insert(0, f"{min_e:.1f}")
            self.bg_peak_max.delete(0, tk.END); self.bg_peak_max.insert(0, f"{min_e+0.8:.1f}") 
            
            # Linear / Hybrid Fitting Ranges
            self.bg_base_min.delete(0, tk.END); self.bg_base_min.insert(0, f"{min_e+4.0:.1f}")
            self.bg_base_max.delete(0, tk.END); self.bg_base_max.insert(0, f"{min_e+5.0:.1f}") 
            self.bg_slope_min.delete(0, tk.END); self.bg_slope_min.insert(0, f"{min_e+5.5:.1f}")
            self.bg_slope_max.delete(0, tk.END); self.bg_slope_max.insert(0, f"{min_e+6.5:.1f}") 
            
            # Derivative Search Range
            self.bg_single_min.delete(0, tk.END); self.bg_single_min.insert(0, f"{min_e+4.0:.1f}")
            self.bg_single_max.delete(0, tk.END); self.bg_single_max.insert(0, f"{min_e+7.0:.1f}")
            
            self.chk_shirley_var.set(False); self.intensity_corrected = None; self.bg_data = None
            
            self.plot_base_graph()
            self.calc_btn.configure(state="normal"); self.calc_bg_btn.configure(state="normal")
            
        except Exception as e: messagebox.showerror("Error", str(e))

    def plot_base_graph(self):
        self.ax.clear()
        if self.chk_shirley_var.get() and self.intensity_corrected is not None:
            self.ax.plot(self.energy, self.intensity, color='gray', alpha=0.3, label='Raw Data')
            self.ax.plot(self.energy, self.bg_data, color='gray', linestyle='--', alpha=0.5, label='Shirley BG')
            self.ax.plot(self.energy, self.intensity_corrected, color='#4a90e2', linewidth=1.5, label='Corrected')
        else:
            self.ax.plot(self.energy, self.intensity, color='#4a90e2', linewidth=1.5, label='Raw Spectrum')
        self.ax.legend(); self.ax.grid(True); self.ax.invert_xaxis()
        self.auto_scale_y()
        self.canvas.draw()

    def get_current_intensity(self):
        return self.intensity_corrected if (self.chk_shirley_var.get() and self.intensity_corrected is not None) else self.intensity

    # ==========================================
    # 5. 計算ロジック
    # ==========================================

    def calculate(self):
        """【Tab 1】VBM解析"""
        if self.energy is None: return
        try:
            y_data = self.get_current_intensity()
            
            bg_r = (float(self.entry_bg_min.get()), float(self.entry_bg_max.get()))
            sl_r = (float(self.entry_slope_min.get()), float(self.entry_slope_max.get()))

            mask_bg = (self.energy >= bg_r[0]) & (self.energy <= bg_r[1])
            popt_bg, _ = curve_fit(linear_func, self.energy[mask_bg], y_data[mask_bg])
            
            mask_sl = (self.energy >= sl_r[0]) & (self.energy <= sl_r[1])
            popt_sl, _ = curve_fit(linear_func, self.energy[mask_sl], y_data[mask_sl])

            vbm_x = (popt_sl[1] - popt_bg[1]) / (popt_bg[0] - popt_sl[0])
            vbm_y = linear_func(vbm_x, *popt_bg)
            self.vbm_label.configure(text=f"VBM: {vbm_x:.3f} eV")

            self.plot_base_graph()
            x_plot = np.linspace(min(self.energy), max(self.energy), 200)
            self.ax.plot(x_plot, linear_func(x_plot, *popt_bg), 'b--', alpha=0.8, label='Base Fit')
            self.ax.plot(x_plot, linear_func(x_plot, *popt_sl), 'r--', alpha=0.8, label='Slope Fit')
            self.ax.plot(vbm_x, vbm_y, 'go', markersize=10, zorder=5, label=f'VBM={vbm_x:.2f}eV')
            self.ax.axvline(vbm_x, color='green', linestyle=':', alpha=0.8)
            
            self.ax.axvspan(bg_r[0], bg_r[1], color='blue', alpha=0.1)
            self.ax.axvspan(sl_r[0], sl_r[1], color='red', alpha=0.1)
            
            self.ax.legend(); self.canvas.draw()
        except Exception as e: messagebox.showerror("Calc Error", str(e))

    def calculate_bandgap(self):
        """【Tab 2】Band Gap解析"""
        if self.energy is None: return
        try:
            y_data = self.get_current_intensity()
            mode = self.bg_mode_var.get()
            
            pk_r = (float(self.bg_peak_min.get()), float(self.bg_peak_max.get()))
            mask_pk = (self.energy >= pk_r[0]) & (self.energy <= pk_r[1])
            if not np.any(mask_pk): raise ValueError("Peak範囲にデータがありません")
            
            x_pk_fit = self.energy[mask_pk]; y_pk_fit = y_data[mask_pk]
            p0 = [np.max(y_pk_fit)-np.min(y_pk_fit), x_pk_fit[np.argmax(y_pk_fit)], (np.max(x_pk_fit)-np.min(x_pk_fit))/4, np.min(y_pk_fit)]
            
            try:
                popt_gauss, _ = curve_fit(gaussian_func, x_pk_fit, y_pk_fit, p0=p0, maxfev=2000)
                peak_x = popt_gauss[1]; peak_y = gaussian_func(peak_x, *popt_gauss)
            except:
                idx_max = np.argmax(y_pk_fit); peak_x = x_pk_fit[idx_max]; peak_y = y_pk_fit[idx_max]; popt_gauss = None

            # 描画
            self.plot_base_graph()
            if popt_gauss is not None:
                x_fine = np.linspace(pk_r[0], pk_r[1], 100)
                self.ax.plot(x_fine, gaussian_func(x_fine, *popt_gauss), 'lime', linestyle='--', linewidth=1.5, label='Gaussian Fit')
            self.ax.plot(peak_x, peak_y, 'g*', markersize=15, zorder=5, label='Main Peak')
            self.ax.axvline(peak_x, color='green', linestyle=':', alpha=0.6)
            self.ax.axvspan(pk_r[0], pk_r[1], color='green', alpha=0.1)

            self.calc_context = {
                'peak_x': peak_x, 'peak_y': peak_y,
                'popt_gauss': popt_gauss, 'y_data': y_data, 'mode': mode
            }

            onset_x, onset_y, gap = 0, 0, 0

            if mode == "Linear":
                base_r = (float(self.bg_base_min.get()), float(self.bg_base_max.get()))
                sl_r = (float(self.bg_slope_min.get()), float(self.bg_slope_max.get()))
                mask_base = (self.energy >= base_r[0]) & (self.energy <= base_r[1])
                mask_sl = (self.energy >= sl_r[0]) & (self.energy <= sl_r[1])
                popt_base, _ = curve_fit(linear_func, self.energy[mask_base], y_data[mask_base])
                popt_sl, _ = curve_fit(linear_func, self.energy[mask_sl], y_data[mask_sl])
                onset_x = (popt_sl[1] - popt_base[1]) / (popt_base[0] - popt_sl[0])
                onset_y = linear_func(onset_x, *popt_base)
                gap = abs(onset_x - peak_x)
                
                self.calc_context.update({'popt_base': popt_base, 'popt_sl': popt_sl, 'base_r': base_r, 'sl_r': sl_r})
                
                x_plot = np.linspace(min(self.energy), max(self.energy), 200)
                self.ax.plot(x_plot, linear_func(x_plot, *popt_base), 'b--', alpha=0.5, label='Base Fit')
                self.ax.plot(x_plot, linear_func(x_plot, *popt_sl), 'r--', alpha=0.5, label='Slope Fit')
                self.ax.axvspan(base_r[0], base_r[1], color='blue', alpha=0.1)
                self.ax.axvspan(sl_r[0], sl_r[1], color='red', alpha=0.1)
                self.draw_result_marker(onset_x, onset_y, gap, "Linear Onset")

            elif mode == "Hybrid":
                base_r = (float(self.bg_base_min.get()), float(self.bg_base_max.get()))
                sl_r = (float(self.bg_slope_min.get()), float(self.bg_slope_max.get()))
                mask_base = (self.energy >= base_r[0]) & (self.energy <= base_r[1])
                mask_sl = (self.energy >= sl_r[0]) & (self.energy <= sl_r[1])
                popt_base, _ = curve_fit(linear_func, self.energy[mask_base], y_data[mask_base])
                popt_sl, _ = curve_fit(linear_func, self.energy[mask_sl], y_data[mask_sl])
                linear_onset_x = (popt_sl[1] - popt_base[1]) / (popt_base[0] - popt_sl[0])
                
                x_plot = np.linspace(min(self.energy), max(self.energy), 200)
                self.ax.plot(x_plot, linear_func(x_plot, *popt_base), 'b--', alpha=0.3)
                self.ax.plot(x_plot, linear_func(x_plot, *popt_sl), 'r--', alpha=0.3)
                self.ax.axvspan(base_r[0], base_r[1], color='blue', alpha=0.1)
                self.ax.axvspan(sl_r[0], sl_r[1], color='red', alpha=0.1)
                
                self.calc_context.update({'popt_base': popt_base, 'popt_sl': popt_sl, 'base_r': base_r, 'sl_r': sl_r})
                search_min, search_max = linear_onset_x - 1.5, linear_onset_x + 1.5
                self.find_and_display_candidates(search_min, search_max, y_data, peak_x)
                self.ax.axvspan(search_min, search_max, color='orange', alpha=0.1, label='Search Region')

            else:
                d_r = (float(self.bg_single_min.get()), float(self.bg_single_max.get()))
                self.find_and_display_candidates(d_r[0], d_r[1], y_data, peak_x)
                if 'x_smooth' in self.calc_context:
                    self.ax.plot(self.calc_context['x_smooth'], self.calc_context['y_smooth'], color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Smoothed')
                self.ax.axvspan(d_r[0], d_r[1], color='orange', alpha=0.1, label='Search Region')

            self.ax.legend(); self.canvas.draw()

        except Exception as e: messagebox.showerror("Calc Error", str(e))

    def find_and_display_candidates(self, search_min, search_max, y_data, peak_x):
        """候補探索 & ドロップダウン更新 (曲率順・スコア非表示)"""
        target_window = 21 
        w_len = min(target_window, len(y_data))
        if w_len % 2 == 0: w_len -= 1
        if w_len < 3: w_len = 3
        y_smooth_all = savgol_filter(y_data, window_length=w_len, polyorder=2)
        
        mask_search = (self.energy >= search_min) & (self.energy <= search_max)
        x_s = self.energy[mask_search]
        y_s_smooth = y_smooth_all[mask_search]
        
        self.calc_context.update({'x_smooth': x_s, 'y_smooth': y_s_smooth})
        
        if len(x_s) < 5: raise ValueError("探索範囲データ不足")
        
        d2y = np.gradient(np.gradient(y_s_smooth, x_s), x_s)
        peaks, properties = find_peaks(d2y, height=0)
        
        self.candidates = []
        if len(peaks) > 0:
            for p_idx in peaks:
                cx = x_s[p_idx]
                cy = y_s_smooth[p_idx]
                raw_score = properties['peak_heights'][list(peaks).index(p_idx)]
                self.candidates.append((cx, cy, raw_score))
            
            # 1. 曲率（生スコア）が大きい順にソート
            self.candidates.sort(key=lambda x: x[2], reverse=True)
            self.candidates = self.candidates[:5]
            
        else:
            mid_idx = len(x_s) // 2
            self.candidates = [(x_s[mid_idx], y_s_smooth[mid_idx], 0)]

        # --- 表示: ヘッダー + 候補 ---
        combo_values = ["Candidates (Curvature Order)"] # ヘッダー
        for i, (cx, cy, r_score) in enumerate(self.candidates):
            gap_val = abs(cx - peak_x)
            combo_values.append(f"{i+1}. Eg={gap_val:.3f} eV")
        
        self.combo_candidates.configure(values=combo_values)
        self.combo_candidates.set(combo_values[1]) # 1位を選択状態にする
        
        best_x, best_y, _ = self.candidates[0]
        self.draw_result_marker(best_x, best_y, abs(best_x - peak_x), "Selected Candidate")

    def on_candidate_selected(self, choice):
        """候補選択時の再描画"""
        if choice.startswith("Candidates"): return # ヘッダー無視
        
        if not self.candidates or not self.calc_context: return
        try:
            idx = int(choice.split(".")[0]) - 1
            cx, cy, _ = self.candidates[idx]
            
            self.plot_base_graph()
            
            peak_x = self.calc_context['peak_x']
            peak_y = self.calc_context['peak_y']
            popt_gauss = self.calc_context['popt_gauss']
            mode = self.calc_context['mode']
            pk_min = float(self.bg_peak_min.get()); pk_max = float(self.bg_peak_max.get())

            if popt_gauss is not None:
                x_fine = np.linspace(pk_min, pk_max, 100)
                self.ax.plot(x_fine, gaussian_func(x_fine, *popt_gauss), 'lime', linestyle='--', linewidth=1.5)
            self.ax.plot(peak_x, peak_y, 'g*', markersize=15, zorder=5)
            self.ax.axvline(peak_x, color='green', linestyle=':', alpha=0.6)
            self.ax.axvspan(pk_min, pk_max, color='green', alpha=0.1)

            if mode == "Hybrid":
                popt_base = self.calc_context['popt_base']
                popt_sl = self.calc_context['popt_sl']
                x_plot = np.linspace(min(self.energy), max(self.energy), 200)
                self.ax.plot(x_plot, linear_func(x_plot, *popt_base), 'b--', alpha=0.3)
                self.ax.plot(x_plot, linear_func(x_plot, *popt_sl), 'r--', alpha=0.3)
                self.ax.axvspan(self.calc_context['base_r'][0], self.calc_context['base_r'][1], color='blue', alpha=0.1)
                self.ax.axvspan(self.calc_context['sl_r'][0], self.calc_context['sl_r'][1], color='red', alpha=0.1)
                
            elif mode == "Deriv":
                if 'x_smooth' in self.calc_context:
                    self.ax.plot(self.calc_context['x_smooth'], self.calc_context['y_smooth'], color='orange', linestyle=':', linewidth=2, alpha=0.8)
            
            self.draw_result_marker(cx, cy, abs(cx - peak_x), "Selected Candidate")
            self.ax.legend(); self.canvas.draw()
            
        except Exception as e: print(f"Selection Error: {e}")

    def draw_result_marker(self, x, y, gap, label):
        """結果マーカー描画"""
        peak_x = self.calc_context['peak_x']
        peak_y = self.calc_context['peak_y']
        
        self.ax.plot(x, y, 'bx', markersize=10, markeredgewidth=3, zorder=6, label=label)
        self.ax.axvline(x, color='orange', linestyle='--', alpha=0.8)
        
        arrow_y = (peak_y + y) / 2
        self.ax.annotate(f'Eg = {gap:.2f} eV', xy=(peak_x, arrow_y), xytext=(x, arrow_y),
                         arrowprops=dict(arrowstyle='<->', color='purple', lw=2),
                         ha='center', va='bottom', fontsize=12, color='purple', fontweight='bold')
        self.lbl_res_gap.configure(text=f"Eg: {gap:.3f} eV")

    def on_closing(self):
        """アプリ終了時処理"""
        plt.close('all'); self.quit(); self.destroy()

if __name__ == "__main__":
    app = BaSALA_App()
    app.mainloop()