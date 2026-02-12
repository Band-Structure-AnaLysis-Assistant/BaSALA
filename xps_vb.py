import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
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
    """
    線形近似用関数 (y = ax + b)
    Curve fitting (scipy.optimize.curve_fit) で使用します。
    """
    return a * x + b

def calculate_shirley_bg(x, y, tol=1e-5, max_iters=50):
    """
    Shirley法によるバックグラウンド（BG）計算関数
    
    XPSスペクトル特有の「非弾性散乱によるベースラインの上昇」を除去します。
    高エネルギー側（左）のBG高さは、低エネルギー側（右）のピーク面積に比例するという
    原理に基づき、反復計算を行ってBG形状を決定します。
    
    Args:
        x: Binding Energy (eV)
        y: Intensity (Counts)
        tol: 収束判定の許容誤差
        max_iters: 最大反復回数
    Returns:
        np.array: 計算されたバックグラウンド強度配列
    """
    # 1. データのソート
    # 積分計算は「低エネルギー(右) → 高エネルギー(左)」の順で行うため、
    # 数値を昇順(小さい順)に並べ替えて処理します。
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    n = len(y)
    
    # 2. 基準点の決定
    # I_start: 低エネルギー側（右端）の強度。BGの始点。
    # I_end:   高エネルギー側（左端）の強度。BGの終点。
    I_start = y_sorted[0]
    I_end = y_sorted[-1]
    
    # 3. 初期BGの作成 (最初は平らな線からスタート)
    bg = np.full(n, I_start)
    
    # 4. 反復計算 (Iterative Calculation)
    for _ in range(max_iters):
        # 信号成分 = 現在の強度 - BG (マイナス値は0にクリップ)
        signal = y_sorted - I_start
        signal[signal < 0] = 0
        
        # 累積積分 (右端から左端へ面積を積み上げる)
        cum_area = np.zeros(n)
        # 台形積分近似
        cum_area[1:] = np.cumsum((signal[:-1] + signal[1:]) / 2 * np.diff(x_sorted))
        
        total_area = cum_area[-1] # 全面積
        
        # 新しいBGの形状を計算
        if total_area == 0:
            k = 0
        else:
            k = (I_end - I_start) / total_area
            
        bg_new = I_start + k * cum_area
        
        # 収束判定
        if np.max(np.abs(bg_new - bg)) < tol:
            bg = bg_new
            break
            
        bg = bg_new

    # 5. 並び順を元に戻す
    bg_original_order = np.zeros(n)
    bg_original_order[sorted_indices] = bg
    
    return bg_original_order

# ==========================================
# 3. GUI アプリケーションクラス
# ==========================================

class XPS_VB_Edge_App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- ウィンドウ設定 ---
        self.title("XPS Analysis Suite - v1.0 (Strong Smoothing)")
        self.geometry("1280x900")
        
        # 終了処理の登録
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- メンバ変数（データ保持用）---
        self.file_path = None
        self.df = None
        self.energy = None              # x軸データ
        self.intensity = None           # y軸データ(生)
        self.intensity_corrected = None # Shirley補正後のy軸データ
        self.bg_data = None             # Shirley BGデータ
        
        # --- メンバ変数（ツール用）---
        self.span = None        # マウス選択範囲
        self.selection_mode = None # 現在の入力欄識別

        # --- GUI構築 ---
        self._create_sidebar()
        self._create_main_area()

    def _create_sidebar(self):
        """左側の操作パネルを作成"""
        self.sidebar = ctk.CTkFrame(self, width=340, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)

        # ロゴ
        self.logo_label = ctk.CTkLabel(self.sidebar, text="XPS Analysis\nv1.0", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.pack(padx=20, pady=(20, 10))

        # --- 共通操作エリア ---
        self.common_frame = ctk.CTkFrame(self.sidebar)
        self.common_frame.pack(padx=10, pady=5, fill="x")
        
        self.load_btn = ctk.CTkButton(self.common_frame, text="Open CSV / Text", command=self.load_csv, fg_color="#1f538d")
        self.load_btn.pack(padx=5, pady=5, fill="x")
        
        self.sep_option = ctk.CTkComboBox(self.common_frame, values=[", (Comma)", "\\t (Tab)", "Space"], height=24)
        self.sep_option.set(", (Comma)")
        self.sep_option.pack(padx=5, pady=(0, 5))

        self.chk_shirley_var = ctk.BooleanVar(value=False)
        self.chk_shirley = ctk.CTkCheckBox(self.common_frame, text="Apply Shirley BG", variable=self.chk_shirley_var, command=self.on_shirley_toggle)
        self.chk_shirley.pack(padx=10, pady=10, anchor="w")

        # --- 機能切り替えタブ ---
        self.tabview = ctk.CTkTabview(self.sidebar, width=320)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.tab_analysis = self.tabview.add("Analysis")       # VBM解析
        self.tab_bg = self.tabview.add("Band Gap")             # バンドギャップ解析
        self.tab_graph = self.tabview.add("Graph Settings")    # 見た目設定

        self._init_vbm_tab()
        self._init_bandgap_tab()
        self._init_graph_tab()

    def _init_vbm_tab(self):
        """Tab 1: VBM解析 (直線交点法) UI"""
        frame = ctk.CTkFrame(self.tab_analysis, fg_color="transparent")
        frame.pack(fill="both", expand=True)
        ctk.CTkLabel(frame, text="Determine VBM by Intersection", font=("Roboto", 12, "bold")).pack(pady=5)

        # 1. Background
        ctk.CTkLabel(frame, text="1. Background Range:", font=("Roboto", 11)).pack(anchor="w", padx=5)
        bg_frame = ctk.CTkFrame(frame, fg_color="transparent")
        bg_frame.pack(fill="x", padx=5)
        self.entry_bg_min = ctk.CTkEntry(bg_frame, width=50); self.entry_bg_min.pack(side="left")
        ctk.CTkLabel(bg_frame, text="-").pack(side="left")
        self.entry_bg_max = ctk.CTkEntry(bg_frame, width=50); self.entry_bg_max.pack(side="left")
        ctk.CTkButton(bg_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg")).pack(side="right")

        # 2. Slope
        ctk.CTkLabel(frame, text="2. VB Slope Range:", font=("Roboto", 11)).pack(anchor="w", padx=5, pady=(10,0))
        slope_frame = ctk.CTkFrame(frame, fg_color="transparent")
        slope_frame.pack(fill="x", padx=5)
        self.entry_slope_min = ctk.CTkEntry(slope_frame, width=50); self.entry_slope_min.pack(side="left")
        ctk.CTkLabel(slope_frame, text="-").pack(side="left")
        self.entry_slope_max = ctk.CTkEntry(slope_frame, width=50); self.entry_slope_max.pack(side="left")
        ctk.CTkButton(slope_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("slope")).pack(side="right")

        # Buttons
        self.btn_reset_mode = ctk.CTkButton(frame, text="Stop Selection", fg_color="transparent", border_width=1, height=24, command=self.deactivate_selector)
        self.btn_reset_mode.pack(pady=10)
        self.calc_btn = ctk.CTkButton(frame, text="Calculate VBM", command=self.calculate, fg_color="#2d8d2d", state="disabled")
        self.calc_btn.pack(padx=10, pady=15, fill="x")
        self.vbm_label = ctk.CTkLabel(frame, text="VBM: --- eV", font=ctk.CTkFont(size=20, weight="bold"), text_color="#4db6ac")
        self.vbm_label.pack(pady=5)

    def _init_bandgap_tab(self):
        """
        Tab 2: バンドギャップ解析 UI
        ※ 可変入力欄はコンテナを使用してレイアウト崩れを防止
        """
        self.bg_tab_frame = ctk.CTkFrame(self.tab_bg, fg_color="transparent")
        self.bg_tab_frame.pack(fill="both", expand=True)

        # モード切替スイッチ
        self.bg_mode_var = ctk.StringVar(value="Linear Fit")
        self.seg_bg_mode = ctk.CTkSegmentedButton(self.bg_tab_frame, values=["Linear Fit", "Derivative"], 
                                                  variable=self.bg_mode_var, command=self.update_bg_ui)
        self.seg_bg_mode.pack(pady=10, padx=10, fill="x")
        self.seg_bg_mode.set("Linear Fit")

        # 1. Main Peak (共通)
        ctk.CTkLabel(self.bg_tab_frame, text="1. Main Peak (Eg Reference):", font=("Roboto", 11)).pack(anchor="w", padx=5)
        self.p_frame = ctk.CTkFrame(self.bg_tab_frame, fg_color="transparent")
        self.p_frame.pack(fill="x", padx=5)
        self.bg_peak_min = ctk.CTkEntry(self.p_frame, width=50); self.bg_peak_min.pack(side="left")
        ctk.CTkLabel(self.p_frame, text="-").pack(side="left")
        self.bg_peak_max = ctk.CTkEntry(self.p_frame, width=50); self.bg_peak_max.pack(side="left")
        ctk.CTkButton(self.p_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_peak")).pack(side="right")

        # ★★★ 入力欄コンテナ ★★★
        self.bg_input_container = ctk.CTkFrame(self.bg_tab_frame, fg_color="transparent")
        self.bg_input_container.pack(fill="x", pady=5)

        # --- A. Linear Mode UI ---
        self.frame_linear = ctk.CTkFrame(self.bg_input_container, fg_color="transparent")
        
        ctk.CTkLabel(self.frame_linear, text="2. Loss Base (Flat):", font=("Roboto", 11)).pack(anchor="w", padx=5, pady=(5,0))
        self.b_frame = ctk.CTkFrame(self.frame_linear, fg_color="transparent")
        self.b_frame.pack(fill="x", padx=5)
        self.bg_base_min = ctk.CTkEntry(self.b_frame, width=50); self.bg_base_min.pack(side="left")
        ctk.CTkLabel(self.b_frame, text="-").pack(side="left")
        self.bg_base_max = ctk.CTkEntry(self.b_frame, width=50); self.bg_base_max.pack(side="left")
        ctk.CTkButton(self.b_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_base")).pack(side="right")

        ctk.CTkLabel(self.frame_linear, text="3. Loss Slope (Rise):", font=("Roboto", 11)).pack(anchor="w", padx=5, pady=(5,0))
        self.s_frame = ctk.CTkFrame(self.frame_linear, fg_color="transparent")
        self.s_frame.pack(fill="x", padx=5)
        self.bg_slope_min = ctk.CTkEntry(self.s_frame, width=50); self.bg_slope_min.pack(side="left")
        ctk.CTkLabel(self.s_frame, text="-").pack(side="left")
        self.bg_slope_max = ctk.CTkEntry(self.s_frame, width=50); self.bg_slope_max.pack(side="left")
        ctk.CTkButton(self.s_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_slope")).pack(side="right")

        # --- B. Derivative Mode UI ---
        self.frame_deriv = ctk.CTkFrame(self.bg_input_container, fg_color="transparent")
        
        ctk.CTkLabel(self.frame_deriv, text="2. Onset Search Region:", font=("Roboto", 11)).pack(anchor="w", padx=5, pady=(5,0))
        ctk.CTkLabel(self.frame_deriv, text="(Select wide range: Base ~ Slope)", font=("Roboto", 10), text_color="gray").pack(anchor="w", padx=5)
        
        self.d_frame = ctk.CTkFrame(self.frame_deriv, fg_color="transparent")
        self.d_frame.pack(fill="x", padx=5)
        self.bg_deriv_min = ctk.CTkEntry(self.d_frame, width=50); self.bg_deriv_min.pack(side="left")
        ctk.CTkLabel(self.d_frame, text="-").pack(side="left")
        self.bg_deriv_max = ctk.CTkEntry(self.d_frame, width=50); self.bg_deriv_max.pack(side="left")
        ctk.CTkButton(self.d_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_deriv")).pack(side="right")

        # 初期表示設定
        self.frame_linear.pack(fill="x", pady=5)

        # 共通ボタン
        ctk.CTkButton(self.bg_tab_frame, text="Stop Selection", fg_color="transparent", border_width=1, height=24, command=self.deactivate_selector).pack(pady=10)
        self.calc_bg_btn = ctk.CTkButton(self.bg_tab_frame, text="Calculate Band Gap", command=self.calculate_bandgap, fg_color="#E07A5F", state="disabled")
        self.calc_bg_btn.pack(pady=5, fill="x")
        self.lbl_res_gap = ctk.CTkLabel(self.bg_tab_frame, text="Eg: --- eV", font=ctk.CTkFont(size=18, weight="bold"), text_color="#E07A5F")
        self.lbl_res_gap.pack(pady=5)

    def update_bg_ui(self, value):
        """モード切替処理"""
        if value == "Linear Fit":
            self.frame_deriv.pack_forget()
            self.frame_linear.pack(fill="x", pady=5)
        else:
            self.frame_linear.pack_forget()
            self.frame_deriv.pack(fill="x", pady=5)

    def _init_graph_tab(self):
        """Tab 3: グラフ設定 UI"""
        frame = ctk.CTkFrame(self.tab_graph, fg_color="transparent")
        frame.pack(fill="both", expand=True)
        
        # Labels
        ctk.CTkLabel(frame, text="Labels & Title", font=("Roboto", 12, "bold")).pack(anchor="w", pady=2)
        self.entry_title = ctk.CTkEntry(frame, placeholder_text="Graph Title"); self.entry_title.pack(fill="x", pady=2)
        self.entry_xlabel = ctk.CTkEntry(frame, placeholder_text="X Label"); self.entry_xlabel.pack(fill="x", pady=2)
        self.entry_ylabel = ctk.CTkEntry(frame, placeholder_text="Y Label"); self.entry_ylabel.pack(fill="x", pady=2)
        
        # Fonts
        ctk.CTkLabel(frame, text="Font Sizes", font=("Roboto", 12, "bold")).pack(anchor="w", pady=(10,2))
        f_frame = ctk.CTkFrame(frame, fg_color="transparent"); f_frame.pack(fill="x")
        ctk.CTkLabel(f_frame, text="Title:").grid(row=0, column=0); self.entry_fs_title = ctk.CTkEntry(f_frame, width=40); self.entry_fs_title.grid(row=0, column=1); self.entry_fs_title.insert(0, "14")
        ctk.CTkLabel(f_frame, text="Label:").grid(row=0, column=2, padx=5); self.entry_fs_label = ctk.CTkEntry(f_frame, width=40); self.entry_fs_label.grid(row=0, column=3); self.entry_fs_label.insert(0, "12")
        ctk.CTkLabel(f_frame, text="Tick:").grid(row=1, column=0, pady=5); self.entry_fs_tick = ctk.CTkEntry(f_frame, width=40); self.entry_fs_tick.grid(row=1, column=1, pady=5); self.entry_fs_tick.insert(0, "10")

        # Range
        ctk.CTkLabel(frame, text="Plot Range (Min / Max)", font=("Roboto", 12, "bold")).pack(anchor="w", pady=(10,2))
        r_frame = ctk.CTkFrame(frame, fg_color="transparent"); r_frame.pack(fill="x")
        ctk.CTkLabel(r_frame, text="X (eV):", width=40).pack(side="left"); self.lim_x_min = ctk.CTkEntry(r_frame, width=50); self.lim_x_min.pack(side="left", padx=2); self.lim_x_max = ctk.CTkEntry(r_frame, width=50); self.lim_x_max.pack(side="left", padx=2)
        r2_frame = ctk.CTkFrame(frame, fg_color="transparent"); r2_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(r2_frame, text="Y (Int):", width=40).pack(side="left"); self.lim_y_min = ctk.CTkEntry(r2_frame, width=50); self.lim_y_min.pack(side="left", padx=2); self.lim_y_max = ctk.CTkEntry(r2_frame, width=50); self.lim_y_max.pack(side="left", padx=2)

        ctk.CTkButton(frame, text="Apply Settings", command=self.apply_graph_settings, fg_color="#E07A5F").pack(pady=20, fill="x")

    def _create_main_area(self):
        """グラフ描画エリア"""
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

    def apply_graph_settings(self):
        try:
            if self.entry_title.get(): self.ax.set_title(self.entry_title.get())
            if self.entry_xlabel.get(): self.ax.set_xlabel(self.entry_xlabel.get())
            if self.entry_ylabel.get(): self.ax.set_ylabel(self.entry_ylabel.get())
            
            self.ax.set_title(self.ax.get_title(), fontsize=int(self.entry_fs_title.get()))
            self.ax.set_xlabel(self.ax.get_xlabel(), fontsize=int(self.entry_fs_label.get()))
            self.ax.set_ylabel(self.ax.get_ylabel(), fontsize=int(self.entry_fs_label.get()))
            self.ax.tick_params(axis='both', which='major', labelsize=int(self.entry_fs_tick.get()))
            
            try: self.ax.set_xlim(float(self.lim_x_max.get()), float(self.lim_x_min.get())) 
            except: pass
            try: self.ax.set_ylim(float(self.lim_y_min.get()), float(self.lim_y_max.get()))
            except: pass
            
            self.canvas.draw()
        except Exception as e: messagebox.showerror("Error", str(e))

    def activate_selector(self, mode):
        self.selection_mode = mode
        if self.span: self.span.set_visible(False); self.span = None
        
        colors = {
            'bg': 'blue', 'slope': 'red',             # VBM
            'bg_peak': 'green',                       # BandGap
            'bg_base': 'blue', 'bg_slope': 'red',     # Linear
            'bg_deriv': 'orange'                      # Deriv
        }
        color = colors.get(mode, 'gray')
        
        self.span = SpanSelector(self.ax, self.on_select, 'horizontal', useblit=True, 
                                 props=dict(alpha=0.3, facecolor=color), interactive=True, drag_from_anywhere=True)
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
            'bg_deriv': (self.bg_deriv_min, self.bg_deriv_max)
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

    # ==========================================
    # 5. データ入出力 & 計算ロジック
    # ==========================================

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

            # 初期値設定
            min_e, max_e = np.min(self.energy), np.max(self.energy)
            
            # VBM Defaults
            self.entry_bg_min.delete(0, tk.END); self.entry_bg_min.insert(0, f"{min_e:.1f}")
            self.entry_bg_max.delete(0, tk.END); self.entry_bg_max.insert(0, f"{min_e+2.0:.1f}")
            self.entry_slope_min.delete(0, tk.END); self.entry_slope_min.insert(0, f"{min_e+3.0:.1f}")
            self.entry_slope_max.delete(0, tk.END); self.entry_slope_max.insert(0, f"{min_e+5.0:.1f}")
            
            # Band Gap Defaults
            self.bg_peak_min.delete(0, tk.END); self.bg_peak_min.insert(0, f"{min_e:.1f}")
            self.bg_peak_max.delete(0, tk.END); self.bg_peak_max.insert(0, f"{min_e+1.0:.1f}")
            self.bg_base_min.delete(0, tk.END); self.bg_base_min.insert(0, f"{min_e+10.0:.1f}")
            self.bg_base_max.delete(0, tk.END); self.bg_base_max.insert(0, f"{min_e+12.0:.1f}")
            self.bg_slope_min.delete(0, tk.END); self.bg_slope_min.insert(0, f"{min_e+13.0:.1f}")
            self.bg_slope_max.delete(0, tk.END); self.bg_slope_max.insert(0, f"{min_e+15.0:.1f}")
            self.bg_deriv_min.delete(0, tk.END); self.bg_deriv_min.insert(0, f"{min_e+10.0:.1f}")
            self.bg_deriv_max.delete(0, tk.END); self.bg_deriv_max.insert(0, f"{min_e+15.0:.1f}")
            
            # Reset & Draw
            self.chk_shirley_var.set(False); self.intensity_corrected = None; self.bg_data = None
            self.entry_title.delete(0, tk.END); self.entry_title.insert(0, f"XPS: {os.path.basename(file_path)}")
            self.lim_x_min.delete(0, tk.END); self.lim_x_min.insert(0, f"{min_e:.1f}")
            self.lim_x_max.delete(0, tk.END); self.lim_x_max.insert(0, f"{max_e:.1f}")
            self.lim_y_min.delete(0, tk.END); self.lim_y_min.insert(0, f"{np.min(self.intensity):.1f}")
            self.lim_y_max.delete(0, tk.END); self.lim_y_max.insert(0, f"{np.max(self.intensity):.1f}")
            
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
        self.ax.legend(); self.ax.grid(True); self.ax.invert_xaxis(); self.apply_graph_settings()
        self.canvas.draw()

    def get_current_intensity(self):
        return self.intensity_corrected if (self.chk_shirley_var.get() and self.intensity_corrected is not None) else self.intensity

    def calculate(self):
        """【Tab 1】VBM解析 (交点法)"""
        if self.energy is None: return
        try:
            y_data = self.get_current_intensity()
            bg_r = (float(self.entry_bg_min.get()), float(self.entry_bg_max.get()))
            sl_r = (float(self.entry_slope_min.get()), float(self.entry_slope_max.get()))

            # Fit Background
            mask_bg = (self.energy >= bg_r[0]) & (self.energy <= bg_r[1])
            popt_bg, _ = curve_fit(linear_func, self.energy[mask_bg], y_data[mask_bg])
            
            # Fit Slope
            mask_sl = (self.energy >= sl_r[0]) & (self.energy <= sl_r[1])
            popt_sl, _ = curve_fit(linear_func, self.energy[mask_sl], y_data[mask_sl])

            # Intersection
            vbm_x = (popt_sl[1] - popt_bg[1]) / (popt_bg[0] - popt_sl[0])
            vbm_y = linear_func(vbm_x, *popt_bg)
            
            self.vbm_label.configure(text=f"VBM: {vbm_x:.3f} eV")

            # Draw
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
        """【Tab 2】バンドギャップ解析: Linear Fit または Derivative Method"""
        if self.energy is None: return
        try:
            y_data = self.get_current_intensity()
            mode = self.bg_mode_var.get()
            
            # --- 共通: Main Peakの特定 ---
            pk_r = (float(self.bg_peak_min.get()), float(self.bg_peak_max.get()))
            mask_pk = (self.energy >= pk_r[0]) & (self.energy <= pk_r[1])
            if not np.any(mask_pk): raise ValueError("Peak範囲にデータがありません")
            
            subset_idx = np.where(mask_pk)[0]
            peak_idx = subset_idx[np.argmax(y_data[subset_idx])]
            peak_x = self.energy[peak_idx]
            peak_y = y_data[peak_idx]

            # ベース描画
            self.plot_base_graph()
            self.ax.plot(peak_x, peak_y, 'g*', markersize=15, zorder=5, label='Main Peak')
            self.ax.axvline(peak_x, color='green', linestyle=':', alpha=0.6)
            self.ax.axvspan(pk_r[0], pk_r[1], color='green', alpha=0.1)

            onset_x, onset_y, gap = 0, 0, 0

            if mode == "Linear Fit":
                # === Method A: 直線交点法 ===
                base_r = (float(self.bg_base_min.get()), float(self.bg_base_max.get()))
                sl_r = (float(self.bg_slope_min.get()), float(self.bg_slope_max.get()))
                
                mask_base = (self.energy >= base_r[0]) & (self.energy <= base_r[1])
                popt_base, _ = curve_fit(linear_func, self.energy[mask_base], y_data[mask_base])
                mask_sl = (self.energy >= sl_r[0]) & (self.energy <= sl_r[1])
                popt_sl, _ = curve_fit(linear_func, self.energy[mask_sl], y_data[mask_sl])
                
                onset_x = (popt_sl[1] - popt_base[1]) / (popt_base[0] - popt_sl[0])
                onset_y = linear_func(onset_x, *popt_base)
                gap = abs(onset_x - peak_x)
                
                x_plot = np.linspace(min(self.energy), max(self.energy), 200)
                self.ax.plot(x_plot, linear_func(x_plot, *popt_base), 'b--', alpha=0.5, label='Base Fit')
                self.ax.plot(x_plot, linear_func(x_plot, *popt_sl), 'r--', alpha=0.5, label='Slope Fit')
                self.ax.axvspan(base_r[0], base_r[1], color='blue', alpha=0.1)
                self.ax.axvspan(sl_r[0], sl_r[1], color='red', alpha=0.1)
                self.ax.plot(onset_x, onset_y, 'ro', markersize=8, zorder=5, label='Linear Onset')

            else:
                # === Method B: 2次微分法 (全体スムージング版) ===
                d_r = (float(self.bg_deriv_min.get()), float(self.bg_deriv_max.get()))
                
                # 1. まず全体をスムージングする (端点誤差を防ぐため)
                # データ点数が少ない場合は調整
                target_window = 51
                w_len = min(target_window, len(y_data))
                if w_len % 2 == 0: w_len -= 1
                if w_len < 3: w_len = 3
                
                # 全データに対してフィルタ適用
                y_smooth_all = savgol_filter(y_data, window_length=w_len, polyorder=2)

                # 2. 必要な範囲だけ切り出す
                mask_d = (self.energy >= d_r[0]) & (self.energy <= d_r[1])
                x_d = self.energy[mask_d]
                
                # 切り出した部分に対応するスムージング済みデータ
                y_d_smooth = y_smooth_all[mask_d] 
                
                if len(x_d) < 5: raise ValueError("微分解析用のデータ点数が少なすぎます")

                # ★ 可視化: 選択範囲のスムージング曲線を表示
                self.ax.plot(x_d, y_d_smooth, color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Smoothed (Slice)')

                # 3. 2次微分 (切り出した滑らかなデータに対して)
                d2y = np.gradient(np.gradient(y_d_smooth, x_d), x_d)
                
                # 4. 最大値探索
                max_d2_idx = np.argmax(d2y)
                max_val = d2y[max_d2_idx]
                
                if max_val <= 0:
                    raise ValueError("選択範囲内に明確な立ち上がり(正の曲率)が見つかりません。")

                onset_x = x_d[max_d2_idx]
                onset_y = y_d_smooth[max_d2_idx] # 平滑化後のY値を採用
                gap = abs(onset_x - peak_x)

                self.ax.axvspan(d_r[0], d_r[1], color='orange', alpha=0.1, label='Search Region')
                self.ax.plot(onset_x, onset_y, 'bx', markersize=10, markeredgewidth=3, zorder=6, label='Deriv Onset')
                self.ax.axvline(onset_x, color='orange', linestyle='--', alpha=0.8)

            # 共通: 矢印
            arrow_y = (peak_y + onset_y) / 2
            self.ax.annotate(f'Eg = {gap:.2f} eV', xy=(peak_x, arrow_y), xytext=(onset_x, arrow_y),
                             arrowprops=dict(arrowstyle='<->', color='purple', lw=2),
                             ha='center', va='bottom', fontsize=12, color='purple', fontweight='bold')
            
            self.lbl_res_gap.configure(text=f"Eg: {gap:.3f} eV ({mode})")
            self.ax.legend(); self.canvas.draw()

        except Exception as e: messagebox.showerror("Calc Error", str(e))

    def on_closing(self):
        """アプリ終了時処理"""
        plt.close('all'); self.quit(); self.destroy()

if __name__ == "__main__":
    app = XPS_VB_Edge_App()
    app.mainloop()