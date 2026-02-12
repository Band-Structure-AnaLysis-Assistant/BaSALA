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

def gaussian_func(x, a, mu, sigma, c):
    """
    ガウス関数 (Gaussian) + 定数オフセット
    ピークフィッティング用
    
    Args:
        x: x値
        a: 振幅 (Amplitude)
        mu: 中心位置 (Mean / Center)
        sigma: 標準偏差 (Width parameter)
        c: ベースラインオフセット (Constant background)
    """
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

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
    # 積分計算は「散乱源(低エネルギー) → 散乱結果(高エネルギー)」の順で行う物理モデルなため、
    # 数値を昇順(小さい順)に並べ替えて処理します。
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    n = len(y)
    
    # 2. 基準点の決定
    # I_start: 低エネルギー側（右端）の強度。ここがBGの基準レベル（始点）になります。
    # I_end:   高エネルギー側（左端）の強度。BGの到達点になります。
    I_start = y_sorted[0]
    I_end = y_sorted[-1]
    
    # 3. 初期BGの作成 (最初は平らな線からスタート)
    bg = np.full(n, I_start)
    
    # 4. 反復計算 (Iterative Calculation)
    for _ in range(max_iters):
        # 信号成分 = 現在の強度 - BG (マイナス値は物理的にありえないので0にクリップ)
        signal = y_sorted - I_start
        signal[signal < 0] = 0
        
        # 累積積分 (右端から左端へ面積を積み上げる)
        # 台形積分 (Trapezoidal rule) で近似計算
        cum_area = np.zeros(n)
        cum_area[1:] = np.cumsum((signal[:-1] + signal[1:]) / 2 * np.diff(x_sorted))
        
        total_area = cum_area[-1] # 全面積
        
        # 新しいBGの形状を計算
        # 「全体の段差(I_end - I_start)」を「面積の累積割合」に応じて配分する
        if total_area == 0:
            k = 0
        else:
            k = (I_end - I_start) / total_area
            
        bg_new = I_start + k * cum_area
        
        # 収束判定 (前回との差が許容値以下なら終了)
        if np.max(np.abs(bg_new - bg)) < tol:
            bg = bg_new
            break
            
        bg = bg_new

    # 5. 並び順を元に戻す (入力された元の配列順序に対応させる)
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
        self.title("XPS Analysis Suite - v1.3 (Auto Scaling)")
        self.geometry("1280x900")
        
        # 「×」ボタンで終了したときの処理を登録 (メモリ解放のため)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- メンバ変数（データ保持用）---
        self.file_path = None
        self.df = None
        self.energy = None              # x軸データ (Binding Energy)
        self.intensity = None           # y軸データ (Raw Intensity)
        self.intensity_corrected = None # Shirley補正後のy軸データ
        self.bg_data = None             # 計算されたShirley BG曲線
        
        # --- メンバ変数（ツール用）---
        self.span = None        # マウス選択範囲 (Matplotlib SpanSelector)
        self.selection_mode = None # 現在どの入力欄を選択中かを識別する文字列

        # --- GUI構築実行 ---
        self._create_sidebar()
        self._create_main_area()

    def _create_sidebar(self):
        """左側の操作パネル（サイドバー）を作成"""
        self.sidebar = ctk.CTkFrame(self, width=340, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)

        # タイトルロゴ
        self.logo_label = ctk.CTkLabel(self.sidebar, text="XPS Analysis\nv1.3", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.pack(padx=20, pady=(20, 10))

        # --- 共通操作エリア (読み込み & BG補正) ---
        self.common_frame = ctk.CTkFrame(self.sidebar)
        self.common_frame.pack(padx=10, pady=5, fill="x")
        
        # ファイル読み込みボタン
        self.load_btn = ctk.CTkButton(self.common_frame, text="Open CSV / Text", command=self.load_csv, fg_color="#1f538d")
        self.load_btn.pack(padx=5, pady=5, fill="x")
        
        # 区切り文字選択 (カンマ, タブ, スペース)
        self.sep_option = ctk.CTkComboBox(self.common_frame, values=[", (Comma)", "\\t (Tab)", "Space"], height=24)
        self.sep_option.set(", (Comma)")
        self.sep_option.pack(padx=5, pady=(0, 5))

        # Shirley BG補正スイッチ
        self.chk_shirley_var = ctk.BooleanVar(value=False)
        self.chk_shirley = ctk.CTkCheckBox(self.common_frame, text="Apply Shirley BG", variable=self.chk_shirley_var, command=self.on_shirley_toggle)
        self.chk_shirley.pack(padx=10, pady=10, anchor="w")

        # --- 機能切り替えタブ (Tab View) ---
        self.tabview = ctk.CTkTabview(self.sidebar, width=320)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.tab_analysis = self.tabview.add("Analysis")       # Tab 1: VBM解析
        self.tab_bg = self.tabview.add("Band Gap")             # Tab 2: バンドギャップ解析
        # ※ Graph Settings タブは削除しました

        # 各タブの中身を初期化する関数を呼び出し
        self._init_vbm_tab()
        self._init_bandgap_tab()

    def _init_vbm_tab(self):
        """Tab 1: VBM解析 (直線交点法) のUI構築"""
        frame = ctk.CTkFrame(self.tab_analysis, fg_color="transparent")
        frame.pack(fill="both", expand=True)
        ctk.CTkLabel(frame, text="Determine VBM by Intersection", font=("Roboto", 12, "bold")).pack(pady=5)

        # 1. Background Range (ベースライン)
        ctk.CTkLabel(frame, text="1. Background Range:", font=("Roboto", 11)).pack(anchor="w", padx=5)
        bg_frame = ctk.CTkFrame(frame, fg_color="transparent")
        bg_frame.pack(fill="x", padx=5)
        self.entry_bg_min = ctk.CTkEntry(bg_frame, width=50); self.entry_bg_min.pack(side="left")
        ctk.CTkLabel(bg_frame, text="-").pack(side="left")
        self.entry_bg_max = ctk.CTkEntry(bg_frame, width=50); self.entry_bg_max.pack(side="left")
        ctk.CTkButton(bg_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg")).pack(side="right")

        # 2. Slope Range (価電子帯の立ち上がり)
        ctk.CTkLabel(frame, text="2. VB Slope Range:", font=("Roboto", 11)).pack(anchor="w", padx=5, pady=(10,0))
        slope_frame = ctk.CTkFrame(frame, fg_color="transparent")
        slope_frame.pack(fill="x", padx=5)
        self.entry_slope_min = ctk.CTkEntry(slope_frame, width=50); self.entry_slope_min.pack(side="left")
        ctk.CTkLabel(slope_frame, text="-").pack(side="left")
        self.entry_slope_max = ctk.CTkEntry(slope_frame, width=50); self.entry_slope_max.pack(side="left")
        ctk.CTkButton(slope_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("slope")).pack(side="right")

        # 選択解除ボタン
        self.btn_reset_mode = ctk.CTkButton(frame, text="Stop Selection", fg_color="transparent", border_width=1, height=24, command=self.deactivate_selector)
        self.btn_reset_mode.pack(pady=10)

        # 計算実行ボタン
        self.calc_btn = ctk.CTkButton(frame, text="Calculate VBM", command=self.calculate, fg_color="#2d8d2d", state="disabled")
        self.calc_btn.pack(padx=10, pady=15, fill="x")

        # 結果表示ラベル
        self.vbm_label = ctk.CTkLabel(frame, text="VBM: --- eV", font=ctk.CTkFont(size=20, weight="bold"), text_color="#4db6ac")
        self.vbm_label.pack(pady=5)

    def _init_bandgap_tab(self):
        """
        Tab 2: バンドギャップ解析 (モード切替機能付き) のUI構築
        """
        self.bg_tab_frame = ctk.CTkFrame(self.tab_bg, fg_color="transparent")
        self.bg_tab_frame.pack(fill="both", expand=True)

        # ★ モード切替スイッチ (Linear Fit / Derivative)
        self.bg_mode_var = ctk.StringVar(value="Linear Fit")
        self.seg_bg_mode = ctk.CTkSegmentedButton(self.bg_tab_frame, values=["Linear Fit", "Derivative"], 
                                                  variable=self.bg_mode_var, command=self.update_bg_ui)
        self.seg_bg_mode.pack(pady=10, padx=10, fill="x")
        self.seg_bg_mode.set("Linear Fit") # 初期選択

        # 1. Main Peak (共通項目: 常に一番上に表示)
        ctk.CTkLabel(self.bg_tab_frame, text="1. Main Peak (Eg Reference):", font=("Roboto", 11)).pack(anchor="w", padx=5)
        self.p_frame = ctk.CTkFrame(self.bg_tab_frame, fg_color="transparent")
        self.p_frame.pack(fill="x", padx=5)
        self.bg_peak_min = ctk.CTkEntry(self.p_frame, width=50); self.bg_peak_min.pack(side="left")
        ctk.CTkLabel(self.p_frame, text="-").pack(side="left")
        self.bg_peak_max = ctk.CTkEntry(self.p_frame, width=50); self.bg_peak_max.pack(side="left")
        ctk.CTkButton(self.p_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_peak")).pack(side="right")

        # ★★★ 可変入力エリア用コンテナ (ここにPackすることで順序を固定) ★★★
        self.bg_input_container = ctk.CTkFrame(self.bg_tab_frame, fg_color="transparent")
        self.bg_input_container.pack(fill="x", pady=5)

        # --- A. Linear Mode用 入力部品群 (Parent = container) ---
        self.frame_linear = ctk.CTkFrame(self.bg_input_container, fg_color="transparent")
        
        # Loss Base (平らな部分)
        ctk.CTkLabel(self.frame_linear, text="2. Loss Base (Flat):", font=("Roboto", 11)).pack(anchor="w", padx=5, pady=(5,0))
        self.b_frame = ctk.CTkFrame(self.frame_linear, fg_color="transparent")
        self.b_frame.pack(fill="x", padx=5)
        self.bg_base_min = ctk.CTkEntry(self.b_frame, width=50); self.bg_base_min.pack(side="left")
        ctk.CTkLabel(self.b_frame, text="-").pack(side="left")
        self.bg_base_max = ctk.CTkEntry(self.b_frame, width=50); self.bg_base_max.pack(side="left")
        ctk.CTkButton(self.b_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_base")).pack(side="right")

        # Loss Slope (立ち上がり部分)
        ctk.CTkLabel(self.frame_linear, text="3. Loss Slope (Rise):", font=("Roboto", 11)).pack(anchor="w", padx=5, pady=(5,0))
        self.s_frame = ctk.CTkFrame(self.frame_linear, fg_color="transparent")
        self.s_frame.pack(fill="x", padx=5)
        self.bg_slope_min = ctk.CTkEntry(self.s_frame, width=50); self.bg_slope_min.pack(side="left")
        ctk.CTkLabel(self.s_frame, text="-").pack(side="left")
        self.bg_slope_max = ctk.CTkEntry(self.s_frame, width=50); self.bg_slope_max.pack(side="left")
        ctk.CTkButton(self.s_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_slope")).pack(side="right")

        # --- B. Derivative Mode用 入力部品群 (Parent = container) ---
        self.frame_deriv = ctk.CTkFrame(self.bg_input_container, fg_color="transparent")
        
        # Onset Region (1つの範囲選択でOK)
        ctk.CTkLabel(self.frame_deriv, text="2. Onset Search Region:", font=("Roboto", 11)).pack(anchor="w", padx=5, pady=(5,0))
        ctk.CTkLabel(self.frame_deriv, text="(Includes BG end & Rise start)", font=("Roboto", 10), text_color="gray").pack(anchor="w", padx=5)
        
        self.d_frame = ctk.CTkFrame(self.frame_deriv, fg_color="transparent")
        self.d_frame.pack(fill="x", padx=5)
        self.bg_deriv_min = ctk.CTkEntry(self.d_frame, width=50); self.bg_deriv_min.pack(side="left")
        ctk.CTkLabel(self.d_frame, text="-").pack(side="left")
        self.bg_deriv_max = ctk.CTkEntry(self.d_frame, width=50); self.bg_deriv_max.pack(side="left")
        ctk.CTkButton(self.d_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_deriv")).pack(side="right")

        # 初期表示 (Linear Mode)
        self.frame_linear.pack(fill="x", pady=5)

        # --- 共通ボタンエリア (常にコンテナの下に表示される) ---
        ctk.CTkButton(self.bg_tab_frame, text="Stop Selection", fg_color="transparent", border_width=1, height=24, command=self.deactivate_selector).pack(pady=10)
        
        self.calc_bg_btn = ctk.CTkButton(self.bg_tab_frame, text="Calculate Band Gap", command=self.calculate_bandgap, fg_color="#E07A5F", state="disabled")
        self.calc_bg_btn.pack(pady=5, fill="x")
        
        # 結果表示ラベル
        self.lbl_res_gap = ctk.CTkLabel(self.bg_tab_frame, text="Eg: --- eV", font=ctk.CTkFont(size=18, weight="bold"), text_color="#E07A5F")
        self.lbl_res_gap.pack(pady=5)

    def update_bg_ui(self, value):
        """モード切替時にUI部品を出し分ける処理"""
        if value == "Linear Fit":
            self.frame_deriv.pack_forget() # Deriv用UIを隠す
            self.frame_linear.pack(fill="x", pady=5) # Linear用UIを表示
        else:
            self.frame_linear.pack_forget() # Linear用UIを隠す
            self.frame_deriv.pack(fill="x", pady=5) # Deriv用UIを表示

    def _create_main_area(self):
        """右側のメイン描画エリアを作成"""
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # MatplotlibのFigure作成
        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.ax.set_xlabel("Binding Energy (eV)")
        self.ax.set_ylabel("Intensity (a.u.)")
        
        # XPSの慣習に従い、X軸(結合エネルギー)は反転(左が高エネルギー)
        self.ax.invert_xaxis()
        
        # Canvasに埋め込み
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # ツールバー (ズーム、保存など)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.main_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ==========================================
    # 4. 操作ロジック (グラフ設定・マウス操作)
    # ==========================================

    def auto_scale_y(self):
        """
        Y軸の範囲を自動調整する機能
        - 上限: 元データ(Raw)の最大値 + マージン
        - 下限: 表示中のデータ(補正後はCorrected)の最小値 - マージン
        """
        if self.intensity is None: return
        
        # 上限は常に元のスペクトルの最大値を基準にする（全体像を維持）
        y_max_raw = np.max(self.intensity)
        
        # 下限は現在表示しているデータ（補正されているかどうか）に合わせる
        y_current = self.get_current_intensity()
        y_min_curr = np.min(y_current)
        
        # マージン計算 (表示範囲の約5%)
        amp = y_max_raw - y_min_curr
        margin = amp * 0.05 if amp > 0 else 10.0
        
        # 範囲設定
        self.ax.set_ylim(y_min_curr - margin, y_max_raw + margin)

    def activate_selector(self, mode):
        """マウス範囲選択モードを有効化"""
        self.selection_mode = mode
        # 既存のセレクタがあれば消去
        if self.span: self.span.set_visible(False); self.span = None
        
        # モードごとの色定義 (視認性を高めるため)
        colors = {
            'bg': 'blue', 'slope': 'red',             # VBM
            'bg_peak': 'green',                       # BandGap: Peak
            'bg_base': 'blue', 'bg_slope': 'red',     # BandGap: Linear
            'bg_deriv': 'orange'                      # BandGap: Deriv
        }
        color = colors.get(mode, 'gray')
        
        # SpanSelector起動
        self.span = SpanSelector(self.ax, self.on_select, 'horizontal', useblit=True, 
                                 props=dict(alpha=0.3, facecolor=color), interactive=True, drag_from_anywhere=True)
        self.canvas.draw()

    def deactivate_selector(self):
        """マウス範囲選択モードを終了"""
        if self.span: self.span.set_visible(False); self.span = None
        self.selection_mode = None
        self.canvas.draw()

    def on_select(self, vmin, vmax):
        """ドラッグ終了時のコールバック: 選択範囲をEntryボックスに入力"""
        min_val, max_val = sorted([vmin, vmax])
        v_min, v_max = f"{min_val:.2f}", f"{max_val:.2f}"
        
        # 現在のモードに対応するEntryボックスのペアを取得
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
        """Shirleyチェックボックス切替時の処理"""
        if self.energy is None: return
        
        if self.chk_shirley_var.get():
            try:
                # 計算実行
                self.bg_data = calculate_shirley_bg(self.energy, self.intensity)
                self.intensity_corrected = self.intensity - self.bg_data
            except: 
                messagebox.showerror("Error", "Shirley calculation failed.")
                self.chk_shirley_var.set(False)
        else: 
            # OFFの場合リセット
            self.bg_data = None; self.intensity_corrected = None
        
        # グラフを再描画して反映
        self.plot_base_graph()

    # ==========================================
    # 5. データ入出力 & 計算ロジック
    # ==========================================

    def load_csv(self):
        """CSVファイルの読み込み"""
        file_path = filedialog.askopenfilename(filetypes=[("Data Files", "*.csv *.txt *.dat"), ("All Files", "*.*")])
        if not file_path: return
        
        # 選択された区切り文字を取得
        sep = {", (Comma)": ",", "\\t (Tab)": "\t", "Space": r"\s+"}[self.sep_option.get()]
        
        try:
            # pandasで読み込み
            self.df = pd.read_csv(file_path, sep=sep, header=None, engine='python')
            if self.df.shape[1] < 2: return
            
            # 数値変換とNaN除去
            self.energy = pd.to_numeric(self.df.iloc[:, 0], errors='coerce').values
            self.intensity = pd.to_numeric(self.df.iloc[:, 1], errors='coerce').values
            mask = ~np.isnan(self.energy) & ~np.isnan(self.intensity)
            self.energy = self.energy[mask]; self.intensity = self.intensity[mask]
            if len(self.energy) == 0: return

            # --- 初期値の自動入力 (全範囲から推定) ---
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
            
            # 状態リセット
            self.chk_shirley_var.set(False); self.intensity_corrected = None; self.bg_data = None
            
            # 描画 (自動軸調整はplot_base_graph内で呼ばれる)
            self.plot_base_graph()
            self.calc_btn.configure(state="normal"); self.calc_bg_btn.configure(state="normal")
            
        except Exception as e: messagebox.showerror("Error", str(e))

    def plot_base_graph(self):
        """ベースとなるスペクトルを描画 (計算結果プロットなし)"""
        self.ax.clear()
        if self.chk_shirley_var.get() and self.intensity_corrected is not None:
            # Shirley ON: 生データ(薄い) + BG線 + 補正データ(濃い)
            self.ax.plot(self.energy, self.intensity, color='gray', alpha=0.3, label='Raw Data')
            self.ax.plot(self.energy, self.bg_data, color='gray', linestyle='--', alpha=0.5, label='Shirley BG')
            self.ax.plot(self.energy, self.intensity_corrected, color='#4a90e2', linewidth=1.5, label='Corrected')
        else:
            # Shirley OFF: 生データのみ
            self.ax.plot(self.energy, self.intensity, color='#4a90e2', linewidth=1.5, label='Raw Spectrum')
            
        self.ax.legend(); self.ax.grid(True); self.ax.invert_xaxis()
        
        # ★ 自動軸調整を適用
        self.auto_scale_y()
        self.canvas.draw()

    def get_current_intensity(self):
        """現在使用すべき強度データを取得 (Shirley ONなら補正データ)"""
        return self.intensity_corrected if (self.chk_shirley_var.get() and self.intensity_corrected is not None) else self.intensity

    def calculate(self):
        """【Tab 1】VBM解析: 直線交点法 (Linear Intersection Method)"""
        if self.energy is None: return
        try:
            y_data = self.get_current_intensity()
            
            # 範囲取得
            bg_r = (float(self.entry_bg_min.get()), float(self.entry_bg_max.get()))
            sl_r = (float(self.entry_slope_min.get()), float(self.entry_slope_max.get()))

            # フィッティング (Background)
            mask_bg = (self.energy >= bg_r[0]) & (self.energy <= bg_r[1])
            popt_bg, _ = curve_fit(linear_func, self.energy[mask_bg], y_data[mask_bg])
            
            # フィッティング (Slope)
            mask_sl = (self.energy >= sl_r[0]) & (self.energy <= sl_r[1])
            popt_sl, _ = curve_fit(linear_func, self.energy[mask_sl], y_data[mask_sl])

            # 交点算出
            vbm_x = (popt_sl[1] - popt_bg[1]) / (popt_bg[0] - popt_sl[0])
            vbm_y = linear_func(vbm_x, *popt_bg)
            
            self.vbm_label.configure(text=f"VBM: {vbm_x:.3f} eV")

            # 描画
            self.plot_base_graph() # ベース描画（ここでオートスケールされる）
            
            x_plot = np.linspace(min(self.energy), max(self.energy), 200)
            self.ax.plot(x_plot, linear_func(x_plot, *popt_bg), 'b--', alpha=0.8, label='Base Fit')
            self.ax.plot(x_plot, linear_func(x_plot, *popt_sl), 'r--', alpha=0.8, label='Slope Fit')
            self.ax.plot(vbm_x, vbm_y, 'go', markersize=10, zorder=5, label=f'VBM={vbm_x:.2f}eV')
            self.ax.axvline(vbm_x, color='green', linestyle=':', alpha=0.8)
            # 範囲可視化
            self.ax.axvspan(bg_r[0], bg_r[1], color='blue', alpha=0.1)
            self.ax.axvspan(sl_r[0], sl_r[1], color='red', alpha=0.1)
            
            self.ax.legend(); self.canvas.draw()
        except Exception as e: messagebox.showerror("Calc Error", str(e))

    def calculate_bandgap(self):
        """
        【Tab 2】バンドギャップ解析
        計算フロー:
        1. 指定されたPeak範囲でガウスフィッティングし、真のピーク位置を特定
        2. LinearまたはDerivativeモードでオンセット(立ち上がり)位置を特定
        3. 差分(Eg)を計算
        """
        if self.energy is None: return
        try:
            y_data = self.get_current_intensity()
            mode = self.bg_mode_var.get()
            
            # --- 1. Main Peak位置の特定 (Gaussian Fit) ---
            pk_r = (float(self.bg_peak_min.get()), float(self.bg_peak_max.get()))
            mask_pk = (self.energy >= pk_r[0]) & (self.energy <= pk_r[1])
            if not np.any(mask_pk): raise ValueError("Peak範囲にデータがありません")
            
            # フィッティング用データ抽出
            x_pk_fit = self.energy[mask_pk]
            y_pk_fit = y_data[mask_pk]
            
            # 初期パラメータ推定 [Amp, Mean, Sigma, Offset]
            p0_amp = np.max(y_pk_fit) - np.min(y_pk_fit)
            p0_mu = x_pk_fit[np.argmax(y_pk_fit)]
            p0_sigma = (np.max(x_pk_fit) - np.min(x_pk_fit)) / 4
            p0_c = np.min(y_pk_fit)
            p0 = [p0_amp, p0_mu, p0_sigma, p0_c]
            
            # ガウスフィッティング実行
            try:
                popt_gauss, _ = curve_fit(gaussian_func, x_pk_fit, y_pk_fit, p0=p0, maxfev=2000)
                peak_x = popt_gauss[1] # Mean (Center)
                peak_y = gaussian_func(peak_x, *popt_gauss) # Height at center
            except:
                # 失敗時は単純な最大値でフォールバック
                print("Gaussian fit failed, using argmax.")
                idx_max = np.argmax(y_pk_fit)
                peak_x = x_pk_fit[idx_max]
                peak_y = y_pk_fit[idx_max]
                popt_gauss = None

            # ベース描画（オートスケール含む）
            self.plot_base_graph()
            
            # フィッティング曲線の描画 (成功時のみ)
            if popt_gauss is not None:
                x_fine = np.linspace(pk_r[0], pk_r[1], 100)
                self.ax.plot(x_fine, gaussian_func(x_fine, *popt_gauss), color='lime', linestyle='--', linewidth=1.5, label='Gaussian Fit')

            # ピーク点プロット
            self.ax.plot(peak_x, peak_y, 'g*', markersize=15, zorder=5, label='Main Peak')
            self.ax.axvline(peak_x, color='green', linestyle=':', alpha=0.6)
            self.ax.axvspan(pk_r[0], pk_r[1], color='green', alpha=0.1)

            # --- 2. Onset位置の特定 ---
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
                target_window = 51
                w_len = min(target_window, len(y_data))
                if w_len % 2 == 0: w_len -= 1
                if w_len < 3: w_len = 3
                
                y_smooth_all = savgol_filter(y_data, window_length=w_len, polyorder=2)

                # 2. 必要な範囲だけ切り出す
                mask_d = (self.energy >= d_r[0]) & (self.energy <= d_r[1])
                x_d = self.energy[mask_d]
                y_d_smooth = y_smooth_all[mask_d] 
                
                if len(x_d) < 5: raise ValueError("微分解析用のデータ点数が少なすぎます")

                # ★ 可視化: 選択範囲のスムージング曲線を表示
                self.ax.plot(x_d, y_d_smooth, color='orange', linestyle=':', linewidth=2, alpha=0.8, label='Smoothed (Slice)')

                # 3. 2次微分 (切り出した滑らかなデータに対して)
                d2y = np.gradient(np.gradient(y_d_smooth, x_d), x_d)
                
                # 4. 最大値探索 (正の曲率のみ)
                max_d2_idx = np.argmax(d2y)
                max_val = d2y[max_d2_idx]
                
                # 安全装置: 2次微分が負なら立ち上がりではないと判断
                if max_val <= 0:
                    raise ValueError("選択範囲内に明確な立ち上がり(正の曲率)が見つかりません。")

                onset_x = x_d[max_d2_idx]
                onset_y = y_d_smooth[max_d2_idx] # 平滑化後のY値を採用
                gap = abs(onset_x - peak_x)

                self.ax.axvspan(d_r[0], d_r[1], color='orange', alpha=0.1, label='Search Region')
                self.ax.plot(onset_x, onset_y, 'bx', markersize=10, markeredgewidth=3, zorder=6, label='Deriv Onset')
                self.ax.axvline(onset_x, color='orange', linestyle='--', alpha=0.8)

            # 共通: 結果矢印の描画
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