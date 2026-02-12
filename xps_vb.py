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
# 設定エリア
# ==========================================
ctk.set_appearance_mode("Dark")       # ダークモード (Lightにも変更可)
ctk.set_default_color_theme("dark-blue") # テーマカラー

# ==========================================
# 数学・物理計算関数
# ==========================================

def linear_func(x, a, b):
    """線形近似用関数: y = ax + b"""
    return a * x + b

def calculate_shirley_bg(x, y, tol=1e-5, max_iters=50):
    """
    Shirley法によるバックグラウンド（BG）計算
    
    XPSスペクトルにおける散乱電子の影響を除去するために使用します。
    高結合エネルギー側（左側）のBGの高さは、低結合エネルギー側（右側）の
    ピーク面積に比例するという原理に基づき、反復計算で求めます。
    
    Args:
        x: Binding Energy (eV)
        y: Intensity (Counts)
        tol: 収束判定の許容誤差
        max_iters: 最大反復回数
    """
    # 1. データのソート
    # XPSデータは通常「左が高エネルギー」だが、積分計算のために「右（低エネルギー）から左」へ
    # 処理しやすいよう、数値を小さい順（昇順）に並べ替える。
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    n = len(y)
    
    # 2. 基準点の決定
    # I_start: 低エネルギー側（右端）の強度。ここがBGの基準レベルになる。
    # I_end:   高エネルギー側（左端）の強度。
    I_start = y_sorted[0]
    I_end = y_sorted[-1]
    
    # 3. 初期BGの作成（平坦なラインからスタート）
    bg = np.full(n, I_start)
    
    # 4. 反復計算 (Iterative Calculation)
    for _ in range(max_iters):
        # 信号成分 = 現在の強度 - 推定BG（マイナス値は0にする）
        signal = y_sorted - I_start
        signal[signal < 0] = 0
        
        # 累積積分 (右端から左端へ面積を積み上げる)
        # np.trapz は古いNumPyでも動く台形積分
        cum_area = np.zeros(n)
        cum_area[1:] = np.cumsum((signal[:-1] + signal[1:]) / 2 * np.diff(x_sorted))
        
        total_area = cum_area[-1] # 全面積
        
        # 新しいBGの形状を計算
        # BGの立ち上がり具合 = 全体の段差 * (その地点までの面積 / 全面積)
        if total_area == 0:
            k = 0
        else:
            k = (I_end - I_start) / total_area
            
        bg_new = I_start + k * cum_area
        
        # 収束判定（前回との差が許容値以下なら終了）
        if np.max(np.abs(bg_new - bg)) < tol:
            bg = bg_new
            break
            
        bg = bg_new

    # 5. 並び順を元に戻す（元の配列順序に対応させる）
    bg_original_order = np.zeros(n)
    bg_original_order[sorted_indices] = bg
    
    return bg_original_order

# ==========================================
# GUI アプリケーションクラス
# ==========================================

class XPS_VB_Edge_App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # ウィンドウの基本設定
        self.title("XPS Analysis Suite - v0.7 (Full Integration)")
        self.geometry("1280x900")
        
        # 「×」ボタンで終了したときの処理を登録
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- メンバ変数（データ保持用）---
        self.file_path = None
        self.df = None
        self.energy = None              # Binding Energy軸
        self.intensity = None           # 生データ強度
        self.intensity_corrected = None # Shirley補正後の強度
        self.bg_data = None             # 計算されたShirley BG
        
        # --- メンバ変数（ツール用）---
        self.span = None        # マウス選択範囲管理用
        self.selection_mode = None # 現在どの入力ボックスを選択中か

        # --- 画面構築 ---
        self._create_sidebar()
        self._create_main_area()

    def _create_sidebar(self):
        """左側の操作パネル（サイドバー）を作成"""
        self.sidebar = ctk.CTkFrame(self, width=340, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)

        # タイトルロゴ
        self.logo_label = ctk.CTkLabel(self.sidebar, text="XPS Analysis\nv0.7", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.pack(padx=20, pady=(20, 10))

        # --- 共通操作エリア (読み込み & BG補正) ---
        self.common_frame = ctk.CTkFrame(self.sidebar)
        self.common_frame.pack(padx=10, pady=5, fill="x")
        
        # CSV読み込みボタン
        self.load_btn = ctk.CTkButton(self.common_frame, text="Open CSV / Text", command=self.load_csv, fg_color="#1f538d")
        self.load_btn.pack(padx=5, pady=5, fill="x")
        
        # 区切り文字選択
        self.sep_option = ctk.CTkComboBox(self.common_frame, values=[", (Comma)", "\\t (Tab)", "Space"], height=24)
        self.sep_option.set(", (Comma)")
        self.sep_option.pack(padx=5, pady=(0, 5))

        # Shirley補正チェックボックス
        self.chk_shirley_var = ctk.BooleanVar(value=False)
        self.chk_shirley = ctk.CTkCheckBox(self.common_frame, text="Apply Shirley BG Subtraction", variable=self.chk_shirley_var, command=self.on_shirley_toggle)
        self.chk_shirley.pack(padx=10, pady=10, anchor="w")

        # --- 機能切り替えタブ (Tab View) ---
        self.tabview = ctk.CTkTabview(self.sidebar, width=320)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)
        
        # タブの追加
        self.tab_analysis = self.tabview.add("VBM")       # VBM解析用
        self.tab_bg = self.tabview.add("Eg")             # バンドギャップ解析用
        self.tab_graph = self.tabview.add("Graph Settings")    # 見た目設定用

        # 各タブの中身を構築
        self._init_vbm_tab()
        self._init_bandgap_tab()
        self._init_graph_tab()

    def _init_vbm_tab(self):
        """Tab 1: VBM解析 (交点法) のUI構築"""
        frame = ctk.CTkFrame(self.tab_analysis, fg_color="transparent")
        frame.pack(fill="both", expand=True)

        ctk.CTkLabel(frame, text="Determine VBM by Intersection", font=("Roboto", 12, "bold")).pack(pady=5)

        # 1. Background Range
        ctk.CTkLabel(frame, text="1. Background Range:", font=("Roboto", 11)).pack(anchor="w", padx=5)
        bg_frame = ctk.CTkFrame(frame, fg_color="transparent")
        bg_frame.pack(fill="x", padx=5)
        self.entry_bg_min = ctk.CTkEntry(bg_frame, width=50); self.entry_bg_min.pack(side="left")
        ctk.CTkLabel(bg_frame, text="-").pack(side="left")
        self.entry_bg_max = ctk.CTkEntry(bg_frame, width=50); self.entry_bg_max.pack(side="left")
        ctk.CTkButton(bg_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg")).pack(side="right")

        # 2. Slope Range
        ctk.CTkLabel(frame, text="2. VB Slope Range:", font=("Roboto", 11)).pack(anchor="w", padx=5, pady=(10,0))
        slope_frame = ctk.CTkFrame(frame, fg_color="transparent")
        slope_frame.pack(fill="x", padx=5)
        self.entry_slope_min = ctk.CTkEntry(slope_frame, width=50); self.entry_slope_min.pack(side="left")
        ctk.CTkLabel(slope_frame, text="-").pack(side="left")
        self.entry_slope_max = ctk.CTkEntry(slope_frame, width=50); self.entry_slope_max.pack(side="left")
        ctk.CTkButton(slope_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("slope")).pack(side="right")

        # 解除ボタン
        self.btn_reset_mode = ctk.CTkButton(frame, text="Stop Selection", fg_color="transparent", border_width=1, height=24, command=self.deactivate_selector)
        self.btn_reset_mode.pack(pady=10)

        # 計算ボタン
        self.calc_btn = ctk.CTkButton(frame, text="Calculate VBM", command=self.calculate, fg_color="#2d8d2d", state="disabled")
        self.calc_btn.pack(padx=10, pady=15, fill="x")

        # 結果ラベル
        self.vbm_label = ctk.CTkLabel(frame, text="VBM: --- eV", font=ctk.CTkFont(size=20, weight="bold"), text_color="#4db6ac")
        self.vbm_label.pack(pady=5)

    def _init_bandgap_tab(self):
        """Tab 2: バンドギャップ解析 (Energy Loss法) のUI構築"""
        self.bg_tab_frame = ctk.CTkFrame(self.tab_bg, fg_color="transparent")
        self.bg_tab_frame.pack(fill="both", expand=True)

        ctk.CTkLabel(self.bg_tab_frame, text="Eg = Loss Onset - Peak", font=("Roboto", 12, "bold")).pack(pady=5)

        # 1. Main Peak Range
        ctk.CTkLabel(self.bg_tab_frame, text="1. Main Peak Region:", font=("Roboto", 11)).pack(anchor="w", padx=5)
        self.p_frame = ctk.CTkFrame(self.bg_tab_frame, fg_color="transparent")
        self.p_frame.pack(fill="x", padx=5)
        self.bg_peak_min = ctk.CTkEntry(self.p_frame, width=50); self.bg_peak_min.pack(side="left")
        ctk.CTkLabel(self.p_frame, text="-").pack(side="left")
        self.bg_peak_max = ctk.CTkEntry(self.p_frame, width=50); self.bg_peak_max.pack(side="left")
        ctk.CTkButton(self.p_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_peak")).pack(side="right")

        # 2. Loss Base Range
        ctk.CTkLabel(self.bg_tab_frame, text="2. Loss Base Region:", font=("Roboto", 11)).pack(anchor="w", padx=5, pady=(5,0))
        self.b_frame = ctk.CTkFrame(self.bg_tab_frame, fg_color="transparent")
        self.b_frame.pack(fill="x", padx=5)
        self.bg_base_min = ctk.CTkEntry(self.b_frame, width=50); self.bg_base_min.pack(side="left")
        ctk.CTkLabel(self.b_frame, text="-").pack(side="left")
        self.bg_base_max = ctk.CTkEntry(self.b_frame, width=50); self.bg_base_max.pack(side="left")
        ctk.CTkButton(self.b_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_base")).pack(side="right")

        # 3. Loss Slope Range
        ctk.CTkLabel(self.bg_tab_frame, text="3. Loss Slope Region:", font=("Roboto", 11)).pack(anchor="w", padx=5, pady=(5,0))
        self.s_frame = ctk.CTkFrame(self.bg_tab_frame, fg_color="transparent")
        self.s_frame.pack(fill="x", padx=5)
        self.bg_slope_min = ctk.CTkEntry(self.s_frame, width=50); self.bg_slope_min.pack(side="left")
        ctk.CTkLabel(self.s_frame, text="-").pack(side="left")
        self.bg_slope_max = ctk.CTkEntry(self.s_frame, width=50); self.bg_slope_max.pack(side="left")
        ctk.CTkButton(self.s_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg_slope")).pack(side="right")

        # 停止 & 計算ボタン
        ctk.CTkButton(self.bg_tab_frame, text="Stop Selection", fg_color="transparent", border_width=1, height=24, command=self.deactivate_selector).pack(pady=10)
        self.calc_bg_btn = ctk.CTkButton(self.bg_tab_frame, text="Calculate Band Gap", command=self.calculate_bandgap, fg_color="#E07A5F", state="disabled")
        self.calc_bg_btn.pack(pady=5, fill="x")
        
        # 結果ラベル
        self.lbl_res_gap = ctk.CTkLabel(self.bg_tab_frame, text="Eg: --- eV", font=ctk.CTkFont(size=18, weight="bold"), text_color="#E07A5F")
        self.lbl_res_gap.pack(pady=5)

    def _init_graph_tab(self):
        """Tab 3: グラフの見た目編集UI構築"""
        frame = ctk.CTkFrame(self.tab_graph, fg_color="transparent")
        frame.pack(fill="both", expand=True)
        
        # タイトルとラベル設定
        ctk.CTkLabel(frame, text="Labels & Title", font=("Roboto", 12, "bold")).pack(anchor="w", pady=2)
        self.entry_title = ctk.CTkEntry(frame, placeholder_text="Graph Title"); self.entry_title.pack(fill="x", pady=2)
        self.entry_xlabel = ctk.CTkEntry(frame, placeholder_text="X Label"); self.entry_xlabel.pack(fill="x", pady=2)
        self.entry_ylabel = ctk.CTkEntry(frame, placeholder_text="Y Label"); self.entry_ylabel.pack(fill="x", pady=2)
        
        # フォントサイズ設定
        ctk.CTkLabel(frame, text="Font Sizes", font=("Roboto", 12, "bold")).pack(anchor="w", pady=(10,2))
        f_frame = ctk.CTkFrame(frame, fg_color="transparent")
        f_frame.pack(fill="x")
        ctk.CTkLabel(f_frame, text="Title:").grid(row=0, column=0); self.entry_fs_title = ctk.CTkEntry(f_frame, width=40); self.entry_fs_title.grid(row=0, column=1)
        self.entry_fs_title.insert(0, "14")
        ctk.CTkLabel(f_frame, text="Label:").grid(row=0, column=2, padx=5); self.entry_fs_label = ctk.CTkEntry(f_frame, width=40); self.entry_fs_label.grid(row=0, column=3)
        self.entry_fs_label.insert(0, "12")
        ctk.CTkLabel(f_frame, text="Tick:").grid(row=1, column=0, pady=5); self.entry_fs_tick = ctk.CTkEntry(f_frame, width=40); self.entry_fs_tick.grid(row=1, column=1, pady=5)
        self.entry_fs_tick.insert(0, "10")

        # 軸範囲設定 (Min/Max)
        ctk.CTkLabel(frame, text="Plot Range (Min / Max)", font=("Roboto", 12, "bold")).pack(anchor="w", pady=(10,2))
        r_frame = ctk.CTkFrame(frame, fg_color="transparent")
        r_frame.pack(fill="x")
        ctk.CTkLabel(r_frame, text="X (eV):", width=40).pack(side="left")
        self.lim_x_min = ctk.CTkEntry(r_frame, width=50); self.lim_x_min.pack(side="left", padx=2)
        self.lim_x_max = ctk.CTkEntry(r_frame, width=50); self.lim_x_max.pack(side="left", padx=2)
        
        r2_frame = ctk.CTkFrame(frame, fg_color="transparent")
        r2_frame.pack(fill="x", pady=5)
        ctk.CTkLabel(r2_frame, text="Y (Int):", width=40).pack(side="left")
        self.lim_y_min = ctk.CTkEntry(r2_frame, width=50); self.lim_y_min.pack(side="left", padx=2)
        self.lim_y_max = ctk.CTkEntry(r2_frame, width=50); self.lim_y_max.pack(side="left", padx=2)

        # 適用ボタン
        ctk.CTkButton(frame, text="Apply Settings", command=self.apply_graph_settings, fg_color="#E07A5F").pack(pady=20, fill="x")

    def _create_main_area(self):
        """右側のグラフ描画エリアを作成"""
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # Matplotlibの図表作成
        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.ax.set_xlabel("Binding Energy (eV)")
        self.ax.set_ylabel("Intensity (a.u.)")
        
        # XPSは左が高エネルギーなので軸反転
        self.ax.invert_xaxis()
        
        # Tkinterに埋め込む
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # ツールバー（ズームとか保存ボタン）
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.main_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ==========================================
    # 操作ロジック (グラフ設定・マウス操作)
    # ==========================================

    def apply_graph_settings(self):
        """[Graph Settings]タブの内容をグラフに反映"""
        try:
            # タイトル・ラベル
            if self.entry_title.get(): self.ax.set_title(self.entry_title.get())
            if self.entry_xlabel.get(): self.ax.set_xlabel(self.entry_xlabel.get())
            if self.entry_ylabel.get(): self.ax.set_ylabel(self.entry_ylabel.get())
            
            # フォントサイズ
            fs_title = int(self.entry_fs_title.get())
            fs_label = int(self.entry_fs_label.get())
            fs_tick = int(self.entry_fs_tick.get())

            self.ax.set_title(self.ax.get_title(), fontsize=fs_title)
            self.ax.set_xlabel(self.ax.get_xlabel(), fontsize=fs_label)
            self.ax.set_ylabel(self.ax.get_ylabel(), fontsize=fs_label)
            self.ax.tick_params(axis='both', which='major', labelsize=fs_tick)

            # 軸範囲 (XPSなのでX軸は大きい順に入るように注意するが、set_xlimは自動対応する)
            # ただし反転状態を維持するために大きい値をleft、小さい値をrightにする
            try:
                x_min = float(self.lim_x_min.get())
                x_max = float(self.lim_x_max.get())
                self.ax.set_xlim(x_max, x_min) # XPS Convention: Max -> Min
            except ValueError: pass

            try:
                y_min = float(self.lim_y_min.get())
                y_max = float(self.lim_y_max.get())
                self.ax.set_ylim(y_min, y_max)
            except ValueError: pass

            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Settings Error", f"設定エラー: {e}")

    def activate_selector(self, mode):
        """マウス範囲選択モードを有効化"""
        self.selection_mode = mode
        
        # 既存のセレクタがあれば消去
        if self.span: self.span.set_visible(False); self.span = None
        
        # モードに応じた色設定 (視認性のため)
        # VBM用: BG=青, Slope=赤
        # BandGap用: Peak=緑, Base=青, Slope=赤
        colors = {
            'bg': 'blue', 'slope': 'red', 
            'bg_peak': 'green', 'bg_base': 'blue', 'bg_slope': 'red'
        }
        color = colors.get(mode, 'gray')
        
        # SpanSelector作成 (グラフ上をドラッグ可能にする)
        self.span = SpanSelector(self.ax, self.on_select, 'horizontal', useblit=True,
                                 props=dict(alpha=0.3, facecolor=color), interactive=True, drag_from_anywhere=True)
        self.canvas.draw()

    def deactivate_selector(self):
        """マウス範囲選択モードを終了（ズーム機能などを復活）"""
        if self.span: self.span.set_visible(False); self.span = None
        self.selection_mode = None
        self.canvas.draw()

    def on_select(self, vmin, vmax):
        """ドラッグ終了時に呼ばれる処理。対応するEntryに入力する"""
        min_val, max_val = sorted([vmin, vmax])
        val_str_min = f"{min_val:.2f}"
        val_str_max = f"{max_val:.2f}"

        # 現在のモードに対応するEntryボックスのペアを取得
        entries = {
            'bg': (self.entry_bg_min, self.entry_bg_max),
            'slope': (self.entry_slope_min, self.entry_slope_max),
            'bg_peak': (self.bg_peak_min, self.bg_peak_max),
            'bg_base': (self.bg_base_min, self.bg_base_max),
            'bg_slope': (self.bg_slope_min, self.bg_slope_max)
        }

        if self.selection_mode in entries:
            emin, emax = entries[self.selection_mode]
            emin.delete(0, tk.END); emin.insert(0, val_str_min)
            emax.delete(0, tk.END); emax.insert(0, val_str_max)

    def on_shirley_toggle(self):
        """Shirleyチェックボックスが切り替えられた時の処理"""
        if self.energy is None: return
        
        if self.chk_shirley_var.get():
            try:
                # 計算実行
                self.bg_data = calculate_shirley_bg(self.energy, self.intensity)
                self.intensity_corrected = self.intensity - self.bg_data
            except Exception as e:
                messagebox.showerror("Error", f"Shirley Calculation Failed:\n{e}")
                self.chk_shirley_var.set(False)
                return
        else:
            # OFFになったらリセット
            self.bg_data = None
            self.intensity_corrected = None
            
        # グラフを再描画して反映
        self.plot_base_graph()

    # ==========================================
    # データ入出力 & 計算ロジック
    # ==========================================

    def load_csv(self):
        """CSVファイルを読み込む"""
        file_path = filedialog.askopenfilename(filetypes=[("Data Files", "*.csv *.txt *.dat"), ("All Files", "*.*")])
        if not file_path: return
        
        # 選択された区切り文字
        sep_map = {", (Comma)": ",", "\\t (Tab)": "\t", "Space": r"\s+"}
        sep = sep_map[self.sep_option.get()]

        try:
            # pandasで読み込み
            self.df = pd.read_csv(file_path, sep=sep, header=None, engine='python')
            if self.df.shape[1] < 2:
                raise ValueError("データ列が足りません（2列以上必要）")

            # 数値変換 (エラー値はNaNに)
            self.energy = pd.to_numeric(self.df.iloc[:, 0], errors='coerce').values
            self.intensity = pd.to_numeric(self.df.iloc[:, 1], errors='coerce').values
            
            # NaNを除去
            mask = ~np.isnan(self.energy) & ~np.isnan(self.intensity)
            self.energy = self.energy[mask]; self.intensity = self.intensity[mask]

            if len(self.energy) == 0: raise ValueError("有効な数値データがありません")

            # --- 初期値の自動入力 ---
            min_e, max_e = np.min(self.energy), np.max(self.energy)
            
            # VBMタブの初期値
            self.entry_bg_min.delete(0, tk.END); self.entry_bg_min.insert(0, f"{min_e:.1f}")
            self.entry_bg_max.delete(0, tk.END); self.entry_bg_max.insert(0, f"{min_e+2.0:.1f}")
            self.entry_slope_min.delete(0, tk.END); self.entry_slope_min.insert(0, f"{min_e+3.0:.1f}")
            self.entry_slope_max.delete(0, tk.END); self.entry_slope_max.insert(0, f"{min_e+5.0:.1f}")
            
            # Band Gapタブの初期値 (全域から適当な幅でセット)
            self.bg_peak_min.delete(0, tk.END); self.bg_peak_min.insert(0, f"{min_e:.1f}")
            self.bg_peak_max.delete(0, tk.END); self.bg_peak_max.insert(0, f"{min_e+1.0:.1f}")
            self.bg_base_min.delete(0, tk.END); self.bg_base_min.insert(0, f"{min_e+10.0:.1f}")
            self.bg_base_max.delete(0, tk.END); self.bg_base_max.insert(0, f"{min_e+12.0:.1f}")
            self.bg_slope_min.delete(0, tk.END); self.bg_slope_min.insert(0, f"{min_e+13.0:.1f}")
            self.bg_slope_max.delete(0, tk.END); self.bg_slope_max.insert(0, f"{min_e+15.0:.1f}")
            
            # Shirley設定リセット
            self.chk_shirley_var.set(False)
            self.intensity_corrected = None
            self.bg_data = None
            
            # Graphタブ設定
            fname = os.path.basename(file_path)
            self.entry_title.delete(0, tk.END); self.entry_title.insert(0, f"XPS: {fname}")
            self.lim_x_min.delete(0, tk.END); self.lim_x_min.insert(0, f"{min_e:.1f}")
            self.lim_x_max.delete(0, tk.END); self.lim_x_max.insert(0, f"{max_e:.1f}")
            self.lim_y_min.delete(0, tk.END); self.lim_y_min.insert(0, f"{np.min(self.intensity):.1f}")
            self.lim_y_max.delete(0, tk.END); self.lim_y_max.insert(0, f"{np.max(self.intensity):.1f}")

            # 描画とボタン有効化
            self.plot_base_graph()
            self.calc_btn.configure(state="normal")
            self.calc_bg_btn.configure(state="normal")
            
        except Exception as e:
            messagebox.showerror("Import Error", str(e))

    def plot_base_graph(self):
        """ベースとなるスペクトルを描画（計算結果なしの状態）"""
        self.ax.clear()
        
        if self.chk_shirley_var.get() and self.intensity_corrected is not None:
            # Shirley ON: 生データ(薄い) + 補正データ(濃い) + BG線
            self.ax.plot(self.energy, self.intensity, color='gray', alpha=0.3, label='Raw Data')
            self.ax.plot(self.energy, self.bg_data, color='gray', linestyle='--', alpha=0.5, label='Shirley BG')
            self.ax.plot(self.energy, self.intensity_corrected, color='#4a90e2', linewidth=1.5, label='Corrected')
        else:
            # Shirley OFF: 生データのみ
            self.ax.plot(self.energy, self.intensity, color='#4a90e2', linewidth=1.5, label='Raw Spectrum')

        self.ax.legend()
        self.ax.grid(True)
        self.ax.invert_xaxis()
        self.apply_graph_settings()
        self.canvas.draw()

    def get_current_intensity(self):
        """現在使用すべき強度データ（Shirley ONなら補正データ）を返す"""
        if self.chk_shirley_var.get() and self.intensity_corrected is not None:
            return self.intensity_corrected
        return self.intensity

    def calculate(self):
        """【Tab 1】VBM解析（直線交点法）を実行"""
        if self.energy is None: return
        try:
            y_data = self.get_current_intensity()
            
            # 入力値取得
            bg_start = float(self.entry_bg_min.get())
            bg_end = float(self.entry_bg_max.get())
            slope_start = float(self.entry_slope_min.get())
            slope_end = float(self.entry_slope_max.get())

            # 1. バックグラウンド範囲のフィッティング
            mask_bg = (self.energy >= bg_start) & (self.energy <= bg_end)
            if not np.any(mask_bg): raise ValueError("Background範囲にデータがありません")
            popt_bg, _ = curve_fit(linear_func, self.energy[mask_bg], y_data[mask_bg])

            # 2. 立ち上がり（Slope）範囲のフィッティング
            mask_slope = (self.energy >= slope_start) & (self.energy <= slope_end)
            if not np.any(mask_slope): raise ValueError("Slope範囲にデータがありません")
            popt_slope, _ = curve_fit(linear_func, self.energy[mask_slope], y_data[mask_slope])

            # 3. 交点計算
            a1, b1 = popt_bg
            a2, b2 = popt_slope
            if abs(a1 - a2) < 1e-10: raise ValueError("平行エラー: 2本の直線が平行です")

            vbm_x = (b2 - b1) / (a1 - a2)
            vbm_y = linear_func(vbm_x, *popt_bg)

            # 結果表示
            self.vbm_label.configure(text=f"VBM: {vbm_x:.3f} eV")

            # --- グラフ更新 ---
            self.plot_base_graph() # ベースを描画
            
            # フィッティング線を描画
            x_range = np.linspace(min(self.energy), max(self.energy), 200)
            self.ax.plot(x_range, linear_func(x_range, *popt_bg), 'b--', alpha=0.8, label='Base Fit')
            self.ax.plot(x_range, linear_func(x_range, *popt_slope), 'r--', alpha=0.8, label='Slope Fit')
            
            # 交点プロット
            self.ax.plot(vbm_x, vbm_y, 'go', markersize=10, zorder=5, label=f'VBM={vbm_x:.2f}eV')
            self.ax.axvline(vbm_x, color='green', linestyle=':', alpha=0.8)
            
            # 範囲の可視化
            self.ax.axvspan(bg_start, bg_end, color='blue', alpha=0.1)
            self.ax.axvspan(slope_start, slope_end, color='red', alpha=0.1)

            self.ax.legend()
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Calc Error", f"解析エラー:\n{e}")

    def calculate_bandgap(self):
        """【Tab 2】バンドギャップ解析 (Linear Fit & 2nd Derivative)"""
        if self.energy is None: return
        try:
            # Shirley ONなら補正データを使用
            if self.chk_shirley_var.get() and self.intensity_corrected is not None:
                y_data = self.intensity_corrected
            else:
                y_data = self.intensity

            # --- 共通: メインピーク位置の特定 ---
            pk_r = (float(self.bg_peak_min.get()), float(self.bg_peak_max.get()))
            mask_pk = (self.energy >= pk_r[0]) & (self.energy <= pk_r[1])
            if not np.any(mask_pk): raise ValueError("Peak範囲データなし")
            
            subset_idx = np.where(mask_pk)[0]
            peak_idx = subset_idx[np.argmax(y_data[subset_idx])]
            peak_x = self.energy[peak_idx]
            peak_y = y_data[peak_idx]

            # ==========================================
            # Method A: 直線外挿法 (Linear Intersection)
            # ==========================================
            base_r = (float(self.bg_base_min.get()), float(self.bg_base_max.get()))
            sl_r = (float(self.bg_slope_min.get()), float(self.bg_slope_max.get()))

            # Base Fit
            mask_base = (self.energy >= base_r[0]) & (self.energy <= base_r[1])
            popt_base, _ = curve_fit(linear_func, self.energy[mask_base], y_data[mask_base])

            # Slope Fit
            mask_sl = (self.energy >= sl_r[0]) & (self.energy <= sl_r[1])
            popt_sl, _ = curve_fit(linear_func, self.energy[mask_sl], y_data[mask_sl])

            # 交点 (Onset)
            onset_linear_x = (popt_sl[1] - popt_base[1]) / (popt_base[0] - popt_sl[0])
            onset_linear_y = linear_func(onset_linear_x, *popt_base)
            gap_linear = abs(onset_linear_x - peak_x)

            # ==========================================
            # Method B: 2次微分法 (BGの傾き変化点)
            # ==========================================
            # Loss領域全体を含む範囲 (Base範囲の端 〜 Slope範囲の端)
            deriv_region_min = min(base_r[0], sl_r[0])
            deriv_region_max = max(base_r[1], sl_r[1])
            
            # 解析用マスク
            mask_d = (self.energy >= deriv_region_min) & (self.energy <= deriv_region_max)
            x_d = self.energy[mask_d]
            y_d = y_data[mask_d]

            if len(x_d) < 10: raise ValueError("微分解析用のデータ点数が少なすぎます")

            # 1. スムージング (Savitzky-Golay filter)
            # window_lengthはデータ点数に合わせて調整 (奇数である必要あり)
            window_len = min(15, len(x_d) - 2)
            if window_len % 2 == 0: window_len -= 1
            y_smooth = savgol_filter(y_d, window_length=window_len, polyorder=3)

            # 2. 2次微分 (np.gradientを2回)
            # xは降順(XPS)かもしれないので、dxの符号に注意
            dy = np.gradient(y_smooth, x_d)
            d2y = np.gradient(dy, x_d)

            # 3. 2次微分の最大値を探す
            # 立ち上がり(傾きの変化が最大) = 2次微分が正の最大値を持つ点
            # ※ ノイズで変なところを拾わないよう、y_dがある程度大きい部分に限定するなどの工夫も可
            max_d2_idx = np.argmax(d2y)
            onset_deriv_x = x_d[max_d2_idx]
            onset_deriv_y = y_d[max_d2_idx] # 元データのY値
            
            gap_deriv = abs(onset_deriv_x - peak_x)

            # ==========================================
            # 結果表示 & 描画
            # ==========================================
            result_text = f"Linear Eg: {gap_linear:.2f} eV\nDeriv Eg: {gap_deriv:.2f} eV"
            self.lbl_res_gap.configure(text=result_text)

            self.plot_base_graph()
            
            # --- Linear Plot ---
            x_plot = np.linspace(min(self.energy), max(self.energy), 200)
            self.ax.plot(peak_x, peak_y, 'g*', markersize=15, zorder=5, label='Main Peak')
            self.ax.axvline(peak_x, color='green', linestyle=':', alpha=0.6)

            self.ax.plot(x_plot, linear_func(x_plot, *popt_base), 'b--', alpha=0.5, label='Linear: Base')
            self.ax.plot(x_plot, linear_func(x_plot, *popt_sl), 'r--', alpha=0.5, label='Linear: Slope')
            self.ax.plot(onset_linear_x, onset_linear_y, 'ro', markersize=8, zorder=5, label='Linear Onset')

            # --- Derivative Plot ---
            self.ax.plot(onset_deriv_x, onset_deriv_y, 'bx', markersize=10, markeredgewidth=3, zorder=6, label='Deriv Onset (Kink)')
            self.ax.axvline(onset_deriv_x, color='orange', linestyle='--', alpha=0.8)

            # 矢印 (今回はLinearの結果に合わせるが、好みで変更可)
            arrow_y = (peak_y + onset_linear_y) / 2
            self.ax.annotate(f'Eg(Lin) = {gap_linear:.2f} eV', xy=(peak_x, arrow_y), xytext=(onset_linear_x, arrow_y),
                             arrowprops=dict(arrowstyle='<->', color='purple', lw=2),
                             ha='center', va='bottom', fontsize=12, color='purple', fontweight='bold')

            # 範囲可視化
            self.ax.axvspan(pk_r[0], pk_r[1], color='green', alpha=0.1)
            self.ax.axvspan(base_r[0], base_r[1], color='blue', alpha=0.1)
            self.ax.axvspan(sl_r[0], sl_r[1], color='red', alpha=0.1)

            self.ax.legend()
            self.canvas.draw()

        except Exception as e:
            messagebox.showerror("Calc Error", f"解析エラー:\n{e}")

    def on_closing(self):
        """アプリ終了時の処理"""
        plt.close('all') # メモリ解放
        self.quit()
        self.destroy()

if __name__ == "__main__":
    app = XPS_VB_Edge_App()
    app.mainloop()