import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
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
        self.tab_analysis = self.tabview.add("Analysis")       # VBM解析用
        self.tab_bg = self.tabview.add("Band Gap")             # バンドギャップ解析用
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