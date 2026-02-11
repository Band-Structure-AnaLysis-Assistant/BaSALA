import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector # ★これを追加
from scipy.optimize import curve_fit
import os

# 設定
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

def linear_func(x, a, b):
    return a * x + b

class XPS_VB_Edge_App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("XPS VB Edge Analyzer (Phase 1) - v0.3 (Mouse Control)")
        self.geometry("1200x850")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.file_path = None
        self.df = None
        self.energy = None
        self.intensity = None
        
        # ★ マウス選択用の変数を追加
        self.span = None 
        self.selection_mode = None # "bg" or "slope"

        # UI構築
        self._create_sidebar()
        self._create_main_area()

    def _create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)

        self.logo_label = ctk.CTkLabel(self.sidebar, text="XPS VB Edge\nAnalyzer", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.pack(padx=20, pady=(30, 10))

        # --- Step 1: Import ---
        self.step1_frame = ctk.CTkFrame(self.sidebar)
        self.step1_frame.pack(padx=10, pady=5, fill="x")
        ctk.CTkLabel(self.step1_frame, text="Step 1: Data Import", font=("Roboto", 14, "bold")).pack(pady=5)
        
        self.load_btn = ctk.CTkButton(self.step1_frame, text="Open CSV / Text", command=self.load_csv, fg_color="#1f538d")
        self.load_btn.pack(padx=10, pady=10, fill="x")

        self.sep_option = ctk.CTkComboBox(self.step1_frame, values=[", (Comma)", "\\t (Tab)", "Space", "; (Semicolon)"])
        self.sep_option.set(", (Comma)")
        self.sep_option.pack(padx=10, pady=(0, 10))

        # --- Step 2: Parameter Setting (マウス操作ボタンを追加) ---
        self.step2_frame = ctk.CTkFrame(self.sidebar)
        self.step2_frame.pack(padx=10, pady=15, fill="x")
        ctk.CTkLabel(self.step2_frame, text="Step 2: Fitting Ranges (eV)", font=("Roboto", 14, "bold")).pack(pady=5)

        # ★ Background Range UI
        ctk.CTkLabel(self.step2_frame, text="Background Range:", font=("Roboto", 12)).pack(anchor="w", padx=10)
        self.bg_frame = ctk.CTkFrame(self.step2_frame, fg_color="transparent")
        self.bg_frame.pack(padx=10, pady=0, fill="x")
        
        self.entry_bg_min = ctk.CTkEntry(self.bg_frame, width=60, placeholder_text="Min")
        self.entry_bg_min.pack(side="left", padx=(0, 2))
        ctk.CTkLabel(self.bg_frame, text="-").pack(side="left")
        self.entry_bg_max = ctk.CTkEntry(self.bg_frame, width=60, placeholder_text="Max")
        self.entry_bg_max.pack(side="left", padx=(2, 5))
        
        # 選択ボタン (BG)
        self.btn_sel_bg = ctk.CTkButton(self.bg_frame, text="🖱️Select", width=60, fg_color="gray", command=lambda: self.activate_selector("bg"))
        self.btn_sel_bg.pack(side="right")

        # ★ Slope Range UI
        ctk.CTkLabel(self.step2_frame, text="VB Slope Range:", font=("Roboto", 12)).pack(anchor="w", padx=10, pady=(10, 0))
        self.slope_frame = ctk.CTkFrame(self.step2_frame, fg_color="transparent")
        self.slope_frame.pack(padx=10, pady=0, fill="x")
        
        self.entry_slope_min = ctk.CTkEntry(self.slope_frame, width=60, placeholder_text="Min")
        self.entry_slope_min.pack(side="left", padx=(0, 2))
        ctk.CTkLabel(self.slope_frame, text="-").pack(side="left")
        self.entry_slope_max = ctk.CTkEntry(self.slope_frame, width=60, placeholder_text="Max")
        self.entry_slope_max.pack(side="left", padx=(2, 5))

        # 選択ボタン (Slope)
        self.btn_sel_slope = ctk.CTkButton(self.slope_frame, text="🖱️Select", width=60, fg_color="gray", command=lambda: self.activate_selector("slope"))
        self.btn_sel_slope.pack(side="right")
        
        # モード解除ボタン
        self.btn_reset_mode = ctk.CTkButton(self.step2_frame, text="Stop Selection (Zoom Mode)", fg_color="transparent", border_width=1, command=self.deactivate_selector)
        self.btn_reset_mode.pack(pady=10)


        # --- Step 3: Analysis ---
        self.calc_btn = ctk.CTkButton(self.sidebar, text="Calculate VBM", command=self.calculate, fg_color="#2d8d2d", state="disabled", height=40, font=("Roboto", 16, "bold"))
        self.calc_btn.pack(padx=20, pady=20, fill="x")

        # 結果表示
        self.result_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.result_frame.pack(padx=10, pady=10, fill="x")
        self.vbm_label = ctk.CTkLabel(self.result_frame, text="VBM: --- eV", font=ctk.CTkFont(size=24, weight="bold"), text_color="#4db6ac")
        self.vbm_label.pack()

    def _create_main_area(self):
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.ax.set_title("XPS Spectrum Analysis", fontsize=12)
        self.ax.set_xlabel("Binding Energy (eV)")
        self.ax.set_ylabel("Intensity (a.u.)")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.main_frame)
        self.toolbar.update()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # --- ★ マウス選択ロジック ---
    def activate_selector(self, mode):
        """範囲選択モードを有効化"""
        self.selection_mode = mode
        
        # 既存のセレクタがあれば消す
        if self.span:
            self.span.set_visible(False)
            self.span = None

        # 色の設定 (BG=青, Slope=赤)
        color = 'blue' if mode == 'bg' else 'red'
        
        # ボタンの色を変えて「今選んでますよ」感を出す
        if mode == 'bg':
            self.btn_sel_bg.configure(fg_color="#1f538d") # Active Blue
            self.btn_sel_slope.configure(fg_color="gray")
        else:
            self.btn_sel_bg.configure(fg_color="gray")
            self.btn_sel_slope.configure(fg_color="#c62828") # Active Red

        # SpanSelectorを作成 (onselectコールバックを登録)
        # direction='horizontal': 横方向のみ選択
        # useblit=True: 高速描画
        self.span = SpanSelector(
            self.ax, 
            self.on_select, 
            'horizontal', 
            useblit=True,
            props=dict(alpha=0.3, facecolor=color),
            interactive=True,
            drag_from_anywhere=True
        )
        self.canvas.draw()

    def deactivate_selector(self):
        """選択モードを終了し、ズーム等が使えるようにする"""
        if self.span:
            self.span.set_visible(False)
            self.span = None
        self.selection_mode = None
        self.btn_sel_bg.configure(fg_color="gray")
        self.btn_sel_slope.configure(fg_color="gray")
        self.canvas.draw()

    def on_select(self, vmin, vmax):
        """範囲が選択されたときに呼ばれる関数"""
        # 小さい順に並べる
        min_val, max_val = sorted([vmin, vmax])
        
        # 現在のモードに応じてEntryに入力
        if self.selection_mode == 'bg':
            self.entry_bg_min.delete(0, tk.END); self.entry_bg_min.insert(0, f"{min_val:.2f}")
            self.entry_bg_max.delete(0, tk.END); self.entry_bg_max.insert(0, f"{max_val:.2f}")
        elif self.selection_mode == 'slope':
            self.entry_slope_min.delete(0, tk.END); self.entry_slope_min.insert(0, f"{min_val:.2f}")
            self.entry_slope_max.delete(0, tk.END); self.entry_slope_max.insert(0, f"{max_val:.2f}")

    # ---------------------------

    def load_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("Data Files", "*.csv *.txt *.dat"), ("All Files", "*.*")])
        if not file_path: return

        sep_map = {", (Comma)": ",", "\\t (Tab)": "\t", "Space": r"\s+", "; (Semicolon)": ";"}
        sep = sep_map[self.sep_option.get()]

        try:
            self.df = pd.read_csv(file_path, sep=sep, header=None, engine='python')
            if self.df.shape[1] < 2:
                messagebox.showerror("Format Error", "データ列が足りません。")
                return

            self.energy = pd.to_numeric(self.df.iloc[:, 0], errors='coerce').values
            self.intensity = pd.to_numeric(self.df.iloc[:, 1], errors='coerce').values
            mask = ~np.isnan(self.energy) & ~np.isnan(self.intensity)
            self.energy = self.energy[mask]
            self.intensity = self.intensity[mask]

            if len(self.energy) == 0:
                messagebox.showerror("Data Error", "有効な数値データなし")
                return

            # 初期値
            min_e = np.min(self.energy)
            self.entry_bg_min.delete(0, tk.END); self.entry_bg_min.insert(0, f"{min_e:.1f}")
            self.entry_bg_max.delete(0, tk.END); self.entry_bg_max.insert(0, f"{min_e+2.0:.1f}")
            self.entry_slope_min.delete(0, tk.END); self.entry_slope_min.insert(0, f"{min_e+3.0:.1f}")
            self.entry_slope_max.delete(0, tk.END); self.entry_slope_max.insert(0, f"{min_e+5.0:.1f}")

            self.ax.clear()
            self.ax.plot(self.energy, self.intensity, color='#4a90e2', linewidth=1.5, label='Raw Spectrum')
            self.ax.set_title(f"Loaded: {os.path.basename(file_path)}")
            self.ax.legend()
            self.ax.grid(True)
            self.canvas.draw()
            
            self.calc_btn.configure(state="normal")
            self.file_path = file_path
            
        except Exception as e:
            messagebox.showerror("Import Error", f"読み込み失敗:\n{e}")

    def calculate(self):
        if self.energy is None: return
        try:
            bg_start = float(self.entry_bg_min.get())
            bg_end = float(self.entry_bg_max.get())
            slope_start = float(self.entry_slope_min.get())
            slope_end = float(self.entry_slope_max.get())

            mask_bg = (self.energy >= bg_start) & (self.energy <= bg_end)
            if not np.any(mask_bg): raise ValueError("Background範囲エラー")
            popt_bg, _ = curve_fit(linear_func, self.energy[mask_bg], self.intensity[mask_bg])

            mask_slope = (self.energy >= slope_start) & (self.energy <= slope_end)
            if not np.any(mask_slope): raise ValueError("Slope範囲エラー")
            popt_slope, _ = curve_fit(linear_func, self.energy[mask_slope], self.intensity[mask_slope])

            a1, b1 = popt_bg
            a2, b2 = popt_slope
            if abs(a1 - a2) < 1e-10: raise ValueError("平行エラー")

            vbm_x = (b2 - b1) / (a1 - a2)
            vbm_y = linear_func(vbm_x, *popt_bg)

            self.vbm_label.configure(text=f"VBM: {vbm_x:.3f} eV")

            self.ax.clear()
            self.ax.plot(self.energy, self.intensity, color='gray', alpha=0.4, label='Raw Data')
            
            x_plot = np.linspace(min(self.energy), max(self.energy), 200)
            self.ax.plot(x_plot, linear_func(x_plot, *popt_bg), 'b--', alpha=0.8, label='Baseline Fit')
            self.ax.plot(x_plot, linear_func(x_plot, *popt_slope), 'r--', alpha=0.8, label='Slope Fit')
            self.ax.plot(vbm_x, vbm_y, 'go', markersize=10, zorder=5, label=f'VBM = {vbm_x:.2f} eV')
            self.ax.axvline(vbm_x, color='green', linestyle=':', alpha=0.8)

            self.ax.axvspan(bg_start, bg_end, color='blue', alpha=0.1)
            self.ax.axvspan(slope_start, slope_end, color='red', alpha=0.1)

            self.ax.set_title(f"Result: VBM = {vbm_x:.3f} eV")
            self.ax.legend()
            self.ax.grid(True)
            
            # 再描画時にSpanSelectorを消さないように注意する
            if self.span: self.span.set_visible(True) # 再表示
            self.canvas.draw()
            
        except ValueError as ve:
             messagebox.showwarning("Range Error", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"解析エラー:\n{e}")

    def on_closing(self):
        plt.close('all')
        self.quit()
        self.destroy()

if __name__ == "__main__":
    app = XPS_VB_Edge_App()
    app.mainloop()