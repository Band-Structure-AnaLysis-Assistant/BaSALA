import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import SpanSelector
from scipy.optimize import curve_fit
from scipy.integrate import simpson
import os

# 設定
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

def linear_func(x, a, b):
    return a * x + b

def calculate_shirley_bg(x, y, tol=1e-5, max_iters=50):
    """
    Shirleyバックグラウンドを計算する関数
    x: Binding Energy (これを使ってソートする)
    y: Intensity
    """
    # 1. データのソート (計算のため低エネルギー -> 高エネルギーの順にする)
    # XPSは結合エネルギーが大きいほうが左(散乱成分多い)なので、
    # 積分は「結合エネルギーが低いほう(右)」から「高いほう(左)」へ累積する
    
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    n = len(y)
    bg = np.zeros(n)
    
    # 両端の強度
    I_start = y_sorted[0]  # Low BE side (Right) -> Background is low
    I_end = y_sorted[-1]   # High BE side (Left) -> Background is high
    
    # 初期推定: 直線
    # bg = I_start + (I_end - I_start) * (x_sorted - x_sorted[0]) / (x_sorted[-1] - x_sorted[0])
    
    # 反復計算
    # B(E) = I_start + k * int_{E_min}^{E} (I(E') - I_start) dE'
    # 積分範囲は 低BE(右) から 対象エネルギー まで
    
    # 全体の面積 (I(E) - I_start)
    total_area = np.trapz(y_sorted - I_start, x_sorted)
    
    for _ in range(max_iters):
        bg_new = np.zeros(n)
        
        # 累積積分 (Cumulative Trapezoidal Integration)
        # 各点までの面積を計算
        # cumtrapz的な処理
        # I(E) - I_start ではなく、I(E) - B_old(E) を使うのが本来だが
        # 簡易実装として I(E) - I_start の割合で I_diff を配分する形が一般的
        
        diff_array = y_sorted - I_start
        # 負の値はクリップ
        diff_array[diff_array < 0] = 0
        
        cum_area = np.zeros(n)
        cum_area[1:] = np.cumsum((diff_array[:-1] + diff_array[1:]) / 2 * np.diff(x_sorted))
        
        # バックグラウンドの形状更新
        # 右端(I_start) + (左端と右端の差) * (その地点までの累積面積 / 全面積)
        if total_area == 0:
            k = 0
        else:
            k = (I_end - I_start) / cum_area[-1]
            
        bg_new = I_start + k * cum_area
        
        # 収束判定
        if np.max(np.abs(bg_new - bg)) < tol:
            bg = bg_new
            break
            
        bg = bg_new

    # 元の順序に戻す
    bg_original_order = np.zeros(n)
    bg_original_order[sorted_indices] = bg
    
    return bg_original_order

class XPS_VB_Edge_App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("XPS VB Edge Analyzer (Phase 1) - v0.6 (Shirley BG)")
        self.geometry("1200x900")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # データ変数
        self.file_path = None
        self.df = None
        self.energy = None
        self.intensity = None
        self.intensity_corrected = None # Shirley適用後のデータ
        self.bg_data = None # 計算されたShirleyカーブ
        
        # ツール変数
        self.span = None 
        self.selection_mode = None

        # UI構築
        self._create_sidebar()
        self._create_main_area()

    def _create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=320, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)

        # ロゴ
        self.logo_label = ctk.CTkLabel(self.sidebar, text="XPS Analyzer\nv0.6", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.pack(padx=20, pady=(20, 10))

        # ★ タブビュー
        self.tabview = ctk.CTkTabview(self.sidebar, width=300)
        self.tabview.pack(padx=10, pady=10, fill="both", expand=True)
        
        self.tab_analysis = self.tabview.add("Analysis")
        self.tab_graph = self.tabview.add("Graph Settings")

        # ==========================================
        # Tab 1: Analysis
        # ==========================================
        
        # Step 1: Import
        self.step1_frame = ctk.CTkFrame(self.tab_analysis)
        self.step1_frame.pack(padx=5, pady=5, fill="x")
        ctk.CTkLabel(self.step1_frame, text="1. Data Import", font=("Roboto", 13, "bold")).pack(pady=2)
        
        self.load_btn = ctk.CTkButton(self.step1_frame, text="Open CSV", command=self.load_csv, fg_color="#1f538d")
        self.load_btn.pack(padx=5, pady=5, fill="x")
        
        self.sep_option = ctk.CTkComboBox(self.step1_frame, values=[", (Comma)", "\\t (Tab)", "Space"], height=24)
        self.sep_option.set(", (Comma)")
        self.sep_option.pack(padx=5, pady=(0, 5))

        # ★ Shirley Checkbox
        self.chk_shirley_var = ctk.BooleanVar(value=False)
        self.chk_shirley = ctk.CTkCheckBox(self.tab_analysis, text="Apply Shirley BG Subtraction", variable=self.chk_shirley_var, command=self.on_shirley_toggle)
        self.chk_shirley.pack(padx=10, pady=(15, 5), anchor="w")

        # Step 2: Ranges
        self.step2_frame = ctk.CTkFrame(self.tab_analysis)
        self.step2_frame.pack(padx=5, pady=10, fill="x")
        ctk.CTkLabel(self.step2_frame, text="2. Fitting Ranges (eV)", font=("Roboto", 13, "bold")).pack(pady=2)

        # BG Range
        ctk.CTkLabel(self.step2_frame, text="Background (Baseline):", font=("Roboto", 11)).pack(anchor="w", padx=5)
        self.bg_frame = ctk.CTkFrame(self.step2_frame, fg_color="transparent")
        self.bg_frame.pack(fill="x", padx=5)
        self.entry_bg_min = ctk.CTkEntry(self.bg_frame, width=50); self.entry_bg_min.pack(side="left")
        ctk.CTkLabel(self.bg_frame, text="-").pack(side="left")
        self.entry_bg_max = ctk.CTkEntry(self.bg_frame, width=50); self.entry_bg_max.pack(side="left")
        self.btn_sel_bg = ctk.CTkButton(self.bg_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("bg"))
        self.btn_sel_bg.pack(side="right")

        # Slope Range
        ctk.CTkLabel(self.step2_frame, text="VB Slope:", font=("Roboto", 11)).pack(anchor="w", padx=5, pady=(5,0))
        self.slope_frame = ctk.CTkFrame(self.step2_frame, fg_color="transparent")
        self.slope_frame.pack(fill="x", padx=5)
        self.entry_slope_min = ctk.CTkEntry(self.slope_frame, width=50); self.entry_slope_min.pack(side="left")
        ctk.CTkLabel(self.slope_frame, text="-").pack(side="left")
        self.entry_slope_max = ctk.CTkEntry(self.slope_frame, width=50); self.entry_slope_max.pack(side="left")
        self.btn_sel_slope = ctk.CTkButton(self.slope_frame, text="Select", width=50, fg_color="gray", command=lambda: self.activate_selector("slope"))
        self.btn_sel_slope.pack(side="right")

        self.btn_reset_mode = ctk.CTkButton(self.step2_frame, text="Stop Selection", fg_color="transparent", border_width=1, height=24, command=self.deactivate_selector)
        self.btn_reset_mode.pack(pady=10)

        # Calculate
        self.calc_btn = ctk.CTkButton(self.tab_analysis, text="Calculate VBM", command=self.calculate, fg_color="#2d8d2d", state="disabled", font=("Roboto", 14, "bold"))
        self.calc_btn.pack(padx=10, pady=15, fill="x")

        # Result
        self.vbm_label = ctk.CTkLabel(self.tab_analysis, text="VBM: --- eV", font=ctk.CTkFont(size=20, weight="bold"), text_color="#4db6ac")
        self.vbm_label.pack(pady=5)

        # ==========================================
        # Tab 2: Graph Settings
        # ==========================================
        self.grp_labels = ctk.CTkFrame(self.tab_graph)
        self.grp_labels.pack(padx=5, pady=5, fill="x")
        ctk.CTkLabel(self.grp_labels, text="Labels & Title", font=("Roboto", 12, "bold")).pack(pady=2)
        
        self.entry_title = ctk.CTkEntry(self.grp_labels, placeholder_text="Graph Title"); self.entry_title.pack(fill="x", padx=5, pady=2)
        self.entry_xlabel = ctk.CTkEntry(self.grp_labels, placeholder_text="X Label"); self.entry_xlabel.pack(fill="x", padx=5, pady=2)
        self.entry_ylabel = ctk.CTkEntry(self.grp_labels, placeholder_text="Y Label"); self.entry_ylabel.pack(fill="x", padx=5, pady=2)
        
        self.grp_font = ctk.CTkFrame(self.tab_graph)
        self.grp_font.pack(padx=5, pady=5, fill="x")
        ctk.CTkLabel(self.grp_font, text="Font Sizes", font=("Roboto", 12, "bold")).pack(pady=2)

        self.font_frame = ctk.CTkFrame(self.grp_font, fg_color="transparent")
        self.font_frame.pack(fill="x")
        
        ctk.CTkLabel(self.font_frame, text="Title:").grid(row=0, column=0, padx=5)
        self.entry_fs_title = ctk.CTkEntry(self.font_frame, width=40); self.entry_fs_title.grid(row=0, column=1)
        self.entry_fs_title.insert(0, "14")
        
        ctk.CTkLabel(self.font_frame, text="Label:").grid(row=0, column=2, padx=5)
        self.entry_fs_label = ctk.CTkEntry(self.font_frame, width=40); self.entry_fs_label.grid(row=0, column=3)
        self.entry_fs_label.insert(0, "12")
        
        ctk.CTkLabel(self.font_frame, text="Tick:").grid(row=1, column=0, padx=5, pady=5)
        self.entry_fs_tick = ctk.CTkEntry(self.font_frame, width=40); self.entry_fs_tick.grid(row=1, column=1, pady=5)
        self.entry_fs_tick.insert(0, "10")

        self.grp_range = ctk.CTkFrame(self.tab_graph)
        self.grp_range.pack(padx=5, pady=5, fill="x")
        ctk.CTkLabel(self.grp_range, text="Plot Range (Min / Max)", font=("Roboto", 12, "bold")).pack(pady=2)
        
        self.range_x_frame = ctk.CTkFrame(self.grp_range, fg_color="transparent")
        self.range_x_frame.pack(fill="x", padx=5)
        ctk.CTkLabel(self.range_x_frame, text="X (eV):", width=40).pack(side="left")
        self.entry_xlim_min = ctk.CTkEntry(self.range_x_frame, width=50); self.entry_xlim_min.pack(side="left", padx=2)
        self.entry_xlim_max = ctk.CTkEntry(self.range_x_frame, width=50); self.entry_xlim_max.pack(side="left", padx=2)

        self.range_y_frame = ctk.CTkFrame(self.grp_range, fg_color="transparent")
        self.range_y_frame.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(self.range_y_frame, text="Y (Int):", width=40).pack(side="left")
        self.entry_ylim_min = ctk.CTkEntry(self.range_y_frame, width=50); self.entry_ylim_min.pack(side="left", padx=2)
        self.entry_ylim_max = ctk.CTkEntry(self.range_y_frame, width=50); self.entry_ylim_max.pack(side="left", padx=2)

        self.btn_apply = ctk.CTkButton(self.tab_graph, text="Apply Settings", command=self.apply_graph_settings, fg_color="#E07A5F")
        self.btn_apply.pack(padx=10, pady=15, fill="x")

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
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def apply_graph_settings(self):
        try:
            title = self.entry_title.get()
            xlabel = self.entry_xlabel.get()
            ylabel = self.entry_ylabel.get()
            
            if title: self.ax.set_title(title)
            if xlabel: self.ax.set_xlabel(xlabel)
            if ylabel: self.ax.set_ylabel(ylabel)

            fs_title = int(self.entry_fs_title.get())
            fs_label = int(self.entry_fs_label.get())
            fs_tick = int(self.entry_fs_tick.get())

            self.ax.set_title(self.ax.get_title(), fontsize=fs_title)
            self.ax.set_xlabel(self.ax.get_xlabel(), fontsize=fs_label)
            self.ax.set_ylabel(self.ax.get_ylabel(), fontsize=fs_label)
            self.ax.tick_params(axis='both', which='major', labelsize=fs_tick)

            try:
                x_min = float(self.entry_xlim_min.get())
                x_max = float(self.entry_xlim_max.get())
                self.ax.set_xlim(x_max, x_min) 
            except ValueError: pass

            try:
                y_min = float(self.entry_ylim_min.get())
                y_max = float(self.entry_ylim_max.get())
                self.ax.set_ylim(y_min, y_max)
            except ValueError: pass

            self.canvas.draw()
        except Exception as e:
            messagebox.showerror("Settings Error", f"{e}")

    def activate_selector(self, mode):
        self.selection_mode = mode
        if self.span: self.span.set_visible(False); self.span = None
        color = 'blue' if mode == 'bg' else 'red'
        
        if mode == 'bg':
            self.btn_sel_bg.configure(fg_color="#1f538d"); self.btn_sel_slope.configure(fg_color="gray")
        else:
            self.btn_sel_bg.configure(fg_color="gray"); self.btn_sel_slope.configure(fg_color="#c62828")

        self.span = SpanSelector(self.ax, self.on_select, 'horizontal', useblit=True,
                                 props=dict(alpha=0.3, facecolor=color), interactive=True, drag_from_anywhere=True)
        self.canvas.draw()

    def deactivate_selector(self):
        if self.span: self.span.set_visible(False); self.span = None
        self.selection_mode = None
        self.btn_sel_bg.configure(fg_color="gray"); self.btn_sel_slope.configure(fg_color="gray")
        self.canvas.draw()

    def on_select(self, vmin, vmax):
        min_val, max_val = sorted([vmin, vmax])
        if self.selection_mode == 'bg':
            self.entry_bg_min.delete(0, tk.END); self.entry_bg_min.insert(0, f"{min_val:.2f}")
            self.entry_bg_max.delete(0, tk.END); self.entry_bg_max.insert(0, f"{max_val:.2f}")
        elif self.selection_mode == 'slope':
            self.entry_slope_min.delete(0, tk.END); self.entry_slope_min.insert(0, f"{min_val:.2f}")
            self.entry_slope_max.delete(0, tk.END); self.entry_slope_max.insert(0, f"{max_val:.2f}")

    def on_shirley_toggle(self):
        """Shirleyチェックボックスが押されたら再描画"""
        if self.energy is None: return
        
        # チェックがついたら計算して表示
        if self.chk_shirley_var.get():
            try:
                self.bg_data = calculate_shirley_bg(self.energy, self.intensity)
                self.intensity_corrected = self.intensity - self.bg_data
                
                # グラフ更新
                self.ax.clear()
                # 生データは薄く
                self.ax.plot(self.energy, self.intensity, color='gray', alpha=0.3, label='Raw Data')
                # Shirley BG
                self.ax.plot(self.energy, self.bg_data, color='gray', linestyle='--', alpha=0.5, label='Shirley BG')
                # 補正後データ (メイン)
                self.ax.plot(self.energy, self.intensity_corrected, color='#4a90e2', linewidth=1.5, label='Shirley Corrected')
                
                self.ax.legend()
                self.ax.grid(True)
                self.ax.invert_xaxis()
                self.apply_graph_settings()
                self.canvas.draw()
            except Exception as e:
                messagebox.showerror("Error", f"Shirley Calculation Failed:\n{e}")
                self.chk_shirley_var.set(False)
        else:
            # 元に戻す
            self.load_csv_redraw_only()

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

            # Reset params
            min_e = np.min(self.energy)
            self.entry_bg_min.delete(0, tk.END); self.entry_bg_min.insert(0, f"{min_e:.1f}")
            self.entry_bg_max.delete(0, tk.END); self.entry_bg_max.insert(0, f"{min_e+2.0:.1f}")
            self.entry_slope_min.delete(0, tk.END); self.entry_slope_min.insert(0, f"{min_e+3.0:.1f}")
            self.entry_slope_max.delete(0, tk.END); self.entry_slope_max.insert(0, f"{min_e+5.0:.1f}")
            
            # Reset Shirley
            self.chk_shirley_var.set(False)
            self.intensity_corrected = None
            self.bg_data = None
            
            # Graph Info
            fname = os.path.basename(file_path)
            self.entry_title.delete(0, tk.END); self.entry_title.insert(0, f"XPS: {fname}")
            
            # Ranges
            self.entry_xlim_min.delete(0, tk.END); self.entry_xlim_min.insert(0, f"{np.min(self.energy):.1f}")
            self.entry_xlim_max.delete(0, tk.END); self.entry_xlim_max.insert(0, f"{np.max(self.energy):.1f}")
            self.entry_ylim_min.delete(0, tk.END); self.entry_ylim_min.insert(0, f"{np.min(self.intensity):.1f}")
            self.entry_ylim_max.delete(0, tk.END); self.entry_ylim_max.insert(0, f"{np.max(self.intensity):.1f}")

            self.load_csv_redraw_only()
            self.calc_btn.configure(state="normal")
            
        except Exception as e:
            messagebox.showerror("Error", f"{e}")

    def load_csv_redraw_only(self):
        self.ax.clear()
        self.ax.plot(self.energy, self.intensity, color='#4a90e2', linewidth=1.5, label='Raw Spectrum')
        self.ax.legend()
        self.ax.grid(True)
        self.ax.invert_xaxis()
        self.apply_graph_settings()
        self.canvas.draw()

    def calculate(self):
        if self.energy is None: return
        try:
            bg_start = float(self.entry_bg_min.get())
            bg_end = float(self.entry_bg_max.get())
            slope_start = float(self.entry_slope_min.get())
            slope_end = float(self.entry_slope_max.get())

            # ★ ShirleyがONなら補正後データを使う
            if self.chk_shirley_var.get() and self.intensity_corrected is not None:
                target_intensity = self.intensity_corrected
                plot_color = '#4a90e2' # 青
            else:
                target_intensity = self.intensity
                plot_color = '#4a90e2' # 青

            # Fitting
            mask_bg = (self.energy >= bg_start) & (self.energy <= bg_end)
            if not np.any(mask_bg): raise ValueError("No BG data")
            popt_bg, _ = curve_fit(linear_func, self.energy[mask_bg], target_intensity[mask_bg])

            mask_slope = (self.energy >= slope_start) & (self.energy <= slope_end)
            if not np.any(mask_slope): raise ValueError("No Slope data")
            popt_slope, _ = curve_fit(linear_func, self.energy[mask_slope], target_intensity[mask_slope])

            a1, b1 = popt_bg
            a2, b2 = popt_slope
            if abs(a1 - a2) < 1e-10: raise ValueError("Parallel lines")

            vbm_x = (b2 - b1) / (a1 - a2)
            vbm_y = linear_func(vbm_x, *popt_bg)

            self.vbm_label.configure(text=f"VBM: {vbm_x:.3f} eV")

            # Plot Result
            self.ax.clear()
            
            # Shirley ON/OFFに応じたベースプロット
            if self.chk_shirley_var.get():
                self.ax.plot(self.energy, self.intensity, color='gray', alpha=0.3, label='Raw Data')
                self.ax.plot(self.energy, self.bg_data, color='gray', linestyle='--', alpha=0.5, label='Shirley BG')
                self.ax.plot(self.energy, target_intensity, color=plot_color, linewidth=1.5, label='Corrected')
            else:
                self.ax.plot(self.energy, self.intensity, color='gray', alpha=0.4, label='Raw Data')

            x_range = np.linspace(min(self.energy), max(self.energy), 200)
            self.ax.plot(x_range, linear_func(x_range, *popt_bg), 'b--', alpha=0.8, label='Baseline Fit')
            self.ax.plot(x_range, linear_func(x_range, *popt_slope), 'r--', alpha=0.8, label='Slope Fit')
            self.ax.plot(vbm_x, vbm_y, 'go', markersize=10, zorder=5, label=f'VBM={vbm_x:.2f}eV')
            self.ax.axvline(vbm_x, color='green', linestyle=':', alpha=0.8)
            
            self.ax.axvspan(bg_start, bg_end, color='blue', alpha=0.1)
            self.ax.axvspan(slope_start, slope_end, color='red', alpha=0.1)

            self.ax.legend()
            self.ax.grid(True)
            self.ax.invert_xaxis()
            self.apply_graph_settings()

            if self.span: self.span.set_visible(True)
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Calc Error", f"{e}")

    def on_closing(self):
        plt.close('all')
        self.quit()
        self.destroy()

if __name__ == "__main__":
    app = XPS_VB_Edge_App()
    app.mainloop()