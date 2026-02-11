import numpy as np
import pandas as pd
import os

# 保存用フォルダを作成
output_dir = "dummy_data"
os.makedirs(output_dir, exist_ok=True)

def create_vb_spectrum(filename, edge_pos, noise_level, slope_intensity, bg_slope):
    """
    XPSのVBスペクトルっぽいデータを作成する関数
    - edge_pos: VBの立ち上がり位置 (eV)
    - noise_level: ノイズの大きさ
    - slope_intensity: VBの立ち上がりの急峻さ
    - bg_slope: バックグラウンドの傾き
    """
    # エネルギー範囲 (-2eV ～ 14eV)
    energy = np.linspace(-2, 14, 800)
    
    # 1. バックグラウンド (Shirleyっぽく高エネルギー側が高い + 線形)
    # 低エネルギー側(Bg)から高エネルギー側にかけて少し上がる
    bg = 500 + bg_slope * energy 
    
    # 2. Valence Bandの信号 (Edgeから立ち上がり、DOSっぽい凸凹を追加)
    signal = np.zeros_like(energy)
    
    # Edgeよりエネルギーが高い部分に信号を乗せる
    mask = energy > edge_pos
    
    # 立ち上がり(直線) + DOSの構造(サイン波の二乗で擬似的に再現)
    dos_structure = np.sin((energy[mask] - edge_pos) * 1.5)**2 * 0.3 # バンド内の構造
    
    signal[mask] = slope_intensity * (energy[mask] - edge_pos) * (1 + dos_structure)
    
    # 高エネルギー側で信号が飽和・減衰する処理（VBの底）
    # 10eV以上でなだらかにする
    cutoff_mask = energy > 10
    signal[cutoff_mask] = signal[cutoff_mask] * np.exp(-(energy[cutoff_mask]-10)*0.5)

    # 3. 合成 + ノイズ付加
    intensity = bg + signal + np.random.normal(0, noise_level, len(energy))
    
    # 4. CSV保存 (ヘッダーなし: 1列目Energy, 2列目Intensity)
    df = pd.DataFrame({"Energy": energy, "Intensity": intensity})
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False, header=False)
    print(f"作成完了: {path}")

# --- データ生成実行 ---

# パターン1: 標準的な半導体 (きれいなデータ)
# Edge: 2.5eV, ノイズ少なめ
create_vb_spectrum(
    filename="sample1_standard.csv", 
    edge_pos=2.5, 
    noise_level=15, 
    slope_intensity=800, 
    bg_slope=10
)

# パターン2: ノイズが多い測定データ (解析が難しいやつ)
# Edge: 3.2eV, ノイズ大, 信号弱め
create_vb_spectrum(
    filename="sample2_noisy.csv", 
    edge_pos=3.2, 
    noise_level=80,      # ノイズ激増
    slope_intensity=300, # 信号弱い
    bg_slope=50
)

# パターン3: チャージアップでシフトしたデータ (帯電)
# Edge: 6.8eV (高エネルギー側にズレている), BGが傾いている
create_vb_spectrum(
    filename="sample3_charged.csv", 
    edge_pos=6.8, 
    noise_level=30, 
    slope_intensity=600, 
    bg_slope=5
)

print("\n=== 完了 ===")
print(f"'{output_dir}' フォルダを確認してください。")