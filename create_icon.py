import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

def create_basala_icon():
    # 1. 設定
    icon_size = 512
    dpi = 100
    fig = plt.figure(figsize=(icon_size/dpi, icon_size/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    # 2. 背景（サイエンス・ブルーのグラデーション）
    # Deep Blue (#1f538d) -> Bright Blue (#4a90e2)
    colors = ["#1f538d", "#4a90e2"]
    cmap = LinearSegmentedColormap.from_list("science_blue", colors)
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, 0, 1])

    # 3. "B" のタイポグラフィ（スペクトル風）
    
    # 縦棒 (Axis)
    rect = Rectangle((0.25, 0.15), 0.12, 0.7, color='white', alpha=0.9)
    ax.add_patch(rect)

    # 曲線部分 (Gaussian Peaks)
    x = np.linspace(0.35, 0.85, 100)
    
    # 上の膨らみ (Peak 1)
    y1 = 0.65 + 0.15 * np.exp(-(x - 0.55)**2 / (2 * 0.1**2))
    # 下の膨らみ (Peak 2 - 少し大きく)
    y2 = 0.35 + 0.18 * np.exp(-(x - 0.6)**2 / (2 * 0.12**2))

    # 線として描画
    ax.plot(x, y1, color='white', linewidth=15, solid_capstyle='round')
    ax.plot(x, y2, color='white', linewidth=15, solid_capstyle='round')
    
    # アクセント：スペクトルデータのような点を打つ（解析ツール感を出す）
    x_dots = np.linspace(0.45, 0.85, 15)
    y_dots = 0.35 + 0.18 * np.exp(-(x_dots - 0.6)**2 / (2 * 0.12**2))
    ax.plot(x_dots, y_dots, 'o', color='#4db6ac', markersize=6, alpha=0.8) # 緑色のドット

    # 4. 保存
    png_filename = "BaSALA_icon.png"
    plt.savefig(png_filename, dpi=dpi)
    plt.close()
    
    print(f"Generated: {png_filename}")

    # 5. .ico ファイルへの変換 (exe化に使用するため)
    try:
        img = Image.open(png_filename)
        ico_filename = "BaSALA_icon.ico"
        img.save(ico_filename, format='ICO', sizes=[(256, 256)])
        print(f"Generated: {ico_filename}")
    except Exception as e:
        print(f"ICO conversion failed: {e}")
        print("Please install Pillow: pip install Pillow")

if __name__ == "__main__":
    create_basala_icon()