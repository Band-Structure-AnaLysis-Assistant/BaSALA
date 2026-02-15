from PIL import Image
import os

# ▼ここに変換したいPNGファイルの名前を入れてください
input_png = "my_icon.png"
# ▼出力するICOファイルの名前
output_ico = "app_icon.ico"

if not os.path.exists(input_png):
    print(f"エラー: {input_png} が見つかりません。")
    exit()

try:
    img = Image.open(input_png)
    
    # Windowsアイコンに必要な標準的なサイズを用意します
    # (これらを1つのファイルにまとめることで、どこで表示されても綺麗に見えます)
    icon_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    
    # ICO形式で保存
    img.save(output_ico, format='ICO', sizes=icon_sizes)
    print(f"成功: {output_ico} を作成しました。")
    
except Exception as e:
    print(f"変換エラー: {e}")

# 実行方法: ターミナルで `python png_to_ico.py`