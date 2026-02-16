BaSALA - Band Gap (Eg) & VBM Analysis Tool
Version: 1.0.0
Release Date: 2026-02-16
Developer: Yuta Okuyama
============================================================

Thank you for downloading BaSALA.
BaSALA is a portable desktop application designed to determine the Band Gap (Eg) and Valence Band Maximum (VBM) from XPS (X-ray Photoelectron Spectroscopy) spectra.

[System Requirements]
OS: Windows 10 / 11 (64-bit)

[Installation]
No installation is required.
1. Unzip the downloaded file to any folder (e.g., Desktop, Documents).
2. Run "BaSALA_v1.0.0.exe" directly.
   (Note: You can carry this folder on a USB stick and run it on any Windows PC.)

[How to Import Data]
BaSALA supports standard CSV and TXT files.
- Column 1: Binding Energy (eV)
- Column 2: Intensity (Counts)

* For ULVAC-PHI MultiPak Users:
  You can directly import CSV files exported from MultiPak.
  (File > Export > Select "ASCII" > Format: Comma Separated)

[Key Features]
- Band Gap Determination: Linear, Derivative, and Hybrid fitting modes.
- VBM Determination: Accurate extraction using Linear and Derivative methods.
- Background Subtraction: Iterative Shirley algorithm (Proctor & Sherwood, 1982).

[Citations]
If you use this software for your research, please cite the following algorithm for background subtraction:
- A. Proctor and P. M. A. Sherwood, "Data analysis techniques in x-ray photoelectron spectroscopy", Anal. Chem. 54, 13 (1982).

[Disclaimer]
This software is provided "as is", without warranty of any kind. The developer is not responsible for any damages or incorrect results arising from the use of this software. Please verify your results with standard analysis methods.

[Contact / Feedback]
If you have any questions or bug reports, please contact:
Email: basala.official@outlook.com

============================================================

BaSALA - バンドギャップおよびVBM解析アシスタント
バージョン: 1.0.0
リリース日：2026-02-16
開発者：奥山 優太

この度はBaSALAをダウンロードいただきありがとうございます。
本ソフトは、XPSスペクトルからバンドギャップ（Eg）および価電子帯上端（VBM）を効率的に決定するためのポータブルツールです。

【動作環境】
Windows 10 / 11 (64-bit)

【インストール・アンインストール】
インストールは不要です。
1. ダウンロードしたZIPファイルを解凍し、任意の場所（デスクトップ等）に配置してください。
2. フォルダ内の「BaSALA_v1.0.0.exe」をダブルクリックして起動します。
   ※USBメモリに入れて持ち運ぶことも可能です。
3. 削除する際は、フォルダごとゴミ箱に入れてください（レジストリは使用していません）。

【データの読み込みについて】
一般的なCSVおよびTXTファイル（1列目：エネルギー、2列目：強度）に対応しています。

※ ULVAC-PHI MultiPakをお使いの方へ:
MultiPakからエクスポートしたCSVファイルを、整形なしでそのまま読み込むことができます。
（MultiPakのメニューから File > Export > "ASCII" を選択し、区切り文字をカンマに設定して保存してください）

【主な機能】
- バンドギャップ解析: Energy Loss信号の立ち上がり解析（Linear / Derivative / Hybrid モード搭載）
- VBM決定: 接線法および微分法による正確なVBM特定
- 背景除去: 反復Shirley法（Proctor & Sherwood, 1982）によるバックグラウンド補正

【引用について】
本ソフトのShirley法BG除去機能を使用した解析結果を論文等に掲載する場合は、以下の原著論文の引用を推奨します。
- A. Proctor and P. M. A. Sherwood, "Data analysis techniques in x-ray photoelectron spectroscopy", Anal. Chem. 54, 13 (1982).

【免責事項】
本ソフトウェアは現状のままで提供され、開発者はその使用から生じるいかなる損害や不利益（計算結果の誤りを含む）についても責任を負いません。研究用途での使用においては、ユーザー自身の責任において結果の検証を行ってください。

【お問い合わせ】
不具合報告や機能要望は、以下のメールアドレスまでお願いいたします。
Email: basala.official@outlook.com

Copyright (C) 2026 [Yuta Okuyama]. All Rights Reserved.