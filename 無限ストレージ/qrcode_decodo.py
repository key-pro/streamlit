import cv2
import os
from datetime import datetime
import numpy as np
from pathlib import Path

# QRコード検出器の初期化
qr_detector = cv2.QRCodeDetector()

# framesディレクトリのパス
frames_dir = Path("frames")

# 出力ファイル
output_file = "qr_data.txt"

# 全ファイル数
total_files = len(list(frames_dir.glob("*.jpg")))
processed_files = 0

print(f"処理を開始します。合計{total_files}個の画像を処理します。")

# テキストファイルをクリア
with open(output_file, "w", encoding="utf-8") as f:
    f.write("")

# 各画像ファイルを処理
for img_path in sorted(frames_dir.glob("*.jpg")):
    # 画像を読み込む
    image = cv2.imread(str(img_path))
    
    # QRコードをデコード
    retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(image)
    
    if retval:
        for qr_data in decoded_info:
            if qr_data:
                # 現在の日時を取得
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # テキストファイルに保存
                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(f"{current_time}: {qr_data}\n")
    
    # 進捗を表示
    processed_files += 1
    progress = (processed_files / total_files) * 100
    print(f"進捗: {progress:.1f}% ({processed_files}/{total_files})")

print(f"処理が完了しました！結果を{output_file}に保存しました。") 