import cv2
import os
import glob
from pathlib import Path

def create_video_from_images(image_folder, output_video, fps=30):
    
    # 画像ファイルのパスを取得（数字順にソート）
    images = sorted(glob.glob(os.path.join(image_folder, "chunk_*.png")), 
                   key=lambda x: int(Path(x).stem.split('_')[1]))
    
    if not images:
        print("画像ファイルが見つかりませんでした。")
        return
    
    # 最初の画像からフレームサイズを取得
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    
    # ビデオライターの設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 各画像をフレームとして追加
    for image in images:
        frame = cv2.imread(image)
        video.write(frame)
    
    # リソースの解放
    video.release()
    print(f"動画が作成されました: {output_video}")

if __name__ == "__main__":
    # 画像フォルダのパス
    image_folder = "無限ストレージ/RAG チャンキング/output/jucc_content_20250422"
    # 出力動画のパス
    output_video = "output_video.mp4"
    
    create_video_from_images(image_folder, output_video) 