import cv2
import os

def extract_frames(video_path, output_dir='frames'):
    
    # 出力ディレクトリが存在しない場合は作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 動画ファイルを開く
    cap = cv2.VideoCapture(video_path)
    
    # フレーム数を取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"総フレーム数: {total_frames}")
    
    frame_count = 0
    while True:
        # フレームを読み込む
        ret, frame = cap.read()
        
        # フレームが読み込めなかった場合は終了
        if not ret:
            break
        
        # フレームを保存
        frame_path = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"処理済みフレーム数: {frame_count}/{total_frames}")
    
    # リソースを解放
    cap.release()
    print(f"フレームの抽出が完了しました。{frame_count}枚のフレームを保存しました。")

if __name__ == "__main__":
    video_path = "output_video.mp4"
    extract_frames(video_path) 