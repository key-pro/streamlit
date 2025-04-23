import qrcode
from qrcode.constants import ERROR_CORRECT_L, ERROR_CORRECT_M, ERROR_CORRECT_Q, ERROR_CORRECT_H
import os
import cv2
from pyzbar.pyzbar import decode
import time
import numpy as np

def get_qr_capacity_info():
    """
    QRコードの各バージョンとエラー訂正レベルごとの容量情報を返す
    """
    # 数字モードでの容量（参考値）
    capacities = {
        "L": {
            1: 41, 2: 77, 3: 127, 4: 187, 5: 255, 6: 322, 7: 370, 8: 461, 9: 552, 10: 652,
            11: 772, 12: 883, 13: 1022, 14: 1101, 15: 1250, 16: 1408, 17: 1548, 18: 1725,
            19: 1903, 20: 2061, 21: 2232, 22: 2409, 23: 2620, 24: 2812, 25: 3057, 26: 3283,
            27: 3517, 28: 3669, 29: 3909, 30: 4158, 31: 4417, 32: 4686, 33: 4965, 34: 5253,
            35: 5529, 36: 5836, 37: 6153, 38: 6479, 39: 6743, 40: 7089
        },
        "M": {
            1: 34, 2: 63, 3: 101, 4: 149, 5: 202, 6: 255, 7: 293, 8: 365, 9: 432, 10: 513,
            11: 604, 12: 691, 13: 796, 14: 871, 15: 991, 16: 1082, 17: 1212, 18: 1346,
            19: 1500, 20: 1600, 21: 1708, 22: 1872, 23: 2059, 24: 2188, 25: 2395, 26: 2544,
            27: 2701, 28: 2857, 29: 3035, 30: 3289, 31: 3486, 32: 3693, 33: 3909, 34: 4134,
            35: 4343, 36: 4588, 37: 4775, 38: 5039, 39: 5313, 40: 5596
        },
        "Q": {
            1: 27, 2: 48, 3: 77, 4: 111, 5: 144, 6: 178, 7: 207, 8: 259, 9: 312, 10: 364,
            11: 427, 12: 489, 13: 580, 14: 621, 15: 703, 16: 775, 17: 876, 18: 948,
            19: 1063, 20: 1159, 21: 1224, 22: 1358, 23: 1468, 24: 1588, 25: 1718, 26: 1804,
            27: 1933, 28: 2085, 29: 2181, 30: 2358, 31: 2473, 32: 2670, 33: 2805, 34: 2949,
            35: 3081, 36: 3244, 37: 3417, 38: 3599, 39: 3791, 40: 3993
        },
        "H": {
            1: 17, 2: 34, 3: 58, 4: 82, 5: 106, 6: 139, 7: 154, 8: 202, 9: 235, 10: 288,
            11: 331, 12: 374, 13: 427, 14: 468, 15: 530, 16: 602, 17: 674, 18: 746,
            19: 813, 20: 919, 21: 969, 22: 1056, 23: 1108, 24: 1228, 25: 1286, 26: 1425,
            27: 1501, 28: 1581, 29: 1677, 30: 1782, 31: 1897, 32: 2022, 33: 2157, 34: 2301,
            35: 2361, 36: 2524, 37: 2625, 38: 2735, 39: 2927, 40: 3057
        }
    }
    # バイナリーモードでの容量
    binary_capacities = {
        "L": {
            1: 17, 2: 32, 3: 53, 4: 78, 5: 106, 6: 134, 7: 154, 8: 192, 9: 230, 10: 271,
            11: 321, 12: 367, 13: 425, 14: 458, 15: 520, 16: 586, 17: 644, 18: 718,
            19: 792, 20: 858, 21: 929, 22: 1003, 23: 1091, 24: 1171, 25: 1273, 26: 1367,
            27: 1465, 28: 1528, 29: 1628, 30: 1732, 31: 1840, 32: 1952, 33: 2068, 34: 2188,
            35: 2303, 36: 2431, 37: 2563, 38: 2699, 39: 2809, 40: 2953
        }
    }
    return capacities, binary_capacities

def generate_qr_code(data, version, filename):
    """
    QRコードを生成して保存する関数
    """
    try:
        qr = qrcode.QRCode(
            version=version,
            error_correction=ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(data)
        qr.make(fit=True)
        
        # QRコード画像を生成
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(filename)
        print(f"QRコードを保存しました: {filename}")
        return True
    except Exception as e:
        print(f"エラー: {filename} の生成に失敗しました - {str(e)}")
        return False

def realtime_qr_detection():
    """
    カメラでリアルタイムにQRコードを認識する関数
    """
    # カメラを起動
    cap = cv2.VideoCapture(0)
    
    # フォント設定
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 前回の検出時間
    last_detection_time = 0
    # 検出間隔（秒）
    detection_interval = 1.0
    
    print("カメラでQRコードを認識中...")
    print("'q'キーを押すと終了します")
    
    while True:
        # フレームを取得
        ret, frame = cap.read()
        if not ret:
            print("カメラからの画像取得に失敗しました")
            break
        
        # 現在時刻
        current_time = time.time()
        
        # 一定間隔でQRコードを検出
        if current_time - last_detection_time >= detection_interval:
            # QRコードをデコード
            decoded_objects = decode(frame)
            
            # 検出したQRコードを処理
            for obj in decoded_objects:
                # QRコードの位置を取得
                points = obj.polygon
                if len(points) > 4:
                    hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
                    points = hull
                
                # 四角形を描画
                n = len(points)
                for j in range(n):
                    cv2.line(frame, points[j], points[(j+1) % n], (0, 255, 0), 3)
                
                # データを表示
                data = obj.data.decode('utf-8')
                print(f"検出されたデータ: {data}")
                
                # テキストを表示
                cv2.putText(frame, data, (points[0].x, points[0].y - 10),
                           font, 0.5, (0, 255, 0), 2)
            
            last_detection_time = current_time
        
        # フレームを表示
        cv2.imshow('QR Code Detection', frame)
        
        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # リソースを解放
    cap.release()
    cv2.destroyAllWindows()

def main():
    capacities, binary_capacities = get_qr_capacity_info()
    
    print("QRコードの情報量上限（Lレベル、数字モードの場合）")
    print("=" * 50)
    
    level = "L"
    print(f"\nエラー訂正レベル: {level}")
    print("-" * 30)
    print("バージョン | 最大文字数")
    print("-" * 20)
    
    for version in range(1, 41):
        if version in capacities[level]:
            print(f"{version:8d} | {capacities[level][version]:8d}")
    
    # 最大容量の情報を表示
    max_version = max(capacities[level].keys())
    max_capacity = capacities[level][max_version]
    print(f"\n最大容量（バージョン{max_version}）: {max_capacity}文字")
    
    # 主なバージョンの情報を強調表示
    print("\n主なバージョンの容量:")
    print("-" * 30)
    important_versions = [1, 10, 20, 30, 40]
    for version in important_versions:
        print(f"バージョン{version:2d}: {capacities[level][version]:4d}文字")

    print("\nQRコードの情報量上限（Lレベル、バイナリーモードの場合）")
    print("=" * 50)
    
    print(f"\nエラー訂正レベル: {level}")
    print("-" * 30)
    print("バージョン | 最大バイト数")
    print("-" * 20)
    
    for version in range(1, 41):
        if version in binary_capacities[level]:
            print(f"{version:8d} | {binary_capacities[level][version]:8d}")
    
    # 最大容量の情報を表示
    max_version = max(binary_capacities[level].keys())
    max_capacity = binary_capacities[level][max_version]
    print(f"\n最大容量（バージョン{max_version}）: {max_capacity}バイト")
    
    # 主なバージョンの情報を強調表示
    print("\n主なバージョンの容量:")
    print("-" * 30)
    important_versions = [1, 10, 20, 30, 40]
    for version in important_versions:
        print(f"バージョン{version:2d}: {binary_capacities[level][version]:4d}バイト")

    # QRコードの生成
    print("\nQRコードの生成")
    print("=" * 50)
    
    # 出力ディレクトリを作成
    os.makedirs("qr_codes", exist_ok=True)
    
    # 各バージョンでテストデータを生成
    test_data = "A" * 10  # 10バイトのテストデータ
    
    for version in important_versions:
        filename = f"qr_codes/version_{version}.png"
        if generate_qr_code(test_data, version, filename):
            print(f"バージョン{version}のQRコードを生成しました")
        print("-" * 30)
    
    # リアルタイムQRコード認識を開始
    print("\nリアルタイムQRコード認識を開始します...")
    realtime_qr_detection()

if __name__ == "__main__":
    main()
