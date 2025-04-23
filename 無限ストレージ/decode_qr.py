import base64
import zlib
from datetime import datetime

def decode_qr_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    decoded_data = []
    for line in lines:
        # タイムスタンプとデータを分離
        timestamp, data = line.strip().split(': ', 1)
        
        # Base64デコード
        decoded_bytes = base64.b64decode(data)
        
        # zlibで解凍
        try:
            decompressed_data = zlib.decompress(decoded_bytes)
            decoded_data.append({
                'timestamp': timestamp,
                'data': decompressed_data.decode('utf-8')
            })
        except zlib.error:
            # 解凍できない場合は生のデータを保存
            decoded_data.append({
                'timestamp': timestamp,
                'data': decoded_bytes.hex()
            })
    
    return decoded_data

if __name__ == '__main__':
    file_path = '/Users/key/Desktop/Streamlit/無限ストレージ/qr_data.txt'
    decoded_data = decode_qr_data(file_path)
    
    # 結果をファイルに保存
    with open('decoded_qr_data.txt', 'w', encoding='utf-8') as f:
        for entry in decoded_data:
            f.write(f"タイムスタンプ: {entry['timestamp']}\n")
            f.write(f"データ: {entry['data']}\n")
            f.write("-" * 50 + "\n") 