import os
import glob
from text_chunker import TextChunker
import qrcode
from PIL import Image
import json
import zlib
import base64

def compress_data(data: dict) -> str:
    """
    データを圧縮してBase64エンコードする
    
    Args:
        data (dict): 圧縮するデータ
        
    Returns:
        str: 圧縮されたBase64エンコードされた文字列
    """
    json_str = json.dumps(data, ensure_ascii=False)
    compressed = zlib.compress(json_str.encode('utf-8'))
    return base64.b64encode(compressed).decode('utf-8')

def process_text_and_generate_qr(input_pattern: str, output_dir: str):
    """
    テキストファイルを読み込み、チャンキングしてQRコードを生成する
    
    Args:
        input_pattern (str): 入力テキストファイルのパターン（ワイルドカード可）
        output_dir (str): 出力ディレクトリのパス
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 入力ファイルの検索
    input_files = glob.glob(input_pattern)
    if not input_files:
        print(f"エラー: 入力ファイルが見つかりません: {input_pattern}")
        return
    
    for input_file in input_files:
        print(f"処理中: {input_file}")
        
        # テキストファイルの読み込み
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # チャンキングの実行（チャンクサイズをさらに小さく設定）
        chunker = TextChunker(
            chunk_size=100,  # チャンクサイズをさらに小さく設定
            chunk_overlap=10  # オーバーラップも小さく設定
        )
        
        # メタデータの設定（最小限に抑える）
        metadata = {
            "source": os.path.basename(input_file)
        }
        
        # テキストのチャンキング
        chunks = chunker.chunk_text(text, metadata)
        
        # ファイルごとの出力ディレクトリを作成
        file_output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file))[0])
        os.makedirs(file_output_dir, exist_ok=True)
        
        # チャンク情報の保存
        chunks_info = []
        for i, chunk in enumerate(chunks):
            # チャンク情報の保存（最小限の情報のみ）
            chunk_info = {
                "id": chunk["chunk_id"],
                "content": chunk["content"]
            }
            chunks_info.append(chunk_info)
            
            # QRコードの生成（最大バージョンを使用）
            qr = qrcode.QRCode(
                version=40,  # 最大バージョンを使用
                error_correction=qrcode.constants.ERROR_CORRECT_H,
                box_size=10,
                border=4,
            )
            
            # データを圧縮してQRコードに設定
            compressed_data = compress_data(chunk_info)
            qr.add_data(compressed_data)
            
            try:
                qr.make(fit=False)
            except qrcode.exceptions.DataOverflowError:
                print(f"警告: チャンク {i} のデータが多すぎます。チャンクサイズをさらに調整してください。")
                continue
            
            # QRコード画像の生成
            img = qr.make_image(fill_color="black", back_color="white")
            
            # QRコード画像の保存
            qr_path = os.path.join(file_output_dir, f"chunk_{i:03d}.png")
            img.save(qr_path)
        
        # チャンク情報のJSONファイルとして保存
        with open(os.path.join(file_output_dir, "chunks_info.json"), "w", encoding="utf-8") as f:
            json.dump(chunks_info, f, ensure_ascii=False, indent=2)
        
        print(f"完了: {input_file} -> {file_output_dir}")

if __name__ == "__main__":
    # 相対パスで指定
    input_pattern = "無限ストレージ/document_data/output/jucc_content_20250422.txt"  # 特定のファイルを指定
    output_dir = "./output"
    
    process_text_and_generate_qr(input_pattern, output_dir)
    print("すべての処理が完了しました。") 