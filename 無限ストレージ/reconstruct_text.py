import json
import re

def reconstruct_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 正規表現でJSONデータを抽出
        pattern = r'データ:\s*({.*?})\s*--------------------------------------------------'
        matches = re.findall(pattern, content, re.DOTALL)
        
        if not matches:
            raise ValueError("JSONデータが見つかりませんでした")
        
        # チャンクを保存する辞書
        chunks = {}
        
        for json_str in matches:
            try:
                data = json.loads(json_str)
                chunk_id = data['id']
                content = data['content']
                chunks[chunk_id] = content
            except (json.JSONDecodeError, KeyError) as e:
                print(f"警告: データのパースに失敗しました: {e}")
                continue
        
        if not chunks:
            raise ValueError("有効なチャンクデータが見つかりませんでした")
        
        # チャンクIDを数値順にソート
        def get_chunk_number(chunk_id):
            match = re.search(r'chunk_(\d+)', chunk_id)
            if match:
                return int(match.group(1))
            return 0
        
        sorted_chunks = sorted(chunks.items(), key=lambda x: get_chunk_number(x[0]))
        reconstructed_text = ''.join(content for _, content in sorted_chunks)
        
        # 結果をファイルに保存
        with open('reconstructed_text.txt', 'w', encoding='utf-8') as f:
            f.write(reconstructed_text)
        
        return reconstructed_text
        
    except FileNotFoundError:
        print(f"エラー: ファイル '{file_path}' が見つかりません")
        return None
    except Exception as e:
        print(f"エラー: テキストの復元中に問題が発生しました: {e}")
        return None

if __name__ == '__main__':
    file_path = 'decoded_qr_data.txt'
    reconstructed_text = reconstruct_text(file_path)
    if reconstructed_text:
        print("文章の復元が完了しました。結果は reconstructed_text.txt に保存されています。") 
        