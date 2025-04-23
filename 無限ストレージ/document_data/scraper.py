import requests
from bs4 import BeautifulSoup
import re
import datetime
import os

def scrape_jucc_site():
    
    url = "https://jucc.sakura.ne.jp/precedent/precedent-2006-12-13.html"
    
    try:
        # サイトにアクセス
        response = requests.get(url)
        
        # レスポンスのエンコーディングを自動検出
        response.encoding = response.apparent_encoding
        
        # BeautifulSoupで解析
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 本文を取得
        content = soup.get_text()
        
        # 不要な空白や改行を整理
        content = re.sub(r'\s+', ' ', content)
        
        # 現在の日付を取得
        current_date = datetime.datetime.now().strftime('%Y%m%d')
        
        # 保存先ディレクトリの作成
        output_dir = 'output'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # ファイル名の作成
        filename = f'{output_dir}/jucc_content_{current_date}.txt'
        
        # ファイルに保存
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content.strip())
        
        print(f'ファイルを保存しました: {filename}')
        return content.strip()
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return None

if __name__ == "__main__":
    content = scrape_jucc_site()
    if content:
        print(content) 