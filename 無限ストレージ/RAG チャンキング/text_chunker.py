from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

class TextChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        テキストチャンカーを初期化
        
        Args:
            chunk_size (int): 各チャンクの最大トークン数
            chunk_overlap (int): チャンク間のオーバーラップトークン数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding("cl100k_base")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.get_token_count
        )
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        テキストをチャンクに分割し、メタデータを追加
        
        Args:
            text (str): 分割するテキスト
            metadata (Dict[str, Any]): 各チャンクに追加するメタデータ
            
        Returns:
            List[Dict[str, Any]]: チャンクとメタデータを含むリスト
        """
        if metadata is None:
            metadata = {}
            
        chunks = []
        split_texts = self.text_splitter.split_text(text)
        
        for i, chunk_text in enumerate(split_texts):
            chunk = {
                "chunk_id": f"chunk_{i:03d}",
                "content": chunk_text,
                "chunk_size": self.get_token_count(chunk_text),
                **metadata
            }
            chunks.append(chunk)
            
        return chunks
    
    def get_token_count(self, text: str) -> int:
        """
        テキストのトークン数を計算
        
        Args:
            text (str): トークン数を計算するテキスト
            
        Returns:
            int: トークン数
        """
        return len(self.encoding.encode(text)) 