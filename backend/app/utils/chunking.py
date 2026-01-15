import re
from typing import List

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks recursively based on separators.
        """
        final_chunks = []
        if not text:
            return final_chunks
            
        separator = self.separators[-1]
        new_separators = []
        for i, _s in enumerate(self.separators):
            if _s == "":
                separator = _s
                break
            if re.search(re.escape(_s), text):
                separator = _s
                new_separators = self.separators[i + 1:]
                break

        _splits = self._split_text_with_regex(text, separator)
        _good_splits = []
        
        for s in _splits:
            if len(s) < self.chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    final_chunks.extend(self._merge_splits(_good_splits, separator))
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s[:self.chunk_size]) # Force split if no separators left
                else:
                    sub_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        separators=new_separators
                    )
                    final_chunks.extend(sub_splitter.split_text(s))
        
        if _good_splits:
            final_chunks.extend(self._merge_splits(_good_splits, separator))
            
        return final_chunks

    def _split_text_with_regex(self, text: str, separator: str) -> List[str]:
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text) # Character split
        return [s for s in splits if s]

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        merged_text = []
        docs = []
        current_doc = []
        total = 0
        
        for split in splits:
            _len = len(split) + (len(separator) if current_doc else 0)
            if total + _len > self.chunk_size:
                if total > self.chunk_size: # Single split is huge
                     pass 
                
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc:
                        docs.append(doc)
                    
                    # Overlap logic
                    while total > self.chunk_overlap or (total + _len > self.chunk_size and total > 0):
                        total -= len(current_doc[0]) + (len(separator) if len(current_doc) > 1 else 0)
                        current_doc.pop(0)
                        
            current_doc.append(split)
            total = len(separator.join(current_doc)) # Recalculate precisely
            
        if current_doc:
            doc = separator.join(current_doc)
            if doc:
                docs.append(doc)
                
        return docs

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)
