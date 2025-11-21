import docx
import sys
import os

def extract_text(filepath):
    try:
        doc = docx.Document(filepath)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(extract_text(sys.argv[1]))
    else:
        print("Usage: python extract_docx.py <path_to_docx>")
