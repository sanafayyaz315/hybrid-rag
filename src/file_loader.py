# file_loader.py

from typing import List, Tuple, Union
import os
import fitz  # for PDF reading

class FileLoader:
    """Handles loading text from local file or a directory.(txt/pdf). Can be extended for cloud sources.
       Returns a list of tuple where a an entry is (source, text)
    """

    @staticmethod
    def load_files(path: Union[str, os.PathLike]) -> List[Tuple[str, str]]:
        texts = []
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

        if os.path.isfile(path):
            texts.append(FileLoader._load_single_file(path))
        elif os.path.isdir(path):
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                try:
                    texts.append(FileLoader._load_single_file(file_path))
                except Exception as e:
                    print(f"Error reading file {file}: {e}")
        else:
            raise ValueError(f"Path is neither a file nor a directory: {path}")

        return [t for t in texts if t is not None]

    @staticmethod
    def _load_single_file(path: str) -> Tuple[str, str]:
        filename = os.path.basename(path)
        ext = os.path.splitext(filename)[1].lower()

        if ext in [".txt", ""]:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            return filename, text
        elif ext == ".pdf":
            pdf_text = []
            with fitz.open(path) as pdf_doc:
                for page in pdf_doc:
                    pdf_text.append(page.get_text("text"))
            return filename, "\n".join(pdf_text)
        else:
            print(f"Skipping unsupported file type: {filename}")
            return None


if __name__ == "__main__":
    file_path = "../data/chapters/Chapter_1"
    dir_path = "../data/chapters"

    text = FileLoader.load_files(path=file_path)
    print("text loaded from file successfully")

    text = FileLoader.load_files(path=dir_path)
    print("text loaded from directory successfully")
    print(text[0])