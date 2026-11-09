import pandas as pd
from docx import Document as DocxDocument
from pptx import Presentation
from langchain_core.documents import Document

from loaders.pdf_loader import PDFLoader


class MultiLoader:

    @staticmethod
    def load_file(file_path, file_name):

        extension = file_name.split(".")[-1].lower()

        # ----------------------------------------------------
        # PDF
        # ----------------------------------------------------
        if extension == "pdf":

            docs = PDFLoader.load_pdf(file_path)

            for doc in docs:
                doc.metadata["source"] = file_name

            return docs

        # ----------------------------------------------------
        # TXT
        # ----------------------------------------------------
        elif extension == "txt":

            with open(file_path, "r", encoding="utf-8") as f:

                text = f.read()

            return [
                Document(
                    page_content=text,
                    metadata={"source": file_name}
                )
            ]

        # ----------------------------------------------------
        # DOCX
        # ----------------------------------------------------
        elif extension == "docx":

            doc = DocxDocument(file_path)

            text = "\n".join(
                [p.text for p in doc.paragraphs]
            )

            return [
                Document(
                    page_content=text,
                    metadata={"source": file_name}
                )
            ]

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------
        elif extension == "csv":

            df = pd.read_csv(file_path)

            text = df.to_string(index=False)

            return [
                Document(
                    page_content=text,
                    metadata={"source": file_name}
                )
            ]

        # ----------------------------------------------------
        # PPTX
        # ----------------------------------------------------
        elif extension == "pptx":

            prs = Presentation(file_path)

            slides_text = []

            for slide in prs.slides:

                for shape in slide.shapes:

                    if hasattr(shape, "text"):

                        slides_text.append(shape.text)

            text = "\n".join(slides_text)

            return [
                Document(
                    page_content=text,
                    metadata={"source": file_name}
                )
            ]

        else:

            raise ValueError(f"Unsupported file type: {extension}")