import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO

class ImagePDFLoader:

    @staticmethod
    def extract_images(pdf_path):

        doc = fitz.open(pdf_path)

        images = []

        for page_num in range(len(doc)):

            page = doc[page_num]

            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list):

                xref = img[0]

                base_image = doc.extract_image(xref)

                image_bytes = base_image["image"]

                image = Image.open(BytesIO(image_bytes))

                images.append({
                    "page": page_num + 1,
                    "image_index": img_index,
                    "image": image
                })

        return images