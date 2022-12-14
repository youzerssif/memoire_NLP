from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text as fallback_text_extraction
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context


# Step 1 Transform pdf to text
def transform_pdf_to_text():
    file_test = PdfReader('presentation.pdf')
    text = ''
    try:
        for pages in file_test.pages:
            text += pages.extract_text()
    except Exception as exc:
        text = fallback_text_extraction("presentation.pdf")
    return text

### Test
# print(transform_pdf_to_text())