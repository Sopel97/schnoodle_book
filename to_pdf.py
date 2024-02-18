from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from PIL import Image
from io import BytesIO
import glob
import os

os.environ['MOZ_HEADLESS'] = '1'

BASE_DPI = 96
#TARGET_DPI = 300
#TARGET_DPI = 192
TARGET_DPI = 96

PROFILE = webdriver.FirefoxProfile()
PROFILE.set_preference("layout.css.devPixelsPerPx", str(TARGET_DPI/BASE_DPI))
BROWSER = webdriver.Firefox(PROFILE)

BASE_OUT_DIR = f'books_images_{TARGET_DPI}dpi'
BASE_PDF_OUT_DIR = f'books_pdf_{TARGET_DPI}dpi'

DO_PDF = True

def save_elem(elem, path):
    if os.path.exists(path):
        print(f'Image already exists: {path}')
        return
    png_bin = elem.screenshot_as_png
    image = Image.open(BytesIO(png_bin)).convert('RGB')
    image.save(path, 'JPEG', optimize=True, quality=85, dpi=(TARGET_DPI, TARGET_DPI))
    print(f'Saved image: {path}')

# Nothing seems to work well...
# wkhtmltox has issues with shadows and blending
# os.system(f'c:/Programy/wkhtmltox/bin/wkhtmltopdf.exe --enable-local-file-access --load-error-handling ignore -B 0 -L 0 -R 0 -T 0 --disable-smart-shrinking "{filename}.html" "{filename}.pdf"')
# os.system(f'c:/Programy/wkhtmltox/bin/wkhtmltoimage.exe --enable-local-file-access --load-error-handling ignore --disable-smart-width --zoom 3.125 --width 2480 "{filename}.html" "{filename}.jpg"')
for filename in glob.glob('*.html'):
    path = os.path.abspath(filename)
    BROWSER.get(f'file:///{path}')
    #BROWSER.execute_script(f"document.body.style.transform='scale({TARGET_DPI/BASE_DPI})'")

    stem = filename.split('.')[0]
    out_dir = os.path.join(BASE_OUT_DIR, stem)
    os.makedirs(out_dir, exist_ok=True)

    image_paths = []
    covers = BROWSER.find_elements(By.CLASS_NAME, value='page_cover')
    path = os.path.join(out_dir, '0000.jpg')
    image_paths.append(path)
    save_elem(covers[0], path)
    i = 1
    for elem in BROWSER.find_elements(By.CLASS_NAME, value='page'):
        path = os.path.join(out_dir, f'{i:04d}.jpg')
        image_paths.append(path)
        save_elem(elem, path)
        i += 1
    path = os.path.join(out_dir, f'{i:04d}.jpg')
    image_paths.append(path)
    save_elem(covers[1], path)

    if DO_PDF:
        print('Creating PDF...')
        pdf_path = os.path.join(BASE_PDF_OUT_DIR, stem + '.pdf')
        Image.open(image_paths[0]).save(pdf_path, "PDF", resolution=TARGET_DPI, save_all=True, append_images=(Image.open(f) for f in image_paths[1:]))
        print(f'Created PDF: {pdf_path}')
