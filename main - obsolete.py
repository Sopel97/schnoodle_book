'''
           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                   Version 2, December 2004

Copyright (C) 2004 Sam Hocevar <sam@hocevar.net>

Everyone is permitted to copy and distribute verbatim or modified
copies of this license document, and changing it is allowed as long
as the name is changed.

           DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
  TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION

 0. You just DO WHAT THE FUCK YOU WANT TO.
'''

import praw
#import psaw
import pmaw
psaw = pmaw
import pickle
import os
import sys
import pprint
import redvid
import requests
import tqdm
import cv2
import glob
import math
import random
import time
import numpy as np
from collections import OrderedDict

def write_roman(num):

    roman = OrderedDict()
    roman[1000] = "M"
    roman[900] = "CM"
    roman[500] = "D"
    roman[400] = "CD"
    roman[100] = "C"
    roman[90] = "XC"
    roman[50] = "L"
    roman[40] = "XL"
    roman[10] = "X"
    roman[9] = "IX"
    roman[5] = "V"
    roman[4] = "IV"
    roman[1] = "I"

    def roman_num(num):
        for r in roman.keys():
            x, y = divmod(num, r)
            yield roman[r] * x
            num -= (r * x)
            if num <= 0:
                break

    return "".join([a for a in roman_num(num)])

CLIENT_ID = '' # Redacted
CLIENT_SECRET = '' # Redacted
USER_AGENT = '' # Redacted
REDIRECT_URI = 'http://localhost:8080'

print('Making reddit instance')
REDDIT = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)
REDDIT.read_only = True
print('Made reddit instance')

print('Making PSAW API')
PSAW_API = psaw.PushshiftAPI()
print('Made PSAW API')

def download(url, filename):
    import functools
    import pathlib
    import shutil
    import requests
    from tqdm.auto import tqdm

    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()  # Will only raise for 4xx codes, so...
        raise RuntimeError(f"Request to {url} returned status code {r.status_code}")
    file_size = int(r.headers.get('Content-Length', 0))

    path = pathlib.Path(filename).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    desc = "(Unknown total file size)" if file_size == 0 else ""
    r.raw.read = functools.partial(r.raw.read, decode_content=True)  # Decompress if needed
    with tqdm.wrapattr(r.raw, "read", total=file_size, desc=desc) as r_raw:
        with path.open("wb") as f:
            shutil.copyfileobj(r_raw, f)

    return path

def download_gfycat_video(submission):
    basepath = './media/'
    path = os.path.join(os.path.abspath(basepath), submission.id + '.mp4')
    if not os.path.exists(path):
        try:
            r = requests.get(submission.url, allow_redirects=True, headers={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0'})
            if r.status_code != 200:
                r.raise_for_status()
                raise RuntimeError(f"Request to {submission.url} returned status code {r.status_code}")

            start = 0
            while True:
                f = r.text.find('"contentUrl":', start)
                if f == -1:
                    raise RuntimeError('')
                start = f+len('"contentUrl":')+1
                end = r.text.find('"', start)
                if end == -1:
                    raise RuntimeError('')

                raw_url = r.text[start:end]
                print(raw_url)
                if raw_url.endswith('.mp4'):
                    break
                else:
                    start = end+1

            download(raw_url, path)
        except:
            # cannot find resource anymore
            with open(path, 'wb') as f:
                pass

    return [path]

def is_imgur_gallery(submission):
    url = submission.url
    return 'gallery' in url or '/a/' in url

def download_imgur_image(submission):
    url = submission.url
    url = url.replace('.gifv', '.mp4')
    url = url.replace('.jpeg', '.jpg')
    if not url.endswith('.jpg') and not url.endswith('.jpeg') and not url.endswith('.mp4'):
        url += '.jpg'
    basepath = './media/'
    if url.endswith('.mp4'):
        path = os.path.join(os.path.abspath(basepath), submission.id + '.mp4')
    else:
        path = os.path.join(os.path.abspath(basepath), submission.id + '.jpg')

    if not os.path.exists(path):
        try:
            download(url, path)
        except:
            # cannot find resource anymore
            with open(path, 'wb') as f:
                pass

    return [path]

def download_reddit_video(submission):
    basepath = './media/'
    path = os.path.join(os.path.abspath(basepath), submission.id + '.mp4')
    if not os.path.exists(path):
        try:
            downloader = redvid.Downloader(max_q=True)
            downloader.url = submission.url
            downloader.path = basepath

            # cheat the filename in
            downloader.check()
            downloader.file_name = path

            downloader.download()
        except:
            # cannot find resource anymore
            with open(path, 'wb') as f:
                pass

    return [path]

def download_reddit_image(submission):
    basepath = './media/'
    url = submission.url
    if '?' in url:
        url = url.split('?')[0]
    url = url.replace('.gifv', '.mp4')
    ext = url.split('.')[-1]
    path = os.path.join(os.path.abspath(basepath), submission.id + '.' + ext)
    path_mp4 = os.path.join(os.path.abspath(basepath), submission.id + '.mp4')
    if not os.path.exists(path) and not os.path.exists(path_mp4):
        try:
            download(url, path)
            if ext == 'gif':
                os.system(f'ffmpeg -i {path} {path_mp4}')
                os.remove(path)
        except:
            # cannot find resource anymore
            if ext == 'gif':
                with open(path_mp4, 'wb') as f:
                    pass
            else:
                with open(path, 'wb') as f:
                    pass

    if ext == 'gif':
        return [path_mp4]
    else:
        return [path]

def download_reddit_gallery(submission):
    basepath = './media/'
    paths = []
    i = 0
    try:
        for k, v in sorted(submission.media_metadata.items(), key=lambda x:x[0]):
            url = v['s']['u'].split('?')[0]
            url = url.replace('preview.redd.it', 'i.redd.it')
            ext = url.split('.')[-1]
            path = os.path.join(os.path.abspath(basepath), f'{submission.id}_{i}.{ext}')
            paths.append(path)
            if not os.path.exists(path):
                try:
                    download(url, path)
                except:
                    # cannot find resource anymore
                    with open(path, 'wb') as f:
                        pass
            i += 1
    except:
        for name in glob.glob(os.path.join(os.path.abspath(basepath), f'{submission.id}_*')):
            paths.append(name)
        if not paths:
            # cannot find resource anymore
            path = os.path.join(os.path.abspath(basepath), f'{submission.id}_0.jpg')
            with open(path, 'wb') as f:
                pass

    return paths

def is_comment_a_poem(comment):
    # at lest 3 verses, with empty line between each
    # should be a good enough heuristic
    return comment.body.count('\n\n') >= 2

class Poem:
    def __init__(self, prompt, verses):
        self.prompt = prompt
        self.verses = verses

def parse_poem_from_comment(comment, parent_comment=None):
    prompt = None
    verses = []

    for line in comment.body.split('\n'):
        line = line.strip()
        if line.startswith('>'):
            prompt = line.strip('>*\'’"')
        elif len(line) > 0 and line.lower().startswith('edit:'):
            break
        elif len(line) > 0 and ord(line[0]) <= 127 and not line.startswith('___') and not line.startswith('---'):
            verses.append(line)

    if prompt is None and parent_comment is not None:
        prompt_lines = []
        for line in parent_comment.body.split('\n'):
            line = line.strip()
            line_lower = line.lower()
            if line_lower.startswith('edit') or line_lower.startswith('/edit') or line_lower.startswith('#edit'):
                break
            if len(line) > 0:
                prompt_lines.append(line)
        if len(prompt_lines) > 0:
            prompt = '\n'.join(prompt_lines)

    return Poem(prompt, verses)

def is_image_path(path):
    return path.endswith('.png') or path.endswith('.jpg')

def resize_image_with_scale(img, scale, max_dpi, max_width_inches, max_height_inches):
    max_width = max_width_inches * max_dpi
    max_height = max_height_inches * max_dpi
    width = img.shape[1]
    height = img.shape[0]
    scale = min((scale, max_width/width, max_height/height))
    new_width = int(width*scale)
    new_height = int(height*scale)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    return img

def maybe_resize_image(img, max_dpi, max_width_inches, max_height_inches):
    max_width = max_width_inches * max_dpi
    max_height = max_height_inches * max_dpi

    width = img.shape[1]
    height = img.shape[0]
    if width > max_width or height > max_height:
        scale = min(max_width/width, max_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

    return img

def make_collage_from_video(path, max_width_inches, max_height_inches):
    print('Max height inches: ', max_height_inches)
    vid = cv2.VideoCapture(path)
    vid_fps = vid.get(cv2.CAP_PROP_FPS)
    vid_frame_count = vid.get(cv2.CAP_PROP_FRAME_COUNT)
    vid_duration_ms = vid_frame_count / vid_fps * 1000
    vid_width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) + 4 # account for border
    vid_height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) + 4
    vid_aspect_ratio = vid_width / vid_height

    best_collage_aspect_ratio = max_width_inches / max_height_inches

    count_ratio = best_collage_aspect_ratio / vid_aspect_ratio

    best = None
    best_fitness = None

    for x in range(1, 9):
        y = x / count_ratio

        fitness_fit = abs(y - round(y)) * 2 / y
        images_per_ms = (x * y) / vid_duration_ms
        if images_per_ms > 0.002:
            fitness_count = 10000
        else:
            fitness_count = math.pow(abs(1000 - images_per_ms) / 1000, 0.5)
        fitness = fitness_fit + fitness_count
        if best_fitness is None or fitness < best_fitness:
            best_fitness = fitness
            best = (x, int(round(y)))

    print(f'Making a {best[0]}x{best[1]} collage')
    collage = []
    i = 0
    num_images = best[0] * best[1]
    for y in range(best[1]):
        collage_line = []
        for x in range(best[0]):
            ts = vid_duration_ms * (i / num_images)
            vid.set(cv2.CAP_PROP_POS_MSEC, ts)
            success, image = vid.read()
            collage_line.append(cv2.copyMakeBorder(
                image,
                2,
                2,
                2,
                2,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255)
            ))
            i += 1
        collage.append(collage_line)

    return cv2.vconcat([cv2.hconcat(line) for line in collage])

def make_collage_from_gallery(media_paths, max_dpi, max_width_inches, max_height_inches):
    print('Max height inches: ', max_height_inches)

    imgs = [maybe_resize_image(cv2.imread(media_path, cv2.IMREAD_UNCHANGED), max_dpi, max_width_inches, max_height_inches) for media_path in media_paths]
    total_space_area = max_width_inches * max_height_inches
    total_img_area = sum((img.shape[1]/max_dpi) * (img.shape[0]/max_dpi) for img in imgs)
    scale_factor = (total_space_area / total_img_area)**0.5
    resized_imgs = [resize_image_with_scale(img, scale_factor, max_dpi, max_width_inches, max_height_inches) for img in imgs]

    space_width_pixels = int(max_width_inches * max_dpi)
    space_height_pixels = int(max_height_inches * max_dpi)

    def find_best_spot(img_width, img_height, num_iters):
        best_x = 0
        best_y = 0
        best_empty_pixels = 0
        for i in range(num_iters):
            x = random.randint(0, space_width_pixels-img_width)
            y = random.randint(0, space_height_pixels-img_height)
            empty_pixels = img_width * img_height - np.count_nonzero(collage_bitmap[y:y+img_height, x:x+img_width])
            if empty_pixels > best_empty_pixels:
                best_x = x
                best_y = y
                best_empty_pixels = empty_pixels

        return best_x, best_y

    NUM_ITERS = 20
    NUM_TRIES = 10

    best_collage = None
    best_collage_empty_pixels = space_height_pixels * space_width_pixels

    # seed the rng to make results deterministic
    # the seed is derived from the file names, which contain submission ids
    filenames = [os.path.basename(path) for path in media_paths]
    random.seed('$'.join(filenames))

    for i in range(NUM_TRIES):
        random.shuffle(resized_imgs)

        collage = np.zeros((space_height_pixels, space_width_pixels, 3), np.uint8)
        collage_bitmap = np.zeros((space_height_pixels, space_width_pixels), np.uint8)

        for img in resized_imgs:
            img_width = img.shape[1]
            img_height = img.shape[0]
            best_x, best_y = find_best_spot(img_width, img_height, NUM_ITERS)
            collage[best_y:best_y+img_height, best_x:best_x+img_width] = img[:,:,:3]
            collage_bitmap[best_y:best_y+img_height, best_x:best_x+img_width] = 1

        collage_empty_pixels = space_height_pixels * space_width_pixels - np.count_nonzero(collage_bitmap)

        if collage_empty_pixels < best_collage_empty_pixels:
            best_collage = collage
            best_collage_empty_pixels = collage_empty_pixels

    empty_space_pct = best_collage_empty_pixels * 100.0 / (space_height_pixels * space_width_pixels)
    print('Made collage with {}% empty space.')

    return best_collage

PAGE_WIDTH_CM = 21
PAGE_HEIGHT_CM = 29.7
PAGE_INNER_WIDTH_CM = PAGE_WIDTH_CM-1.35*2
PAGE_INNER_HEIGHT_CM = PAGE_HEIGHT_CM-1.35*2
MAX_DPI = 300

class PoemPage:
    def __init__(self, comment, media):
        self.poem = parse_poem_from_comment(comment, comment._parent_comment if hasattr(comment, '_parent_comment') else None)
        self.created_utc = comment.created_utc
        self.comment_permalink = 'https://www.reddit.com' + comment.permalink
        self.submission_permalink = 'https://www.reddit.com' + comment.submission.permalink

        # cheat to get author name without pulling the whole object
        if comment.submission.author is not None:
            old_fetched = comment.submission.author._fetched
            comment.submission.author._fetched = True
            self.submission_author = comment.submission.author.name
            comment.submission.author._fetched = old_fetched
        else:
            self.submission_author = None

        self.submission_votes = comment.submission.score
        self.comment_votes = comment.score
        self.submission_title = comment.submission.title
        self.subreddit_name = '' #comment.submission.subreddit.name
        self.source_media_paths = media
        prompt_lines_est = 0 if self.poem.prompt is None else math.ceil(len(self.poem.prompt) / 50)
        self.image_height_inches = PAGE_INNER_HEIGHT_CM/2.54 - (0.5 + (0.3 * math.ceil(len(self.submission_title) / 40)) + 0.125 + (0.75 * (self.poem.prompt is not None) + 0.125 * prompt_lines_est) + len(self.poem.verses) * 0.23)
        self.processed_media_path = self.process_media(media, self.image_height_inches)
        self.invalid = self.processed_media_path == None

    def process_media(self, media_paths, max_height_inches):
        max_width_inches = PAGE_INNER_WIDTH_CM / 2.54

        if len(media_paths) == 0 or max_height_inches <= 1:
            return None

        if len(media_paths) > 1:
            new_path = media_paths[0].replace('media', 'media_processed')
            new_path = '_'.join(new_path.split('.')[0].split('_')[:-1]) + '.jpg'

            if not os.path.exists(new_path):
                img = make_collage_from_gallery(media_paths, MAX_DPI, max_width_inches, max_height_inches)
                cv2.imwrite(new_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            return new_path

        media_path = media_paths[0]

        if is_image_path(media_path):
            new_path = media_path.replace('media', 'media_processed')
            new_path = new_path.split('.')[0] + '.jpg'

            if not os.path.exists(new_path):
                img = cv2.imread(media_path, cv2.IMREAD_UNCHANGED)
                img = maybe_resize_image(img, MAX_DPI, max_width_inches, max_height_inches)
                cv2.imwrite(new_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            return new_path

        elif media_path.endswith('.mp4'):
            new_path = media_path.replace('media', 'media_processed')
            new_path = new_path.split('.')[0] + '.jpg'

            if not os.path.exists(new_path):
                img = make_collage_from_video(media_path, max_width_inches, max_height_inches)
                img = maybe_resize_image(img, MAX_DPI, max_width_inches, max_height_inches)
                cv2.imwrite(new_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            return new_path
        else:
            return media_path

    def get_formatted_poem(self):
        formatted_poem_lines = []

        for line in self.poem.verses:
            line = line.replace('…', '...')
            formatted_line = ''
            inside_italics = False
            inside_bold = False
            inside_superscript = 0
            inside_superscript_parens = []
            while len(line) > 0:
                if line.startswith('**'):
                    inside_bold = not inside_bold
                    if inside_bold:
                        formatted_line += '<b>'
                    else:
                        formatted_line += '</b>'
                    line = line[2:]
                elif line.startswith('*'):
                    inside_italics = not inside_italics
                    if inside_italics:
                        formatted_line += '<i>'
                    else:
                        formatted_line += '</i>'
                    line = line[1:]
                elif line.startswith('^('):
                    inside_superscript += 1
                    inside_superscript_parens.append(True)
                    formatted_line += '<sup>'
                    line = line[2:]
                elif line.startswith('^'):
                    inside_superscript += 1
                    inside_superscript_parens.append(False)
                    formatted_line += '<sup>'
                    line = line[1:]
                elif inside_superscript > 0 and (inside_superscript_parens[-1] and line.startswith(')') or (not inside_superscript_parens[-1] and line.startswith(' '))):
                    inside_superscript -= 1
                    inside_superscript_parens.pop()
                    formatted_line += '</sup>'
                    line = line[1:]
                else:
                    formatted_line += line[0]
                    line = line[1:]

            if inside_bold:
                formatted_line += '</b>'
            if inside_italics:
                formatted_line += '</i>'
            while inside_superscript > 0:
                formatted_line += '</sup>'
                inside_superscript -= 1

            formatted_poem_lines.append(formatted_line)

        return '<br/>'.join(formatted_poem_lines)

    def generate_html(self, page_number):
        lines = []
        lines.append(f'<div class="page">')
        lines.append(f'<div class="page_inner">')
        lines.append(f'<h2 class="title"><a class="title_link" href="{self.submission_permalink}">{self.submission_title}</a></h2>')
        lines.append(f'<br/>')

        if self.processed_media_path is not None:
            if self.processed_media_path.endswith('.jpg') or self.processed_media_path.endswith('.png'):
                src_path = os.path.relpath(self.processed_media_path.replace('\\', '/'))
                lines.append('<div class="media">')
                lines.append(f'<img src="{src_path}" class="media_image" style="max-height:{self.image_height_inches*2.54}cm"/>')
                lines.append(f'<img src="img/pin.png" class="pin_top_left"/>')
                lines.append(f'<img src="img/pin.png" class="pin_top_right"/>')
                if self.submission_author is not None:
                    lines.append(f'<p class="media_watermark">u/{self.submission_author}</p>')
                lines.append('</div>')
            else:
                lines.append(f'Media: {self.processed_media_path}')
                lines.append(f'<br/>')

        formatted_poem = self.get_formatted_poem()
        poem_card_id = random.Random(formatted_poem).randrange(4)
        lines.append(f'<div class="poem_card_{poem_card_id}">')
        if self.poem.prompt:
            lines.append(f'<p class="prompt"><blockquote><i>’{self.poem.prompt}’</i></blockquote></p>')
            lines.append(f'<hr style="position:relative;margin-top:8mm;margin-left:8mm;width:90mm;color: #bbb;">')
        lines.append(f'<p class="poem"><font face="poem_font">{formatted_poem}</font></p>')
        lines.append(f'<p class="poem_author">u/SchnoodleDoodleDo</p>')
        lines.append(f'<img src="img/pin.png" class="pin_top_left"/>')
        lines.append(f'<img src="img/pin.png" class="pin_top_right"/>')
        lines.append(f'</div>')
        #lines.append(f'Subreddit: {self.subreddit_name}')
        #lines.append(f'Submission: {self.submission_permalink}')
        #lines.append(f'Submission author: {self.submission_author}')
        #lines.append(f'Submission votes: {self.submission_votes}')
        #lines.append(f'Comment: {self.comment_permalink}')
        #lines.append(f'Comment votes: {self.comment_votes}')
        lines.append(f'</div>')
        if page_number is not None:
            lines.append(f'<p class="page_number">{page_number}</p>')
        lines.append(f'</div>')

        return '\n'.join(lines)

def make_front_cover_page(tome_number, min_created_utc, max_created_utc):
    min_time = time.strftime('%d %B %Y', time.localtime(min_created_utc))
    max_time = time.strftime('%d %B %Y', time.localtime(max_created_utc))
    tome_id = write_roman(tome_number)

    lines = []
    lines.append(f'<div class="page_cover">')
    lines.append(f'<div class="book_title_main_1 burn" data-text="SchnoodleDoodleDo"></div>')
    lines.append(f'<div class="book_title_main_2 burn" data-text="Poetry Book"></div>')
    lines.append(f'<div class="book_title_tome burn" data-text="Tome {tome_id}"></div>')
    lines.append(f'<div class="book_title_date_span burn" data-text="{min_time} - {max_time}"></div>')
    lines.append(f'</div>')

    return '\n'.join(lines)

def make_back_cover_page():
    return '<div class="page_cover"></div>'

def get_html_header():
    lines = []
    lines.append('<style>')
    lines.append('@font-face {')
    lines.append('    font-family: "poem_font";')
    lines.append('    src: url("steel-city-comic.regular.ttf");')
    lines.append('}')
    lines.append('.page_inner {')
    lines.append(f'    width: {PAGE_INNER_WIDTH_CM}cm;')
    lines.append(f'    height: {PAGE_INNER_HEIGHT_CM}cm;')
    lines.append('    margin: 0;')
    lines.append('    padding: 1.35cm 1.35cm 1.35cm 1.35cm;')
    lines.append('}')
    lines.append('.page {')
    lines.append('    page-break-after: always;')
    lines.append(f'    width: {PAGE_WIDTH_CM}cm;')
    lines.append(f'    height: {PAGE_HEIGHT_CM}cm;')
    lines.append('    background-size: cover;')
    lines.append('    background-image: url(\'img/cork_background.jpg\');')
    lines.append('    position: relative;')
    lines.append('    margin: 0;')
    lines.append('}')
    lines.append('.page_cover {')
    lines.append('    page-break-after: always;')
    lines.append(f'    width: {PAGE_WIDTH_CM}cm;')
    lines.append(f'    height: {PAGE_HEIGHT_CM}cm;')
    lines.append('    background-size: cover;')
    lines.append('    background-image: url(\'img/cover_background.jpg\');')
    lines.append('    position: relative;')
    lines.append('    margin: 0;')
    lines.append('}')
    lines.append('.title {')
    lines.append('    font-variant-caps: small-caps;')
    lines.append('    font-family: "Times New Roman";')
    lines.append('    text-shadow: 0 0 5mm white, 0 0 5mm white, 0 0 5mm white, 0 0 5mm white, 0 0 5mm white, 0 0 5mm white, 0 0 5mm white, 0 0 5mm white, 0 0 5mm white, 0 0 5mm white;')
    lines.append('    text-align: center;')
    lines.append('    font-size: 7mm;')
    lines.append('    line-height: 7mm;')
    lines.append('    margin: 0;')
    lines.append('    color: darkgreen;')
    lines.append('}')
    lines.append('''
        .title_link {
            color: inherit;
            text-decoration: inherit;
        }
        body {
            margin: 0;
        }
        .book_title_main_1 {
            position: absolute;
            top: 6cm;
            width: 100%;
            text-align: center;
            font-family: "Times New Roman";
            font-size: 18mm;
            font-weight: bold;
            color: transparent;
        }
        .book_title_main_2 {
            position: absolute;
            top: 8.5cm;
            width: 100%;
            text-align: center;
            font-family: "Times New Roman";
            font-size: 18mm;
            font-weight: bold;
            color: transparent;
        }
        .book_title_tome {
            position: absolute;
            top: 18cm;
            width: 100%;
            text-align: center;
            font-family: "Times New Roman";
            font-size: 9mm;
        }
        .book_title_date_span {
            position: absolute;
            top: 19.5cm;
            width: 100%;
            text-align: center;
            font-family: "Times New Roman";
            font-size: 9mm;
        }
        .burn::before {
            content: attr(data-text);
            text-shadow: 0mm 0mm 0.5mm rgba(52,0,0);
            -webkit-text-stroke-color: rgba(100, 20, 20, 0.5);
            -webkit-text-stroke-width: 0.7mm;
        }
        .burn::after {
            content: attr(data-text);
            position: absolute;
            width:100%;
            left: 0;
            top: 0;
            background-position: inherit;
            background-size: 100%;
            background-image: url('img/cover_background_dark.jpg');
            background-clip: text;
            -webkit-background-clip: text;
            color: transparent;
        }
        .pin_top_left {
            position: absolute;
            top: -6mm;
            left: 0;
            z-index: 10000;
        }
        .pin_top_right {
            position: absolute;
            top: -6mm;
            right: -5mm;
            z-index: 10000;
        }
        .page_number {
            position: absolute;
            width: 100%;
            bottom: 0mm;
            margin: 0;
            text-align: center;
            left: 0;
            font-family: "Times New Roman";
            font-size: 7.5mm;
            font-weight: bold;
            text-shadow: 0mm 0mm 1mm #444, 0mm 0mm 1mm #444, 0mm 0mm 1mm #444;
            color: #999;
            mix-blend-mode: color-dodge;
        }
        .media {
            position: relative;
            display: inline-block;
            margin: 0;
        }
        .media_watermark {
            position: absolute;
            bottom: 3mm;
            right: 3mm;
            text-shadow: 0mm 0mm 1mm black, 0mm 0mm 1mm black, 0mm 0mm 1mm black;
            font-family: "Times New Roman";
            margin: 0;
            color: lightgray;
            z-index: 1;
            font-size: 3mm;
        }
        .poem_author {
            text-shadow: 0 0 1mm rgb(240, 220, 190), 0 0 1mm rgb(240, 220, 190); /* so that the text is visible over drawings. matches background color well */
            font-family: "Times New Roman";
            margin: 0;
            width: 105mm;
            text-align: right;
            color: black;
            font-size: 3mm;
            color: #555;
        }
        .media::after {
          box-shadow: inset 1mm 1mm 1mm 1mm rgba(255,255,255,0.3), inset -1mm -1mm 1mm 1mm rgba(0,0,0,0.1), -1mm -1mm 1mm 1mm rgba(0,0,0,0.15), 1mm 1mm 1mm 1mm rgba(0,0,0,0.25);
          content: '';
          position: absolute;
          top: 0px;
          left: 0;
          width: 100%;
          height: calc(100% - 3px);
        }
        blockquote {
          border-left: 2.75mm;
          font-size: 3.85mm;
          margin: -5mm 10mm;
          margin-bottom: -8mm;
          padding: 3mm 2.75mm;
          max-width: 8.5cm;
          quotes: "\\201C""\\201D""\\2018""\\2019";
        }
        blockquote:before {
          color: #aaa;
          content: open-quote;
          font-size: 12mm;
          line-height: 1mm;
          margin-right: 3mm;
          position: relative;
          top: 3mm;
          left: 3px;
        }
        blockquote::after {
          visibility: hidden;
          content: close-quote;
        }
        blockquote p {
          display: inline;
        }
        .poem {
            font-size: 4.95mm;
            margin: 0;
            margin-left: 15mm;
            text-shadow: 0 0 1mm rgb(240, 220, 190), 0 0 1mm rgb(240, 220, 190); /* so that the text is visible over drawings. matches background color well */
        }
        .poem_card_0 {
            background-image: url('img/poem_background_0.jpg');
            background-size: 100%;
            background-position: bottom right;
            box-shadow: inset 1mm 1mm 1mm 1mm rgba(255,255,255,0.3), inset -1mm -1mm 1mm 1mm rgba(0,0,0,0.1), -1mm -1mm 1mm 1mm rgba(0,0,0,0.15), 1mm 1mm 1mm 1mm rgba(0,0,0,0.25);
            padding: 3mm 0mm 3mm 0mm;
            position: relative;
            margin-top: 2mm;
        }
        .poem_card_1 {
            background-image: url('img/poem_background_1.jpg');
            background-size: 100%;
            background-position: bottom right;
            box-shadow: inset 1mm 1mm 1mm 1mm rgba(255,255,255,0.3), inset -1mm -1mm 1mm 1mm rgba(0,0,0,0.1), -1mm -1mm 1mm 1mm rgba(0,0,0,0.15), 1mm 1mm 1mm 1mm rgba(0,0,0,0.25);
            padding: 3mm 0mm 3mm 0mm;
            position: relative;
            margin-top: 2mm;
        }
        .poem_card_2 {
            background-image: url('img/poem_background_2.jpg');
            background-size: 100%;
            background-position: bottom right;
            box-shadow: inset 1mm 1mm 1mm 1mm rgba(255,255,255,0.3), inset -1mm -1mm 1mm 1mm rgba(0,0,0,0.1), -1mm -1mm 1mm 1mm rgba(0,0,0,0.15), 1mm 1mm 1mm 1mm rgba(0,0,0,0.25);
            padding: 3mm 0mm 3mm 0mm;
            position: relative;
            margin-top: 2mm;
        }
        .poem_card_3 {
            background-image: url('img/poem_background_3.jpg');
            background-size: 100%;
            background-position: bottom right;
            box-shadow: inset 1mm 1mm 1mm 1mm rgba(255,255,255,0.3), inset -1mm -1mm 1mm 1mm rgba(0,0,0,0.1), -1mm -1mm 1mm 1mm rgba(0,0,0,0.15), 1mm 1mm 1mm 1mm rgba(0,0,0,0.25);
            padding: 3mm 0mm 3mm 0mm;
            position: relative;
            margin-top: 2mm;
        }
    ''')
    lines.append('.media_image {')
    lines.append(f'    max-width: {PAGE_INNER_WIDTH_CM}cm;')
    lines.append(f'    box-shadow: -1mm -1mm 1mm 1mm rgba(0,0,0,0.15),1mm 1mm 1mm 1mm rgba(0,0,0,0.25);')
    lines.append('}')
    lines.append('</style>')
    return '\n'.join(lines)

def all_media_have_minimum_size(media_paths):
    for path in media_paths:
        if is_image_path(path) and os.path.getsize(path) < 10000:
            return False
        elif path.endswith('.mp4') and os.path.getsize(path) < 50000:
            return False

    return True

def in_chunks(seq, size):
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]

os.makedirs('./media', exist_ok=True)

print('Loading comments from cache')
comments = []
try:
    with open('comments.dat', 'rb') as f:
        comments = pickle.load(f)
except:
    comments = []

old_comment_ids = set()
for comment in comments:
    old_comment_ids.add(comment.id)

print(f'Loaded {len(comments)} comments from cache.')

before = None
after = None
for comment in comments:
    if before is None or comment.created_utc < before:
        before = int(comment.created_utc)
    if after is None or comment.created_utc > after:
        after = int(comment.created_utc)

print(before, after, time.time())

SKIP_QUERIES = True
USE_PRAW_LAST_ONLY = True

if not SKIP_QUERIES:
    new_comments_ids = []
    if USE_PRAW_LAST_ONLY:
        for comment in REDDIT.redditor('SchnoodleDoodleDo').comments.new(limit=None):
            if comment.id in old_comment_ids:
                continue
            if comment.created_utc <= after:
                break
            new_comments_ids.append(comment.id)
            comments.append(comment)
            print(comment.body)
            print(f'Successfully queried {len(new_comments_ids)} additional comments from PRAW in total.')
    else:
        if after is not None:
            for comment in PSAW_API.search_comments(author='SchnoodleDoodleDo', after=after):
                cid = f't1_{comment.id}'
                new_comments_ids.append(cid)
                print(comment.body)
                print(f'Successfully queried {len(new_comments_ids)} additional comments from PSAW in total.')

        for comment in PSAW_API.search_comments(author='SchnoodleDoodleDo', before=before):
            cid = f't1_{comment.id}'
            new_comments_ids.append(cid)
            print(comment.body)
            print(f'Successfully queried {len(new_comments_ids)} additional comments from PSAW in total.')

        i = 0
        for comment in REDDIT.info(fullnames=new_comments_ids):
            comments.append(comment)
            i += 1
            print(f'Successfully queried {i}/{len(new_comments_ids)} additional comments from PRAW in total.')

    parent_comment_ids = []
    parent_comment_ids_map = dict()

    for comment in comments:
        if comment.parent_id is not None and comment.parent_id.startswith('t1_') and not hasattr(comment, '_parent_comment'):
            parent_comment_ids.append(comment.parent_id)
            parent_comment_ids_map[comment.parent_id[3:]] = comment

    for i, parent_comment in enumerate(REDDIT.info(fullnames=parent_comment_ids)):
        parent_comment_id = parent_comment.id
        child_comment = parent_comment_ids_map[parent_comment_id]
        child_comment._parent_comment = parent_comment
        print(f'Successfully queried {i}/{len(parent_comment_ids)} additional parent comments from PRAW in total.')

pages = []
poems = set() # for catching duplicates

ff = open('other_links.txt', 'w')
ff2 = open('no_media.txt', 'w')
num_new_submissions = 0
for i, comment in enumerate(comments):
    print(f'Processing comment {i+1}/{len(comments)}.')

    if not is_comment_a_poem(comment):
        print('Not a poem. Skipping.')
        print(comment.permalink)
        print(comment.body)
        continue

    if comment._submission is None:
        num_new_submissions += 1

    submission = comment.submission
    url = submission.url
    # should be fetched now
    # we reset some fields that we don't need to prevent blowing up pickled size
    comment._submission._comments_by_id = None
    comment._submission._comments = None

    print(f'Fetched submission: {submission.permalink}')
    print(f'Processing URL: {url}')
    media = []
    if 'v.redd.it' in url:
        print('    VIDEO')
        media += download_reddit_video(submission)
    elif 'i.redd.it' in url or 'i.imgur.com' in url:
        print('    IMAGE')
        media += download_reddit_image(submission)
    elif 'reddit.com/gallery' in url:
        print('    GALLERY')
        media += download_reddit_gallery(submission)
        print(media)
    elif 'gfycat' in url:
        print('    GFYCAT')
        media += download_gfycat_video(submission)
    elif 'imgur' in url and not is_imgur_gallery(submission):
        print('    IMGUR IMAGE')
        media += download_imgur_image(submission)
    elif 'imgur' in url:
        print('    IMGUR GALLERY')
        ff.write(url)
        ff.write('\n')
    else:
        print('    UNKNOWN')
        ff.write(url)
        ff.write('\n')

    if len(media) == 0:
        ff2.write(submission.id + '\n')

    if len(media) > 0 and all_media_have_minimum_size(media):
        page = PoemPage(comment, media)
        poem_key = ''.join(page.poem.verses).replace(' ', '')
        if not page.invalid and len(page.poem.verses) >= 3 and not poem_key in poems:
            poems.add(poem_key)
            pages.append(page)

    # TODO: do something with media

    if (num_new_submissions + 1) % 100 == 0:
        # save every 10 new submissions
        with open('comments.dat', 'wb') as f:
            pickle.dump(comments, f)
        print('pickle dump')

pages.sort(key=lambda x:x.created_utc)

NUM_FIRST_TO_DISCARD = 31 # we know that this many early contributions that are qualified as "poems" are not
pages = pages[NUM_FIRST_TO_DISCARD:]

NUM_PAGES_PER_BOOK = 400 # ~4cm thick book?

for i, pages_chunk in enumerate(in_chunks(pages, NUM_PAGES_PER_BOOK)):
    #last = min(len(pages), i*NUM_PAGES_PER_BOOK+NUM_PAGES_PER_BOOK)
    last = i*NUM_PAGES_PER_BOOK+NUM_PAGES_PER_BOOK # otherwise the last might not get replaced...
    filename = f'Schnoodle Book - Tome {i+1} - Poems {i*NUM_PAGES_PER_BOOK+1}-{last}'
    with open(f'{filename}.html', 'w', encoding='utf-8') as f:
        f.write('<!DOCTYPE html>\n')
        f.write('<html>\n')
        f.write('<head>\n')
        f.write(get_html_header())
        f.write('</head>\n')
        f.write('<body>\n')
        f.write(make_front_cover_page(i+1, pages_chunk[0].created_utc, pages_chunk[-1].created_utc))
        for j, page in enumerate(pages_chunk):
            f.write(page.generate_html(j+1))
        f.write(make_back_cover_page())
        f.write('</body>\n')
        f.write('</html>\n')

with open('comments.dat', 'wb') as f:
    pickle.dump(comments, f)
