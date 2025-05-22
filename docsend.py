from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from PIL import Image
import requests
from bs4 import BeautifulSoup


class DocSend:

    def __init__(self, doc_id: str):
        self.doc_id = doc_id.rpartition('/')[-1]
        self.url = f'https://docsend.com/view/{doc_id}'
        self.s = requests.Session()
        # Add browser-like headers to avoid bot detection
        self.s.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0',
        })
        self.auth_token = None
        self.pages = 0
        self.image_urls = []
        self.images = []

    def fetch_meta(self):
        r = self.s.get(self.url)
        r.raise_for_status()
        soup = BeautifulSoup(r.content, 'html.parser')
        auth_element = soup.find('input', attrs={'name': 'authenticity_token'})
        if auth_element:
            self.auth_token = str(auth_element.get('value'))  # type: ignore[attr-defined]

        # Find the last document thumb container and get page number
        thumb_containers = soup.find_all(class_='document-thumb-container')
        if thumb_containers:
            self.pages = int(thumb_containers[-1].get('data-page-num'))  # type: ignore[attr-defined]
        else:
            self.pages = 0

    def authorize(self, email: str, passcode: str | None = None):
        form = {
            'utf8': '✓',
            '_method': 'patch',
            'authenticity_token': self.auth_token,
            'link_auth_form[email]': email,
            'link_auth_form[passcode]': passcode,
            'commit': 'Continue',
        }
        f = self.s.post(self.url, data=form)
        f.raise_for_status()

    def fetch_images(self):
        pool = ThreadPoolExecutor(self.pages)
        self.images = list(pool.map(self._fetch_image, range(1, self.pages + 1)))

    def _fetch_image(self, page: int):
        meta = self.s.get(f'{self.url}/page_data/{page}')
        meta.raise_for_status()
        data = self.s.get(meta.json()['imageUrl'])
        data.raise_for_status()
        rgba = Image.open(BytesIO(data.content))  # type: ignore[attr-defined]
        rgb = Image.new('RGB', rgba.size, (255, 255, 255))  # type: ignore[attr-defined]
        rgb.paste(rgba)  # type: ignore[attr-defined]
        return rgb

    def save_pdf(self, name: str):
        self.images[0].save(  # type: ignore[attr-defined]
            name,
            format='PDF',
            append_images=self.images[1:],
            save_all=True
        )

    def save_images(self, name: str):
        path = Path(name)
        path.mkdir(exist_ok=True)
        for page, image in enumerate(self.images, start=1):
            image.save(path / f'{page}.png', format='PNG')  # type: ignore[attr-defined]
