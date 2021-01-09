import wikipedia as wiki
import requests
import logging
from typing import List

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)


class WikiClient:

    def __init__(self):
        self.image_client = DogImageClient()

    def search(self, breed: str, num_images: int = 3) -> str:
        wiki_query = self._prepare_query(breed)
        img_query = self._prepare_img_query(breed)
        logging.info(f'Searching for {wiki_query} in wiki and {img_query} in dog.ceo')

        try:
            images = self.image_client.fetch_images(img_query, num_images)
            page = wiki.page(wiki_query, auto_suggest=False)
            img_src = 'dog.ceo API'

            if len(images) == 0:
                images = page.images[:num_images]
                img_src = 'Wikipedia API'
           
            return {
                'images': images,
                'summary': f'{page.summary[0:400]} ... (by wikipedia)',
                'image_src': img_src
            }
        except:
            if len(images) > 0:
                return {
                    'images': images,
                    'summary': '',
                    'image_src': img_src
                }
    

        return {
            'images': ['https://upload.wikimedia.org/wikipedia/commons/0/0d/Tumbeasts_sign1.png'],
            'summary': '',
            'image_src': ''
        }


    def _prepare_query(self, breed: str) -> str:
        return breed.lower().strip()

    def _prepare_img_query(self, breed: str) -> str:
        return ''.join(breed.lower().replace('dog', '').split(' '))


class DogImageClient:

    def __init__(self):
        self.endpoint = 'https://dog.ceo/api'

    def fetch_images(self, query: str, num_images: str = 3) -> List[str]:
        breed_endpoint = f'{self.endpoint}/breed/{query}/images/random/{num_images}'

        data = requests.get(breed_endpoint).json()

        if data['status'] == 'success':
            return data.get('message', [])

        return []

