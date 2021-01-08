import wikipedia as wiki
import requests
import logging
from typing import List

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)


class WikiClient:

    def __init__(self):
        self.fallback_client = DogImageClient()

    def search(self, breed: str, num_images: int = 3) -> str:
        query = self._prepare_query(breed)
        logging.info(f'Searching for {query}')

        try:
            page = wiki.page(query, auto_suggest=False, preload='images')

            return {
                'images': page.images[:num_images],
                'summary': f'{page.summary[0:400]} ...'
            }
        except:
            images = self.fallback_client.fetch_images(query, num_images)

            if len(images) > 0:
                return {
                    'images': images,
                    'summary': ''
                }
    

        return {
            'images': ['https://upload.wikimedia.org/wikipedia/commons/0/0d/Tumbeasts_sign1.png'],
            'summary': ''
        }

    def _prepare_query(self, breed: str) -> str:
        return breed.lower().replace('_', ' ')


class DogImageClient:

    def __init__(self):
        self.endpoint = 'https://dog.ceo/api'

    def fetch_images(self, query: str, num_images: str = 3) -> List[str]:
        breed_endpoint = f'{self.endpoint}/breed/{query}/images/random/{num_images}'

        data = requests.get(breed_endpoint).json()

        if data['status'] == 'success':
            return data.get('message', [])

        return []

