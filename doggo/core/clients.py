import wikipedia as wiki
import requests
import logging
from typing import List

NOT_FOUND_IMG = 'https://upload.wikimedia.org/wikipedia/commons/0/0d/Tumbeasts_sign1.png'
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)


def prepare_img_query(breed: str) -> str:
    """
    Prepare query for image search
    INPUT
        breed - name of dog breed, str
    OUTPUT
        query: formatted breed query, str
    """
    return ''.join(breed.lower().replace('dog', '').split(' '))


def prepare_wiki_query(breed: str) -> str:
    """
    Prepare query for wikipedia search
    INPUT
        breed - name of dog breed, str
    OUTPUT
        query: formatted breed query, str
    """
    return breed.lower().strip()


class WikiClient:
    """ Wikipedia client to provide data such as description and photos of a dog breed"""

    def __init__(self):
        self.image_client = DogImageClient()

    def search(self, breed: str, num_images: int = 3):
        """
        Search on wikipedia and dog.ceo API to find information and photos of the provided dog breed
        INPUT
            breed - name of dog breed, str
            num_images - number of photos to be loaded, str
        OUTPUT
            result: object with images and description of the provided dog breed, dict
        """
        global images, img_src
        wiki_query = prepare_wiki_query(breed)
        img_query = prepare_img_query(breed)
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
            'images': [NOT_FOUND_IMG],
            'summary': '',
            'image_src': ''
        }


class DogImageClient:
    """ Http client of dog.ceo API: https://dog.ceo/dog-api """

    def __init__(self):
        self.endpoint = 'https://dog.ceo/api'

    def fetch_images(self, query: str, num_images: int = 3) -> List[str]:
        """
        Fetch random photos for the provided query which is a dog breed
        INPUT
            query - formatted name of dog breed, str
            num_images - number of photos to be loaded, str
        OUTPUT
            result: object with images of the provided dog breed, dict
        """
        breed_endpoint = f'{self.endpoint}/breed/{query}/images/random/{num_images}'
        data = requests.get(breed_endpoint).json()

        if data['status'] == 'success':
            return data.get('message', [])

        return []
