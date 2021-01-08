from typing import List
import requests


def prepare_query(breed: str) -> str:
    return breed.lower().replace('_', ' ')


class DogImageClient:

    def __init__(self):
        self.endpoint = 'https://dog.ceo/api'

    def fetch_images(self, breed: str, num_images: str = 3) -> List[str]:
        query = prepare_query(breed)
        breed_endpoint = f'{self.endpoint}/breed/{query}/images/random/{num_images}'

        data = requests.get(breed_endpoint).json()

        if data['status'] == 'success':
            return data.get('message', [])

        return []


class WikiClient:

    def __init__(self):
        self.endpoint = 'https://en.wikipedia.org/w/api.php'
    
    def fetch_snippet(self, breed: str):
        query = prepare_query(breed)
        params = {
            'action': 'query',
            'format': 'json',
            'list': 'search',
            'srlimit': 1,
            'srsearch': query
        }
       
        data = requests.get(self.endpoint, params).json()
        results = data.get('query', {}).get('search', [])

        if len(results) > 0:
            return results[0].get('snippet', '')
        
        return ''