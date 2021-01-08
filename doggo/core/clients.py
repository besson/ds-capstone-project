import wikipedia as wiki
import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class WikiClient:

    def search(self, breed: str, number_images: int = 3) -> str:
        query = self._prepare_query(breed)
        logging.info(f'Searching for {query}')

        try:
            page = wiki.page(query, auto_suggest=False, preload='images')

            return {
                'images': page.images[:number_images],
                'summary': f'{page.summary[0:400]} ...'
            }
        except:
            return {
                'images': ['https://upload.wikimedia.org/wikipedia/commons/0/0d/Tumbeasts_sign1.png'],
                'summary': ''
            }

    def _prepare_query(self, breed: str) -> str:
        return breed.lower().replace('_', ' ')

