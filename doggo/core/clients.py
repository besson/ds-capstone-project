import wikipedia as wiki

class WikiClient:

    def search(self, breed: str, number_images: int = 3) -> str:
        query = self._prepare_query(breed)
        page = wiki.page(query, preload='images')

        if not page:
            return {}

        return {
            'images': page.images[:number_images],
            'url': page.url,
            'summary': f'{page.summary[0:400]} ...'
        }

    def _prepare_query(self, breed: str) -> str:
        return breed.lower().replace('_', ' ')

