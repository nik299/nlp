import json
import urllib.request
from tqdm import tqdm


class nboostRank:

    def Rank(self, queries, url_base="http://localhost:8000/cranfield/_search?pretty&q=passage:"):
        final_results = []
        for query in tqdm(queries):
            url_text = url_base + query.replace(' ', '+') + "&size=10"
            with urllib.request.urlopen(url_text) as url:
                data = json.loads(url.read().decode())
            ranks = []
            for rank_dict in data['hits']['hits']:
                ranks.append(int(rank_dict['_id']))
            final_results.append(ranks)
        return final_results
