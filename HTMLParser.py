#####################################################
# HTMLParser.py
#
# CSCI 6373 - Information Retrieval and Web Search
# Dr. Zhixiang Chen
# Fall 2025
# Project Part 1
#
# Created: 2025-09-04
# Last Edited: 2025-10-09
#
# Authors:
#   - Ariana Gutierrez
#   - Luis Martinez
#   - Tristan Rodriguez
#   - Alfredo Peña
#####################################################

import zipfile
import os
import re
import math

STOPWORDS = {
    'a','an','and','are','as','at','be','but','by','for','if','in','into','is','it',
    'no','not','of','on','or','such','that','the','their','then','there','these','they',
    'this','to','was','will','with','from','we','you','your','i','our','us'
}

class HTMLParser:
    def __init__(self, zip_path="Jan.zip", folder_name="Jan"):
        self.zip_path = zip_path
        self.folder_name = folder_name
        self.index = {}
        self.inverted_index = {}
        self.documents = {}
        self.doc_stats = {}
        self.links = {}
        self._build_index()
        self._build_inverted_index()

    def _build_index(self):
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            for file_name in z.namelist():
                if file_name.startswith(self.folder_name) and file_name.endswith(".html"):
                    with z.open(file_name) as f:
                        html_content = f.read().decode("utf-8", errors="ignore")
                        words = self.parse(html_content)
                        base_name = os.path.basename(file_name)
                        self.doc_stats[base_name] = {'length': len(words)}
                        self.documents[base_name] = words
                        self.links[base_name] = self._extract_links(html_content)

                        for pos, word in enumerate(words):
                            if word not in self.index:
                                self.index[word] = {}
                            if base_name not in self.index[word]:
                                self.index[word][base_name] = []
                            self.index[word][base_name].append(pos)

    def _build_inverted_index(self):
        """
        inverted_index: {
            'term': {
                'df': int,
                'docs': {
                    doc_id: {
                        'freq': int,
                        'tfidf': float,
                        'norm_tfidf': float,
                        'positions': list[int]
                    }
                }
            }
        }
        """
        N = len(self.documents)
        
        # aggregate freq and positions using base_name ids
        for doc_name, words in self.documents.items():
            for pos, word in enumerate(words):
                if word not in self.inverted_index:
                    self.inverted_index[word] = {'df': 0, 'docs': {}}
                word_entry = self.inverted_index[word]
                if doc_name not in word_entry['docs']:
                    word_entry['docs'][doc_name] = {'freq': 0, 'tfidf': 0.0, 'norm_tfidf': 0.0, 'positions': []}
                    word_entry['df'] += 1
                posting = word_entry['docs'][doc_name]
                posting['freq'] += 1
                posting['positions'].append(pos)

        # compute preliminary tf-idf
        idf_cache = {}
        for word, word_entry in self.inverted_index.items():
            df = word_entry['df']
            idf_cache[word] = math.log((N + 1) / (df + 1)) + 1.0

        # raw tf-idf with log tf
        doc_norm = {doc: 0.0 for doc in self.documents}
        for word, word_entry in self.inverted_index.items():
            idf = idf_cache[word]
            for doc_name, posting in word_entry['docs'].items():
                freq = posting['freq']
                tf = 0.0 if freq <= 0 else (1.0 + math.log(freq))
                w = tf * idf
                posting['tfidf'] = w
                doc_norm[doc_name] += w * w

        # normalize per doc
        for doc_name in doc_norm:
            doc_norm[doc_name] = math.sqrt(doc_norm[doc_name]) if doc_norm[doc_name] > 0 else 1.0

        for word_entry in self.inverted_index.values():
            for doc_name, posting in word_entry['docs'].items():
                posting['norm_tfidf'] = posting['tfidf'] / doc_norm[doc_name]
                
    def _extract_links(self, html_content):
        # returns a list of urls in html
        links = re.findall(r'href\s*=\s*["\']([^"\']+)["\']', html_content, flags=re.IGNORECASE)
        return links

    def parse(self, html_content):
        text = re.sub(r"<[^>]+>", " ", html_content, flags=re.IGNORECASE) # tag killer
        words = re.findall(r"\b\w+\b", text.lower()) # split and lowercase
        words = [w for w in words if w not in STOPWORDS] # remove stop-words
        return words

    def search(self, query):
        pattern = r'\b\w+\b\s+(and|but|or)\s+\b\w+\b' # pattern for boolean query
        terms = query.lower().split()
        if not terms:
            return []
        
        if re.findall(pattern, query):
            return self.boolean_search(query)
        
        firstword = terms[0]
        if firstword not in self.index:
            return []

        results = []
        for file, postions in self.index[firstword].items():
            words = self.documents[file]
            for pos in postions:
                if words[pos:pos + len(terms)] == terms:
                    results.append(file)
                    break
        return sorted(results)
    
    def get_indexed_words(self):
        return sorted(self.index.keys())
    
    def get_doc_names(self, word):
        """
            gets document set for a given word from the inverted index.
            returns an empty set if not found.
        """
        entry = self.inverted_index.get(word)
        if entry is None:
            return set()
        return set(entry['docs'].keys())

    def boolean_search(self, query):
        words = re.split(r'\s+(and|or|but)\s+', query.lower())
        results = self.get_doc_names(words[0])
        for i in range(1, len(words), 2):
            operator = words[i]
            right = self.get_doc_names(words[i+1])
            if operator == 'and':
                results &= right
            elif operator == 'or':
                results |= right
            elif operator == 'but':
                results -= right
        # temp sorted list of base_name ids, later we gone rank aight
        return sorted(results)