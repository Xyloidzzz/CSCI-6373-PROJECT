#####################################################
# HTMLParser.py
#
# CSCI 6373 - Information Retrieval and Web Search
# Dr. Zhixiang Chen
# Fall 2025
# Project Part 1
#
# Created: 2025-09-04
# Last Edited: 2025-09-07
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

class HTMLParser:
    def __init__(self, zip_path="Jan.zip", folder_name="Jan"):
        self.zip_path = zip_path
        self.folder_name = folder_name
        self.index = {}
        self.inverted_index = {}
        self.documents = {}
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
                        self.documents[base_name] = words
                        for pos, word in enumerate(words):
                            if word not in self.index:
                                self.index[word] = {}
                            if base_name not in self.index[word]:
                                self.index[word][base_name] = [] 
                            self.index[word][base_name].append(pos)

    def _build_inverted_index(self):
        """
        inverted_index: {
            {
                'term':str,
                'df':int,
                'postings':{
                    {'freq':int, 'tfidf':float, 'positions':List[int]}
                }
            }
        }
        """
        doc_lengths = {}
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            N = len(z.namelist())
            for file_name in z.namelist():
                with z.open(file_name) as f:
                    html_content = f.read().decode("utf-8", errors="ignore")
                    words = self.parse(html_content)
                    doc_lengths[file_name] = len(words)
                    for pos, word in enumerate(words):
                        if word not in self.inverted_index:
                            self.inverted_index[word] = {
                                'df': 0,
                                'docs':{} # doc_id ->{freq, tfdif, positions}
                            }
                        word_docs = self.inverted_index[word]
                        docs = word_docs['docs']

                        if file_name not in docs:
                            word_docs['df'] += 1
                            docs[file_name] = {'freq':0, 'tfidf':0, 'positions':[]}
                        
                        docs = docs[file_name]
                        docs['freq'] += 1
                        docs['positions'].append(pos)

        for word, word_entry in self.inverted_index.items():
            df = word_entry['df']
            idf = math.log(N / (df + 1), 2) + 1

            for doc_name, doc_entry in word_entry['docs'].items():
                freq = doc_entry['freq']
                tf = freq / doc_lengths[doc_name]
                doc_entry['tfidf'] = tf * idf

    def parse(self, html_content):
        text = re.sub(r"<[^>]+>", " ", html_content) # tag killer
        words = re.findall(r"\b\w+\b", text.lower()) # split and lowercase
        return words

    def search(self, query):
        terms = query.lower().split()
        if not terms:
            return []
        
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
    
    def invert_index(self):
        """
        index: {word:{doc_name:[idx1, idx2,...],...},...} ->
        inverted_index: [
            {
                'index':int,
                'term':str,
                'df':int,
                'postings':[
                    {'doc_id':int , 'freq':int, 'tfidf':float, 'positions':List[int]}
                ]
            }
        ]
        """
        for word in self.index:
            print(word, self.index[word], self.inverted_index.keys())
            print(self.index[word].keys())
            if word not in self.inverted_index.keys():
                self.inverted_index[word] = (1)
            else:
                self.inverted_index[word] += 1

            break

        print(self.inverted_index)

