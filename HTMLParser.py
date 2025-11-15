#####################################################
# HTMLParser.py
#
# CSCI 6373 - Information Retrieval and Web Search
# Dr. Zhixiang Chen
# Fall 2025
#
# Created: 2025-09-04
# Last Edited: 2025-11-15
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
from collections import deque
import posixpath
from pathlib import PurePosixPath

STOPWORDS = {
    'a','an','and','are','as','at','be','but','by','for','if','in','into','is','it',
    'no','not','of','on','or','such','that','the','their','then','there','these','they',
    'this','to','was','will','with','from','we','you','your','i','our','us'
}

# query reformulation parameters
QR_TOP_DOCS = 5
QR_CANDIDATE_KEYWORDS = 20
QR_EXPANSION_TERMS = 5

class HTMLParser:
    def __init__(self, zip_path="rhf.zip", folder_name="rhf"):
        self.zip_path = zip_path
        self.folder_name = folder_name
        self.index = {}
        self.inverted_index = {}
        self.documents = {}
        self.doc_stats = {}
        self.links = {}
        self.titles = {}
        self.snippets = {}
        self.doc_keywords = {}
        self._build_index_from_crawl()
        # self._build_index()
        self.num_docs = len(self.documents)
        self._build_inverted_index()
        self._build_doc_keywords()

    def _build_index(self):
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            for file_name in z.namelist():
                if file_name.startswith(self.folder_name) and (file_name.endswith(".html") or file_name.endswith(".htm")):
                    with z.open(file_name) as f:
                        html_content = f.read().decode("utf-8", errors="ignore")
                    words = self.parse(html_content)

                    rel_path = file_name
                    prefix = self.folder_name.rstrip('/') + '/'
                    if rel_path.startswith(prefix):
                        rel_path = rel_path[len(prefix):]
                    rel_path = rel_path.replace('\\', '/').lstrip('./').lstrip('/')

                    rel_path = rel_path.replace('\\', '/').lstrip('./').lstrip('/')
                    doc_id = os.path.basename(rel_path)  # for display/search
                    self.documents[doc_id] = words
                    self.doc_stats[doc_id] = {'length': len(words)}
                    self.links[doc_id] = rel_path  # full path for the actual link
                    
                    # Extract and store title/anchor text
                    title = self._extract_title(html_content)
                    self.titles[doc_id] = title if title else doc_id
                    # Extract and store a short snippet for display
                    snippet = self._extract_snippet(html_content)
                    self.snippets[doc_id] = snippet if snippet else ''

                    for pos, word in enumerate(words):
                        if word not in self.index:
                            self.index[word] = {}
                        if doc_id not in self.index[word]:
                            self.index[word][doc_id] = []
                        self.index[word][doc_id].append(pos)

    def _build_index_from_crawl(self):
        path = self.folder_name + '/'
        start =  path + 'index.html'
        queue = deque([start])
        seen = set()
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            while queue:
                file_name = queue.popleft()
                if file_name in seen:
                    continue
                seen.add(file_name)
                if file_name.startswith(self.folder_name) and (file_name.endswith(".html") or file_name.endswith(".htm")):
                        with z.open(file_name) as f:
                            html_content = f.read().decode("utf-8", errors="ignore")
                        links = self._extract_links(html_content)
                        path = posixpath.dirname(file_name) + '/'
                        for l in links:
                            queue.append( self._clean_path(path + l))

                        words = self.parse(html_content)
                        rel_path = file_name
                        prefix = self.folder_name.rstrip('/') + '/'
                        if rel_path.startswith(prefix):
                            rel_path = rel_path[len(prefix):]
                        rel_path = rel_path.replace('\\', '/').lstrip('./').lstrip('/')

                        rel_path = rel_path.replace('\\', '/').lstrip('./').lstrip('/')
                        doc_id = os.path.basename(rel_path)  # for display/search
                        self.documents[doc_id] = words
                        self.doc_stats[doc_id] = {'length': len(words)}
                        self.links[doc_id] = rel_path  # full path for the actual link
                        
                        # Extract and store title/anchor text
                        title = self._extract_title(html_content)
                        self.titles[doc_id] = title if title else doc_id
                        # Extract and store a short snippet for display
                        snippet = self._extract_snippet(html_content)
                        self.snippets[doc_id] = snippet if snippet else ''  

                        for pos, word in enumerate(words):
                            if word not in self.index:
                                self.index[word] = {}
                            if doc_id not in self.index[word]:
                                self.index[word][doc_id] = []
                            self.index[word][doc_id].append(pos)
        print(f'DONE: files scanned {len(seen)}')

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
            idf_cache[word] = math.log((self.num_docs + 1) / (df + 1)) + 1.0

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

        # sort posting lists by document id for doc correlation computation later
        for word_entry in self.inverted_index.values():
            word_entry['docs'] = dict(sorted(word_entry['docs'].items()))

    def _build_doc_keywords(self):
        """
            for each document create sorted list of term tfidf pairs then populate doc_keywords
        """
        for doc_id in self.documents:
            self.doc_keywords[doc_id] = []

        for word, word_entry in self.inverted_index.items():
            for doc_id, posting in word_entry['docs'].items():
                self.doc_keywords[doc_id].append((word, posting['norm_tfidf']))

        # sort each doc's keywords by tfidf descending
        for doc_id in self.doc_keywords:
            self.doc_keywords[doc_id].sort(key=lambda x: -x[1])

    def compute_term_correlation(self, term1, term2):
        """
            compute correlation between two terms using their normalized tfidf values
            go through posting lists in parallel and sum products for common documents
        """
        entry1 = self.inverted_index.get(term1)
        entry2 = self.inverted_index.get(term2)

        if not entry1 or not entry2:
            return 0.0

        docs1 = entry1['docs']
        docs2 = entry2['docs']

        # find common documents and sum tfidf products
        common_docs = set(docs1.keys()) & set(docs2.keys())

        correlation = 0.0
        for doc_id in common_docs:
            tfidf1 = docs1[doc_id]['norm_tfidf']
            tfidf2 = docs2[doc_id]['norm_tfidf']
            correlation += tfidf1 * tfidf2

        return correlation

    def extract_keywords_from_docs(self, doc_ids, top_k = QR_CANDIDATE_KEYWORDS):
        """
            extract top keywords from a list of documents
            aggregate tfidf scores across docs and return top k by average score
        """
        term_scores = {}

        for doc_id in doc_ids:
            if doc_id not in self.doc_keywords:
                continue

            for term, tfidf in self.doc_keywords[doc_id]:
                if term in STOPWORDS:
                    continue

                if term not in term_scores:
                    term_scores[term] = []
                term_scores[term].append(tfidf)

        # compute average tfidf across documents
        avg_scores = []
        for term, scores in term_scores.items():
            avg_scores.append((term, sum(scores) / len(scores)))

        # sort by average score descending
        avg_scores.sort(key = lambda x: -x[1])

        return avg_scores[:top_k]

    def find_correlated_keywords(self, query_terms, candidate_keywords, top_n = QR_EXPANSION_TERMS):
        """
            find the keywords most correlated with query terms
            compute average correlation of each with all query terms
            return top n  terms by correlation score
        """
        keyword_correlations = []

        for candidate_term, _ in candidate_keywords:
            if candidate_term in query_terms:
                continue

            # compute correlation with each query term
            correlations = []
            for query_term in query_terms:
                corr = self.compute_term_correlation(query_term, candidate_term)
                correlations.append(corr)

            if correlations:
                avg_corr = sum(correlations) / len(correlations)
                keyword_correlations.append((candidate_term, avg_corr))

        # sort by correlation descending
        keyword_correlations.sort(key=lambda x: -x[1])

        return [term for term, _ in keyword_correlations[:top_n]]

    def _extract_links(self, html_content):
        # returns a list of urls in html
        links = re.findall(r'href\s*=\s*["\']([^"\']+)["\']', html_content, flags=re.IGNORECASE)
        return links
    
    def _extract_title(self, html_content):
        # Extract title from HTML content
        # First try to get <title> tag
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, flags=re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
            # Remove HTML tags from title if any
            title = re.sub(r'<[^>]+>', '', title).strip()
            # Remove [rec.humor.funny] or similar bracketed text anywhere in the title
            title = re.sub(r'\s*\[[\w\.]+\]', '', title).strip()
            if title:
                return title
        
        # Fallback to first <h1> tag
        h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', html_content, flags=re.IGNORECASE | re.DOTALL)
        if h1_match:
            h1_text = h1_match.group(1).strip()
            h1_text = re.sub(r'<[^>]+>', '', h1_text).strip()
            # Remove [rec.humor.funny] or similar bracketed text anywhere in the title
            h1_text = re.sub(r'\s*\[[\w\.]+\]', '', h1_text).strip()
            if h1_text:
                return h1_text
        
        # If no title or h1, return None
        return None

    def _extract_snippet(self, html_content, maxlen=160):
        if not html_content:
            return ''
        m = re.search(r'<meta\s+name=["\']description["\']\s+content=["\'](.*?)["\']', html_content, flags=re.IGNORECASE | re.DOTALL)
        if m:
            desc = m.group(1).strip()
        else:
            p = re.search(r'<p[^>]*>(.*?)</p>', html_content, flags=re.IGNORECASE | re.DOTALL)
            if p:
                desc = re.sub(r'<[^>]+>', '', p.group(1)).strip()
            else:
                body = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', html_content, flags=re.IGNORECASE | re.DOTALL)
                text = re.sub(r'<[^>]+>', ' ', body)
                text = re.sub(r'\s+', ' ', text).strip()
                title = self._extract_title(html_content) or ''
                if title and title in text:
                    text = text.replace(title, '')
                desc = text
        desc = re.sub(r'\s*\[[\w\.\- ]+\]\s*', ' ', desc).strip()
        desc = re.sub(r'\s+', ' ', desc)

        if not desc:
            return ''

        if len(desc) <= maxlen:
            return desc
        end = desc.rfind('.', 0, maxlen)
        if end != -1 and end >= int(maxlen*0.5):
            return desc[:end+1].strip()
        cut = desc.rfind(' ', 0, maxlen)
        if cut == -1:
            return desc[:maxlen].strip() + '...'
        return desc[:cut].strip() + '...'
    
    def _idf(self, term):
        # compute idf using df from inverted index and num_docs
        entry = self.inverted_index.get(term)
        if not entry:
            return 0.0
        df = entry['df']
        return math.log((self.num_docs + 1) / (df + 1)) + 1.0
    
    def _tokenize_query(self, text):
        # split, lowercase, and remove stop-words
        tokens = re.findall(r"\b\w+\b", text.lower())
        return [t for t in tokens if t not in STOPWORDS]

    def _clean_path(self, path):
        parts = []
        for part in PurePosixPath(path).parts:
            if part == "..":
                if parts:
                    parts.pop()
            elif part != ".":
                parts.append(part)
        return "/".join(parts)

    def parse(self, html_content):
        text = re.sub(r"<[^>]+>", " ", html_content, flags=re.IGNORECASE) # tag killer
        return self._tokenize_query(text)
    
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
    
    def search(self, query):
        q = query.strip()
        if not q:
            return []

        # boolean search
        if re.search(r"\b(and|or|but)\b", q.lower()):
            return self.boolean_search(q)

        # quoted phrase
        m = re.search(r'"([^"]+)"', q)
        if m:
            phrase = m.group(1)
            return self.phrase_search(phrase) # TODO: implement phrase search

        # DEFAULT BEHAVIOR -> vector-space ranking
        return self.vector_search(q)

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
    
    def phrase_search(self, phrase):
        terms = self._tokenize_query(phrase)

        if not terms:
           return []

        if len(terms) == 1:
            return sorted(self.get_doc_names(terms[0]))

        # find common documents containing all terms
        docs = None
        for term in terms:
            term_docs = self.get_doc_names(term)
            if not term_docs:
                return []
            
            if docs is None:
                docs = term_docs
            else:
                docs &= term_docs

        if not docs:
            return []
        
        m_docs = []

        for d_name in docs:
            if self.hasConTerms(d_name,terms):
                m_docs.append(d_name)
        
        return sorted(m_docs)

    def hasConTerms(self, d_name, terms):
        f_term = terms[0]
        f_term_entry = self.inverted_index.get(f_term)
        if not f_term_entry or d_name not in f_term_entry['docs']:
            return False

        f_pos = f_term_entry['docs'][d_name]['positions']

        # Check positions of first term and see if subsequent terms appear in consecutive positions
        for start in f_pos:
            found = True

            for i, term in enumerate(terms[1:], 1):
                exp_pos = start + i
                term_entry = self.inverted_index.get(term)

                if not term_entry or d_name not in term_entry['docs']:
                    found = False
                    break

                term_positions = term_entry['docs'][d_name]['positions']

                if exp_pos not in term_positions:
                    found = False
                    break
            if found:
                return True
        return False
    
    def vector_search(self, query, top_k=None):
        # tokenize query
        terms = self._tokenize_query(query)
        if not terms:
            return []

        # compute query term frequencies
        qtf = {}
        for t in terms:
            qtf[t] = qtf.get(t, 0) + 1

        # compute query weights = log tf * idf
        qw = {}
        for t, f in qtf.items():
            tf = 0.0 if f <= 0 else (1.0 + math.log(f))
            idf = self._idf(t)
            w = tf * idf
            if w > 0:
                qw[t] = w

        if not qw:
            return []

        # compute query norm
        qnorm_sq = sum(w*w for w in qw.values())
        qnorm = math.sqrt(qnorm_sq) if qnorm_sq > 0 else 1.0

        # accumulate scores using normalized doc weights from postings
        # doc -> score numerator = sum w_tq * norm_w_td
        scores = {}
        for t, w_tq in qw.items():
            entry = self.inverted_index.get(t)
            if not entry:
                continue
            for doc_name, posting in entry['docs'].items():
                contrib = w_tq * posting.get('norm_tfidf', 0.0)
                if contrib != 0.0:
                    scores[doc_name] = scores.get(doc_name, 0.0) + contrib

        if not scores:
            return []

        # divide by query norm to complete cosine
        for d in scores:
            scores[d] = scores[d] / qnorm

        # sort by descending score, then doc name
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        if top_k is not None:
            ranked = ranked[:top_k]

        # return only doc names in ranked order
        return [doc for doc, _ in ranked]
    