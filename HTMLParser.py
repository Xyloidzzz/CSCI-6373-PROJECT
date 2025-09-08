#####################################################
# HTMLParser.py
#
# CSCI 6373 - Information Retrieval and Web Search
# Dr. Zhixiang Chen
# Fall 2025
# Project Part 1
#
# Created: 2025-09-04
# Last Edited: 2025-09-04
#
# Authors:
#   - Aly Gutierrez (ID: 20027057)
#   - Luis Martinez (ID: 20578284)
#   - Tristan Rodriguez (ID: 20499834)
#   - Alfredo Peña (ID: 20217083)
#####################################################

import zipfile
import os
import re

class HTMLParser:
    def __init__(self, zip_path="Jan.zip", folder_name="Jan"):
        self.zip_path = zip_path
        self.folder_name = folder_name
        self.index = {}
        self._build_index()

    def _build_index(self):
        with zipfile.ZipFile(self.zip_path, 'r') as z:
            for file_name in z.namelist():
                if file_name.startswith(self.folder_name) and file_name.endswith(".html"):
                    with z.open(file_name) as f:
                        html_content = f.read().decode("utf-8", errors="ignore")
                        words = self.parse(html_content)
                        base_name = os.path.basename(file_name)
                        for word in words:
                            if word not in self.index:
                                self.index[word] = set()
                            self.index[word].add(base_name)
                            
    def parse(self, html_content):
        text = re.sub(r"<[^>]+>", " ", html_content) # tag killer
        words = re.findall(r"\b\w+\b", text.lower()) # split and lowercase
        return words

    def search(self, term):
        term = term.lower()
        if term in self.index:
            return sorted(self.index[term])
        return []
