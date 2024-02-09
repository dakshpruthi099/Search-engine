import nltk
from bs4 import BeautifulSoup
import re
import numpy as np
from collections import defaultdict, Counter
from nltk.corpus import stopwords

# Commented out NLTK download lines
# nltk.download('punkt')
# nltk.download('stopwords')

# Define a TrieNode class to represent nodes in the trie data structure
class TrieNode:
    def __init__(self, letter):
        self.letter = letter
        self.is_end_of_word = False
        self.children = {}  # Dictionary to store child nodes
        self.data = defaultdict(float)  # Dictionary to store data associated with the node
        self.words = set()  # Set to store words associated with the node

    # Compress method to compress the trie by merging single-child nodes
    def compress(self):
        if len(self.children) == 1 and not self.is_end_of_word:
            (key, child) = next(iter(self.children.items()))
            new_letter = self.letter + key
            self.letter = new_letter
            self.is_end_of_word = child.is_end_of_word
            self.children = child.children
            self.data = child.data
            self.compress()

# Define the Trie class for implementing autocomplete and search functionality
class Trie:
    def __init__(self):
        self.root = TrieNode("")  # Initialize the root node

    # Insert a word into the trie with associated data
    def insert(self, word, document_id, score):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode(char)
            node = node.children[char]
            node.words.add(word)
        node.is_end_of_word = True
        node.data[document_id] += score

    # Compress the trie to optimize storage
    def compress(self):
        for child in self.root.children.values():
            child.compress()

    # Autocomplete function to suggest words based on a given prefix
    def autocomplete(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []  # No suggestions found for the given prefix
            node = node.children[char]
        return list(node.words)  # Return the words associated with the node

    # Search function to retrieve documents based on a query
    def search(self, query, documents, tf_idf_scores):
        query = query.lower()
        query_tokens = query.split()

        # Exact Phrase Match
        exact_match_documents = []
        for doc, tokens in documents.items():
            content = ' '.join(tokens).lower()  # Reconstruct the content from tokens
            if query in content:
                exact_match_documents.append((doc, 1.0))  # Assign a high score for exact matches

        if exact_match_documents:
            return exact_match_documents

        # Individual Word Match with TF-IDF Ranking
        document_scores = defaultdict(float)
        for token in query_tokens:
            node = self.root
            for char in token:
                if char not in node.children:
                    break
                node = node.children[char]
            if node.is_end_of_word:
                for doc_id in node.data:
                    # Update the relevance score based on matching terms and their TF-IDF values
                    relevance_score = 0
                    for query_token in query_tokens:
                        tf_idf_value = tf_idf_scores[doc_id].get(query_token, 0)
                        relevance_score += node.data[doc_id] * tf_idf_value
                    document_scores[doc_id] += relevance_score

        return sorted(document_scores.items(), key=lambda x: x[1], reverse=True)

# Preprocess and tokenize text
def preprocess_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [token for token in tokens if token not in stop_words]

# Calculate TF-IDF scores for documents
def calculate_tf_idf(documents):
    N = len(documents)
    df = defaultdict(int)
    tf = defaultdict(Counter)

    for doc_id, tokens in documents.items():
        for token in set(tokens):
            df[token] += 1
        tf[doc_id] = Counter(tokens)

    tf_idf = defaultdict(dict)
    for doc_id, counts in tf.items():
        for token, count in counts.items():
            tf_idf[doc_id][token] = (count / len(counts)) * np.log(N / df[token])
    return tf_idf

# Get autocomplete suggestions for a given query
def get_autocomplete_suggestions(trie, query):
    words = query.split()
    if not words:
        return []
    last_word = words[-1]
    suggestions = trie.autocomplete(last_word)
    return [' '.join(words[:-1] + [suggestion]) for suggestion in suggestions]

# Main function
def main():
    trie = Trie()  # Create a Trie data structure
    documents = defaultdict(list)  # Dictionary to store tokenized documents
    file_paths = [
        "./Webpages/Checkmate - Wikipedia.html",
        "./Webpages/Chess - Wikipedia.html",
        "./Webpages/Chess piece - Wikipedia.html",
        "./Webpages/Hindi - Wikipedia.html",
        "./Webpages/Kanye West - Wikipedia.html",
        "./Webpages/Kendrick Lamar - Wikipedia.html",
        "./Webpages/List of chess openings - Wikipedia.html",
        "./Webpages/Magnus Carlsen - Wikipedia.html",
        "./Webpages/Seedhe Maut - Wikipedia.html",
        "./Webpages/World Chess Championship - Wikipedia.html"
    ]

    # Process and tokenize the text from HTML documents
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            text = soup.get_text()
            tokens = preprocess_and_tokenize(text)
            documents[file_path] = tokens

    tf_idf_scores = calculate_tf_idf(documents)  # Calculate TF-IDF scores for documents

    # Insert tokens into the Trie and compress it for efficient storage
    for doc_id, tokens in documents.items():
        for token in tokens:
            trie.insert(token, doc_id, tf_idf_scores[doc_id][token])

    trie.compress()  # Compress the Trie

    while True:
        query = input("Enter search term or prefix (or type 'exit' to quit): ").lower()
        if query.lower() == 'exit':
            break

        if not query:
            continue

        query_tokens = query.split()
        last_word = query_tokens[-1]

        if len(query_tokens) > 1:
            suggestions = get_autocomplete_suggestions(trie, last_word)
            print("Autocomplete suggestions for last word:", suggestions)
        else:
            suggestions = get_autocomplete_suggestions(trie, query)
            print("Autocomplete suggestions:", suggestions)

        # Search directly with the given query
        results = trie.search(query, documents, tf_idf_scores)
        if results:
            for doc_id, score in results:
                print(f"Document found: {doc_id.split('/')[-1]}")
        else:
            print("No documents found.")

if __name__ == "__main__":
    main()