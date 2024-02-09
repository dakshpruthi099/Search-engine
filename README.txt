CS 600
Daksh Pruthi

Project Description: Search Engine with Autocomplete

This project is an implementation of a search engine with autocomplete functionality. It allows users to search for documents based on keywords and provides autocomplete suggestions as the user types. The project uses a Trie data structure to efficiently store and retrieve words and their associated data. It also incorporates TF-IDF (Term Frequency-Inverse Document Frequency) ranking to rank search results based on relevance.

Algorithms and Data Structures:

TrieNode Class: The TrieNode class represents nodes in the Trie. Each node contains information about a letter in a word, whether it's the end of a word, children nodes, data associated with the node (used for ranking), and a set of words associated with the node. The Trie is used for storing and efficiently retrieving words and their data.

Trie Class: The Trie class is the main data structure that uses TrieNode objects. It supports operations like inserting words with associated data, compressing the Trie to optimize storage, autocomplete suggestions based on a given prefix, and searching for documents based on a query.

TF-IDF Ranking:

calculate_tf_idf Function: This function calculates the TF-IDF scores for all documents in the corpus. It uses the Term Frequency (TF) and Inverse Document Frequency (IDF) to assign a relevance score to each term in each document. TF-IDF is commonly used in information retrieval to rank documents based on keyword relevance.

Execution:
Inserting and Indexing: The program starts by reading HTML documents, preprocessing and tokenizing the text, and then inserting the tokens into the Trie. TF-IDF scores are calculated for each document.

Autocomplete: As the user types a query, the program suggests completions based on the Trie. It provides suggestions for both single words and phrases.

Search: When the user submits a query, the program searches for documents containing the query terms. It first looks for exact matches and then ranks documents using TF-IDF scores.

Boundary Conditions:
The program handles boundary conditions such as empty queries and queries with no matching documents gracefully, providing appropriate feedback to the user.