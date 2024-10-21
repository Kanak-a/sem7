class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.documents = {}

    def add_document(self, doc_id, text):
        # Store the document
        self.documents[doc_id] = text
        # Tokenize the text
        terms = text.lower().split()
        for term in terms:
            if term not in self.index:
                self.index[term] = set()
            self.index[term].add(doc_id)

    def search(self, query):
        # Tokenize the query
        query_terms = query.lower().split()
        # Retrieve document IDs for each term in the query
        result_sets = []
        for term in query_terms:
            if term in self.index:
                result_sets.append(self.index[term])
            else:
                result_sets.append(set())

        # Get the intersection of all result sets
        if result_sets:
            result_docs = set.intersection(*result_sets) if len(result_sets) > 1 else result_sets[0]
        else:
            result_docs = set()

        return result_docs

# Example usage
if __name__ == "__main__":
    # Create an inverted index instance
    inverted_index = InvertedIndex()

    # Add some documents
    inverted_index.add_document(1, "The cat in the hat.")
    inverted_index.add_document(2, "A cat is a fine pet.")
    inverted_index.add_document(3, "Dogs are also great pets.")
    inverted_index.add_document(4, "The dog in the fog.")

    # Search for documents containing specific terms
    query1 = "cat"
    print(f"Documents containing '{query1}':", inverted_index.search(query1))

    query2 = "the"
    print(f"Documents containing '{query2}':", inverted_index.search(query2))

    query3 = "cat dog"
    print(f"Documents containing '{query3}':", inverted_index.search(query3))
