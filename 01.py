import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # 1. Tokenization
    tokens = word_tokenize(text)

    # 2. Convert to lowercase
    tokens = [word.lower() for word in tokens]

    # 3. Remove punctuation
    tokens = [word for word in tokens if word not in string.punctuation]

    # 4. Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # 5. Stemming
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens]
    # Sample text document
text_document = """
Natural language processing (NLP) is a subfield of artificial intelligence (AI)
that focuses on the interaction between computers and humans through natural language.
The ultimate goal of NLP is to read, decipher, understand, and make sense of human
language in a valuable way.
"""

# Preprocess the text
processed_text = preprocess_text(text_document)
print("Processed Text:", processed_text)