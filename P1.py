from gensim.downloader import load
#To install gensim library
#pip install gensim
# Load the pre-trained GloVe model (50 dimensions)
print("Loading pre-trained GloVe model (50 dimensions)...")
model = load("glove-wiki-gigaword-50")

# Function to perform vector arithmetic and analyze relationships
def ewr():
    # Example 1: Analogy king - man + woman
    result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
    print("\nking - man + woman = ?", result[0][0])
    print("similarity:", result[0][1])

    # Example 2: Analogy paris - france + italy
    result = model.most_similar(positive=['paris', 'italy'], negative=['france'], topn=1)
    print("\nparis - france + italy = ?", result[0][0])
    print("similarity:", result[0][1])

    # Example 3: Top 5 similar words to 'programming'
    result = model.most_similar(positive=['programming'], topn=5)
    print("\nTop 5 words similar to 'programming':")
    for word, similarity in result:
        print(word, similarity)

# Run the function
ewr()
