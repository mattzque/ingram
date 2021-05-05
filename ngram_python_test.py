from pprint import pprint
from ngram import NGram


def search(docs, queries):
    index = NGram(N=2)
    for i, doc in enumerate(docs):
        print(f'{i} -> {doc}')
        index.add(doc)

    for q in queries:
        res = index.search(q, 0.0)
        print(f'{q} -> {res}')


search(['ABCD'], ['AB', 'ABCD', 'CD'])
print('')
search(['ABC'], ['A', 'B', 'C', 'AB', 'ABC'])
