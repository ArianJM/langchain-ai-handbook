from datasets import load_dataset
import tiktoken

# data[0]['text'] would contain the text from the first page.
data = load_dataset('jamescalam/langchain-docs-23-06-27', split='train')

tokenizer = tiktoken.get_encoding('cl100k_base')

def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

token_counts = [ tiktoken_len(doc['text']) for doc in data ]

print(f'''Min: {min(token_counts)}
Avg: {sum(token_counts)/len(token_counts)}
Max: {max(token_counts)}''')

# Visualize
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# sns.set_style('whitegrid')
# sns.set_palette('muted')
# 
# plt.figure(figsize=(12, 6))
# sns.histplot(token_counts, kde=False, bins=50)
# 
# plt.title('Token Counts Histogram')
# plt.xlabel('Token Count')
# plt.ylabel('Frequency')
# 
# plt.show()

# Creating snippets of text, to keep below token limit.
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', ''],
)

chunks = text_splitter.split_text(data[5]['text'])
# print(len(chunks))
# print(tiktoken_len(chunks[0]), tiktoken_len(chunks[1]))

import hashlib
from tqdm.auto import tqdm

m = hashlib.md5()
documents = []

for doc in tqdm(data):
    url = doc.get('url')
    m.update(url.encode('utf-8'))
    uid = m.hexdigest()[:12]
    chunks = text_splitter.split_text(doc['text'])
    for i, chunk in enumerate(chunks):
        documents.append({
            'id': f'{uid}-{i}',
            'text': chunk,
            'url': url,
        })

import json

with open('train.jsonl', 'w') as f:
    for doc in documents:
        f.write(json.dumps(doc) + '\n')
