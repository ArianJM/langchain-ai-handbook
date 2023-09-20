import json
import tiktoken
from datasets import load_dataset
import pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings

with open('../keys.json') as f:
    keys = json.loads(f.read())

openai_api_key = keys['OPENAI_API_KEY']

data = load_dataset('wikipedia', '20220301.simple', split='train[:10000]')

tokenizer = tiktoken.get_encoding('cl100k_base')

# Length function
def tiktoken_len(text):
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

# print(tiktoken_len('Hello I am a chunk of text and using the tiktone_len function we can find the length of this chunk of text in tokens'))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)

# chunks = text_splitter.split_text(data[6]['text'])[:3]
# print(chunks)

# Creating embeddings
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    document_model_name=model_name,
    query_model_name=model_name,
    openai_api_key=openai_api_key
)

texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here',
]

res = embed.embed_documents(texts)
# print(len(res), len(res[0]))

pinecone_api_key = keys['PINECONE_API_KEY']
pinecone_env = keys['PINECONE_ENV']

index_name = 'langchain-retrieva-augmentation'
pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_env,
)

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=len(res[0]),
    )

populate_index = False
if populate_index:
    from tqdm.auto import tqdm
    from uuid import uuid4

    index = pinecone.GRPCIndex(index_name)
    batch_limit = 100

    texts = []
    metadatas = []

    for i, record in enumerate(tqdm(data)):
        # first get metadata fields for this record
        metadata = {
            'wiki-id': str(record['id']),
            'source': record['url'],
            'title': record['title'],
        }

        # now we create chunks from the record text
        record_texts = text_splitter.split_text(record['text'])
        
        # Create individual metadata dicts for each chunk
        record_metadatas = [ {
            'chunk': j, 'text': text, **metadata
        } for j, text in enumerate(record_texts) ]

        # append these to current batches
        texts.extend(record_texts)
        metadatas.extend(record_metadatas)

        # if we have reached the batch_limit we can add texts
        if (len(texts) > batch_limit):
            ids = [str(uuid4()) for _ in range(len(texts))]
            embeds = embed.embed_documents(texts)
            index.upsert(vectors=zip(ids, embeds, metadatas))
            texts = []
            metadatas = []
    
    if len(texts) > 0:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))

    print(index.describe_index_stats())

# Creating a vector store and querying
from langchain.vectorstores import Pinecone

text_field = 'text'

# switch back to normal index for langchain
index = pinecone.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

query = 'Who was Shoghi Effendi?'
# Return the 3 most similar documents for the query.
# print(vectorstore.similarity_search(query, k=3))

# Generative Question-Answering
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Completion LLM
llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name='gpt-3.5-turbo',
    temperature=0.0,
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vectorstore.as_retriever()
)

# print(qa.run(query))

# Include sources of information
from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vectorstore.as_retriever()
)

print(qa_with_sources(query))
