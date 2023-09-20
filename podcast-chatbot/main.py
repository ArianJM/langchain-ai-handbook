import json


with open('../keys.json') as f:
    keys = json.loads(f.read())

openai_api_key = keys['OPENAI_API_KEY']
pinecone_api_key = keys['PINECONE_API_KEY']
pinecone_env = keys['PINECONE_ENV']

# Load dataset
from datasets import load_dataset

data = load_dataset(
    'jamescalam/lex-transcripts',
    split='train'
)

# print(data)

import pod_gpt

index_name = 'pod-gpt'
indexer = pod_gpt.Indexer(
    openai_api_key=openai_api_key,
    pinecone_api_key=pinecone_api_key,
    pinecone_environment=pinecone_env,
    index_name=index_name
)

index_data = False
if index_data:
    from tqdm.auto import tqdm

    for row in data:
        row['published'] = row['published'].strftime('%Y%m%d')

        # Indexer divides the text into smaller chunks
        indexer(pod_gpt.VideoRecord(**row))

# Initialize pinecone and prepare index.
import pinecone

pinecone.init(
    api_key=pinecone_api_key,
    environment=pinecone_env
)

if index_name not in pinecone.list_indexes():
    raise ValueError(
        f'No \'{index_name}\' index exists. You must create the index before running this.'
    )

index = pinecone.Index(index_name)
# Print a row from Pinecone. You can see the format its stored in.
# print(index.query([0.0]*1536, top_k=1, include_metadata=True))

# Initialize retrieval components
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
vectordb = Pinecone(
    index=index,
    embedding_function=embeddings.embed_query,
    text_key='text'
)

# Initialize gpt-3.5-turbo
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    temperature=0,
    model_name='gpt-3.5-turbo',
)

# Initialize Retrieval QA
from langchain.chains import RetrievalQA

retriever = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=vectordb.as_retriever()
)

# Create a tool for the agent to use.
from langchain.agents import Tool

tool_desc = '''Use this tool to answer user questions using Lex
Fridman podcasts. If the user states 'ask Lex' use this tool to get
the answer. This tool can also be used for follow up questions from
the user.'''

tools = [Tool(
    func=retriever.run,
    description=tool_desc,
    name='Lex Fridman DB'
)]

# Ensure we use memory for the conversation.
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# Initialize conversation agent
from langchain.agents import initialize_agent

conversational_agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=2,
    early_stopping_method='generate',
    memory=memory,
)

# Conversatino agent prompt
sys_msg = '''You are a helpful chatbot that answers the user's questions.'''

prompt = conversational_agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools,
)
conversational_agent.agent.llm_chain.prompt = prompt

print('I\'m ready for your questions.')
for i in range(5):
    print('> ', end='')
    question = input()
    if (question == 'exit'):
        break
    print(conversational_agent(question))

