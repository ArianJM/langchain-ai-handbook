import inspect, json

from langchain import OpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory,
                                                  ConversationSummaryMemory,
                                                  ConversationBufferWindowMemory,
                                                  ConversationSummaryBufferMemory,
                                                  ConversationKGMemory)
from langchain.callbacks import get_openai_callback
import tiktoken

with open('../keys.json') as f:
    keys = json.loads(f.read())

openai_api_key = keys['OPENAI_API_KEY']

llm = OpenAI(
    temperature=0,
    openai_api_key=openai_api_key,
    model_name='text-davinci-003',
)

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens.')

    return result

conversation = ConversationChain(llm=llm)

# Conversation Chain prompt. Includes instructions, history, and input.
# print(conversation.prompt.template)

def converse(conversationMemory):
    prompts = [
        'Good morning AI',
        'I\'m trying to learn Farsi, it is my mother tounge, so I have some basic knowledge, but I need to expand my vocabulary, and learn to read better.',
        'I enjoy the persian food a lot, it\'s one of the reasons why I would consider visiting, but I don\'t intend to go until the Bahá\'ís are treated as equal citizens.',
        'So what do you think are some good resources to achieve my goal of expanding my vocabulary and learn some reading?',
    ]
    for prompt in prompts:
        print(count_tokens(conversationMemory, prompt))

# Conversation buffer memory
conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(),
)
# print('\n\nConversation Buffer memory:')
# converse(conversation_buf)

# Conversation summary memory
conversation_sum = ConversationChain(
    llm=llm,
    memory=ConversationSummaryMemory(llm=llm),
)
# print('\n\nConversation Summary memory:')
# converse(conversation_sum)

# Conversation buffer window memory
conversation_buf_win = ConversationChain(
    llm=llm,
    memory=ConversationBufferWindowMemory(k=1),
)
# print('\n\nConversation Buffer window memory:')
# converse(conversation_buf_win)

# Conversation summary buffer window memory
conversation_sum_win = ConversationChain(
    llm=llm,
    memory=ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=650
    ),
)
# print('\n\nConversation Summary buffer window memory:')
# converse(conversation_sum_win)

# Conversation KG memory
conversation_kg = ConversationChain(
    llm=llm,
    memory=ConversationKGMemory(llm=llm),
)
print('\n\nConversation KG memory:')
converse(conversation_kg)

print(conversation_kg.memory.kg.get_triples())
