import json
from langchain import OpenAI
from langchain.chains import LLMMathChain
from langchain.callbacks import get_openai_callback

with open('../keys.json') as f:
    keys = json.loads(f.read())

openai_api_key = keys['OPENAI_API_KEY']

llm = OpenAI(
    temperature = 0,
    openai_api_key=openai_api_key
)

def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens.')

    return result

llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)

count_tokens(llm_math, 'What is 13 raised to the .3432 power?')
