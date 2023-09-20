import inspect
import json
import re

from getpass import getpass
from langchain import OpenAI, PromptTemplate
from langchain.chains import LLMChain, LLMMathChain, TransformChain, SequentialChain
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

def transform_func(inputs: dict) -> dict:
    text = inputs['text']

    # replace multiple new lines an dmultiple spaces with a single one.
    text = re.sub(r'(\r\n|\r|\n){2,}', r'\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return { 'output_text': text }

clean_extra_spaces_chain = TransformChain(
    input_variables=['text'],
    output_variables=['output_text'],
    transform=transform_func
)

template = '''Paraphrase this text:

{output_text}

In the style of {style}.

Paraphrase: '''
prompt = PromptTemplate(
    input_variables=['style', 'output_text'],
    template=template
)

style_paraphrase_chain = LLMChain(llm=llm, prompt=prompt, output_key='final_output')

sequential_chain = SequentialChain(
    chains=[clean_extra_spaces_chain, style_paraphrase_chain],
    input_variables=['text', 'style'],
    output_variables=['final_output']
)

input_text = '''
Chains allow us to combine multiple 


components together to create a single, coherent application. 

For example, we can create a chain that takes user input,       format it with a PromptTemplate, 

and then passes the formatted response to an LLM. We can build more complex chains by combining     multiple chains together, or by 


combining chains with other components.
'''

print(count_tokens(sequential_chain, {'text': input_text, 'style': 'a teenage mutant ninja turtle'}))
