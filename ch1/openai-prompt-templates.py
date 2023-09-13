from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
import os
import json

with open('../keys.json') as f:
    keys = json.loads(f.read())

os.environ['OPENAI_API_KEY'] = keys['OPENAI_API_KEY']

with open('questions.json') as f:
    questions = json.loads(f.read())['questions']

davinci = OpenAI(model_name='text-davinci-003')

template = '''Question: {question}

Answer: '''
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

llm_chain = LLMChain(
    prompt=prompt,
    llm=davinci
)

# print(llm_chain.run(questions[0]))
qs = [ { 'question': questions[index] } for index in range(3) ]

res = llm_chain.generate(qs)
print(res)

# Try multiple questions in one prompt
multi_template = '''Answer the following questions one at a time.

Questions:
{questions}

Answers:
'''
long_prompt = PromptTemplate(
    template=multi_template,
    input_variables=['questions']
)

llm_chain = LLMChain(
    prompt=long_prompt,
    llm=davinci
)

qs_str= [ '\n'.join([ questions[index] for index in range(3) ]) ]

print(llm_chain.run(qs_str))
