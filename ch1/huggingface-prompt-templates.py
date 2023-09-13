from langchain import HuggingFaceHub, LLMChain, PromptTemplate
import os
import json

with open('../keys.json') as f:
    keys = json.loads(f.read())

os.environ['HUGGINGFACEHUB_API_TOKEN'] = keys['HUGGINGFACEHUB_API_TOKEN']

with open('questions.json') as f:
    questions = json.loads(f.read())['questions']

# create prompt template > LLM chain
template = '''Question: {question}

Answer: '''

prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# Initialize Hub LLM
hub_llm = HuggingFaceHub(
    repo_id='google/flan-t5-large',
    model_kwargs={ 'temperature': 1e-10 }
)

llm_chain = LLMChain(
    prompt=prompt,
    llm=hub_llm,
)

# User question
qs = [ { 'question': questions[index] } for index in range(3) ]

# res = llm_chain.generate(qs)
# print(res)

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
    llm=hub_llm
)

qs_str= [ '\n'.join([ questions[index] for index in range(3) ]) ]

print(llm_chain.run(qs_str))
