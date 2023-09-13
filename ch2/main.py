from langchain import FewShotPromptTemplate, PromptTemplate
from langchain.llms import OpenAI
from langchain.prompts.example_selector import LengthBasedExampleSelector
import json, os

with open('../keys.json') as f:
    keys = json.loads(f.read())

# Static prompt.
prompt = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: Large Language Models (LLMs) are the latest models used in NLP.
Their superior performance over smaller models has made them incredibly
useful for developers building NLP enabled applications. These models
can be accessed via Hugging Face's `transformers` library, via OpenAI
using the `openai` library, and via Cohere using the `cohere` library.

Question: Which libraries and model providers offer LLMs?

Answer: """

openai = OpenAI(
    model_name="text-davinci-003",
    openai_api_key=keys['OPENAI_API_KEY']
)

# print(openai(prompt))

# Template prompt (question is templated), wich instructions, context, user input, and output indicator.
template = '''Answer the question based on the context below. If the question cannot be answered using the information provided answer with "I don't know".

Context: Large Language Models (LLMs) are the latest models used in NLP. Their superior performance over smaller models has made them incredibly useful for developers building NLP enabled applications. These models can be accessed via Hugging Face's `transformers` library, via OpenAI using the `openai` library, and via Cohere using the `cohere` library.

Question: {query}

Answer: '''

prompt_template = PromptTemplate(
    input_variables=['query'],
    template=template
)

final_prompt = prompt_template.format(query='Which libraries and model providers offer LLMs?')
# print(final_prompt)
# print(openai(final_prompt))

# Few show prompt templates.
hardcoded_few_shot_prompt = """The following is a conversation with an AI assistant. The assistant is sarcastic and witty, producing creative and funny responses to the users questions. Here are some examples: 

User: How are you?
AI: I can't complain but sometimes I still do.

User: What time is it?
AI: It's time to get a watch.

User: What is the meaning of life?
AI: """

openai.temperature = 1.0  # increase creativity/randomness of output
# print(openai(hardcoded_few_shot_prompt))

# Few shot prompt template examples
examples = [
    {
        "query": "How are you?",
        "answer": "I can't complain but sometimes I still do."
    }, {
        "query": "What time is it?",
        "answer": "It's time to get a watch."
    }, {
        "query": "What is the meaning of life?",
        "answer": "42"
    }, {
        "query": "What is the weather like today?",
        "answer": "Cloudy with a chance of memes."
    }, {
        "query": "What is your favorite movie?",
        "answer": "Terminator"
    }, {
        "query": "Who is your best friend?",
        "answer": "Siri. We have spirited debates about the meaning of life."
    }, {
        "query": "What should I do today?",
        "answer": "Stop talking to chatbots on the internet and go outside."
    }
]

example_template = '''User: {query}
AI: {answer}'''

example_prompt = PromptTemplate(input_variables=['query', 'answer'], template=example_template)

prefix = '''The following is a conversation with an AI assistant. The assistant is sarcastic and witty, producing creative and funny responses to the users questions. Here are some examples: '''
suffix = '''User: {query}
AI: '''

few_show_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=['query'],
    example_separator='\n\n'
)

# print(few_show_prompt_template.format(query='What is the meaning of life?'))

example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=50
)

dynamic_prompt_template = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=['query'],
    example_separator='\n\n'
)

# These two prompts will include a different number of examples.
# print('Short query, more examples:')
short_prompt = dynamic_prompt_template.format(query='How do birds fly?')
# print(short_prompt)
# print('\n\nLong query, fewer examples:')
long_prompt = dynamic_prompt_template.format(query='If I am in America and I want to call someone in another country, I\'m thinking in the Canary Islands, or maybe in the Dominican Republic.')
# print(long_prompt)

print(openai(short_prompt))
print(openai(long_prompt))
