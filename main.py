from llama_cpp import Llama
from langchain.llms import LlamaCpp
from langchain import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
import json
import time
from langchain import FewShotPromptTemplate
from langchain import PromptTemplate


load_dotenv()


model_path = os.environ.get('MODEL_PATH')
print(model_path)

example =[
  {
    "question": "A lion lives in the Jungle",
    "answer": """ 
    Looking for a subject, 'A' is not a noun, ignore
    Next is 'lion', it is noun. It is our subject.

    All the non noun words between our subject and object can be our predicate.
    Looking for next noun, next noun is 'jungle', it is our object.
    At last, we simiplify for our Predicate , ' lives in the'  can be simplified as 'lives in', which is our predicate
    <sem:triple>
    <sem:subject>lion</sem:subject>
    <sem:predicate>lives in</sem:predicate>
    <sem:object>jungle</sem:object>
</sem:triple>
"""
  },]
# Load the model
print("....Loading LLAMA")
#llm = Llama(model_path=model_path, n_ctx=2048, n_threads=8)
# llm = ChatOpenAI(
#         temperature=0, model_name="gpt-3.5-turbo"
#     )  # Temp = 0 , means no additional playfullness from our model

#llm = LlamaWrapper(model_path=model_path, n_ctx=2048, n_threads=8)
llm = LlamaCpp(
    model_path=model_path, n_ctx=2048, n_threads=8)

text="Conclusions: Findings highlight the possibility that observed longitudinal associations between social media use and mental health might reflect other risk processes or vulnerabilities and/or the precision of measures used. More detailed and frequently collected prospective data about online experiences, including those from social media platforms themselves will help to provide a clearer picture of the relationship between social media use and mental health."

template = """/
Given the text data {text}, I want you to:
 extract all semantic triples in the format of (subject, predicate,object)"""

triple_template = PromptTemplate(input_variables=["text"], template=template)
#print(triple_template)
#triple_template.format(text=t)
chain = LLMChain(llm=llm, prompt=triple_template)
#Run the model
print("RUnning Model.....")
print(chain.run(text=text))

# while True:
#     query = input("Enter a query \n")

#     if query == "exit":
#         break
#     if query.strip() == "":
#         continue
#     start =  time.time()
#     output = llm(query, max_tokens=2048, temperature=0, repeat_penalty=1.2 )
#     end = time.time()
#     print(json.dumps(output, indent=2))
#     print(f"\n> Answer (took {round(end - start, 2)} s.):")