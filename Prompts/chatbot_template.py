from langchain_core.prompts import ChatPromptTemplate

template=ChatPromptTemplate([
    ('system','You are an helpful {domain} Expert'),
    ('human','Explain me in Simple Terms what is {topic}')
])

prompt=template.invoke({'domain':'cricket','topic':'pacer'})
print(prompt)

