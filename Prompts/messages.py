from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = ChatPromptTemplate([
    ('system', "You are a Helpful Customer Agent"),
    MessagesPlaceholder(variable_name='chat_his'),
    ('human', '{query}')
])

chat_his = []
with open('chat_his.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Human:"):
            chat_his.append(HumanMessage(content=line.removeprefix("Human:").strip()))
        elif line.startswith("AI:"):
            chat_his.append(AIMessage(content=line.removeprefix("AI:").strip()))

prompt = template.invoke({'chat_his': chat_his, 'query': 'Where is my refund'})
print(prompt)