from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)

history=[
    SystemMessage("You are an AI assistant")
]

while True:
    user_i=input("You: ")
    history.append(HumanMessage(user_i))
    if(user_i=='exit'):
        break
    result=model.invoke(history)
    history.append(AIMessage(result.text))
    print("AI: ",result.text)
