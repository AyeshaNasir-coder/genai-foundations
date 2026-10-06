import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


load_dotenv() 

llm = init_chat_model(
    model=os.getenv("MODEL_NAME"),
    temperature=1.0
)

history = [{"role": "system", "content": "You are helpful ai assistant!"}]

print("Welcome to cue chat!")
while True:
   question = input("You: ").strip()

   if not question:
      continue
   if question in ("exit", "quit", "bye"):
      print("Goodbye!")
      break

   history.append({
      "role": "user",
      "content": question
   })
   print("Bot: ", end="", flush=True)

   full = None  
   for chunk in llm.stream(history):
      print(chunk.text, end="", flush=True)
      full = chunk if full is None else full + chunk
   print()

   history.append(full)