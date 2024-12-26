from src.interfaces.agent import Agent
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from typing import Optional
from huggingface_hub import login
from dotenv import load_dotenv
from jinja2 import Template
import os
import torch

class Chatbot(Agent):
    def __init__(self,
                 model: str,
                 system_message: str,
                 device,
                 ):
        super().__init__(model, device)
        self.history = [
            {
                "role": "system",
                "content": system_message
            }
        ]
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device
        )
        
    def format_conversation(self, template_str, messages, bos_token, eos_token):
        template = Template(template_str)
        
        return template.render(messages=messages, bos_token=bos_token, eos_token=eos_token)

    def invoke(self, query):
        structured_query = {
            "role": "user",
            "content": query
        }

        self.history.append(structured_query)
        
        outputs = self.pipe(
            self.history,
            max_new_tokens=100,
        )

        response = outputs[0]["generated_text"][-1]

        self.history.append(response)

        return response["content"]



if __name__ == '__main__':

    load_dotenv()
    login(token=os.getenv("HUGGINGFACE_API_KEY"))

    SYSTEM_MESSAGE = "You are a chatbot, you're tasked with assisting users in their queries. Be precise, helpful and nice."
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    chatbot = Chatbot(
        model=MODEL_ID,
        system_message=SYSTEM_MESSAGE,
        device=device,
    )
    
    while True:
        query = input("You: ")
        if query == "exit":
            break
        response = chatbot.invoke(query)
        print("Bot:", response)