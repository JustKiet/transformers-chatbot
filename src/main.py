from src.interfaces.llm import LLM
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional, Any
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch
import gc

class Chatbot(LLM):
    def __init__(self,
                 model: str,
                 system_message: str,
                 device,
                 quant_config: Optional[Any] = None,
                 ):
        super().__init__(model, device)
        self.quant_config = quant_config
        self.history = [
            {
                "role": "system",
                "content": system_message
            }
        ]
        self.model = AutoModelForCausalLM.from_pretrained(model, 
                                                          torch_dtype="auto",
                                                          quantization_config=self.quant_config)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def invoke(self, query):
        structured_query = {
            "role": "user",
            "content": query
        }

        self.history.append(structured_query)
        
        self.pipe.model.to(self.device)
        
        outputs = self.pipe(
            self.history,
            max_new_tokens=100,
        )

        response = outputs[0]["generated_text"][-1]

        self.history.append(response)

        return response["content"]



if __name__ == '__main__':
    # Load the API key from .env file
    load_dotenv()
    login(token=os.getenv("HUGGINGFACE_API_KEY"))

    # Clear memory
    gc.collect()    
    torch.cuda.empty_cache()

    # Define quantization configuration
    double_quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Setting up hyperparams
    SYSTEM_MESSAGE = "You are a chatbot, you're tasked with assisting users in their queries. Be precise, helpful and nice."
    MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize chatbot
    chatbot = Chatbot(
        model=MODEL_ID,
        system_message=SYSTEM_MESSAGE,
        device=device,
        quant_config=double_quant_config
    )
    
    # Start chat
    while True:
        query = input("You: ")
        if query == "exit":
            break
        response = chatbot.invoke(query)
        print("Bot:", response)