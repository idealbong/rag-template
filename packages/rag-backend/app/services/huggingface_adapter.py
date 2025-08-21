from dotenv import load_dotenv
import os

from transformers import BitsAndBytesConfig
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# .env 파일 로드
load_dotenv()

class HuggingfaceAdapter():
    def __init__(self):
        # Initialize Huggingface specific settings
        self.model = None
        self.model_loaded = False
            
        
    def load(self) -> ChatHuggingFace:
        # Load the model from Huggingface
            
        try:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="float16",
                bnb_4bit_use_double_quant=True,
            )
            
            llm = HuggingFacePipeline.from_model_id(
                model_id="HuggingFaceH4/zephyr-7b-beta",
                task="text-generation",
                pipeline_kwargs=dict(
                    max_new_tokens=512,
                    do_sample=False,
                    repetition_penalty=1.03,
                    return_full_text=False,
                ),
                model_kwargs={"quantization_config": quantization_config},
            )

            self.model = ChatHuggingFace(llm=llm)
            
            self.model_loaded = True
            print(f"✅ model {self.model_name} loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("💡 Fallback to template-based responses")
            self.model = None
            self.model_loaded = False
            
        return self.model