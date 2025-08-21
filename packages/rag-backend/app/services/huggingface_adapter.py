from dotenv import load_dotenv
import os

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# .env 파일 로드
load_dotenv()
class HuggingfaceAdapter():
    def __init__(self):
        # Initialize Huggingface specific settings
        self.model = None
        self.model_id = os.getenv("HUGGINGFACE_MODEL_ID", "kakaocorp/kanana-1.5-2.1b-instruct-2505")
        self.model_name = self.model_id.split("/")[-1]
        self.model_loaded = False
            
        
    def load(self) -> ChatHuggingFace:
        # Load the model from Huggingface
            
        try:
            llm = HuggingFacePipeline.from_model_id(
                model_id=self.model_id,
                task="text-generation",
                pipeline_kwargs=dict(
                    max_new_tokens=512,
                    do_sample=False,
                    repetition_penalty=1.03,
                    return_full_text=False,
                ),
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
    
    def get_info(self) -> dict:
        """모델 정보를 반환합니다."""
        return {
            "model_provider": "huggingface",
            "model_id": self.model_id,
            "status": "loaded" if self.model_loaded else "template_mode",
            "model_name": self.model_name
        }