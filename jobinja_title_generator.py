from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import os
import sys
from pathlib import Path
import torch

# Force UTF-8 encoding for all operations
os.environ["PYTHONUTF8"] = "1"
sys.stdout.reconfigure(encoding='utf-8')

class JobTitleGenerator:
    def __init__(self, model_path="./jobinja_model"):
        self.model_path = Path(model_path)
        self.pipeline = None
        self.load_model()

    def load_model(self):
        """Load the fine-tuned model using pipeline with proper device handling"""
        try:
            print("Loading model...")
            
            # First check if the directory exists
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model directory not found at {self.model_path.absolute()}")

            # Check if this is a PEFT model (has adapter files)
            is_peft_model = (self.model_path / "adapter_config.json").exists()
            
            if is_peft_model:
                print("Detected PEFT model (with adapters)")
                # Load base model config
                config = PeftConfig.from_pretrained(str(self.model_path))
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    config.base_model_name_or_path,
                    use_fast=True
                )
                tokenizer.pad_token = tokenizer.eos_token
                
                # Load base model with device_map="auto"
                base_model = AutoModelForCausalLM.from_pretrained(
                    config.base_model_name_or_path,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto"
                )
                
                # Load adapters
                model = PeftModel.from_pretrained(base_model, str(self.model_path))
                
                # Create pipeline without device parameter (let accelerate handle it)
                self.pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            else:
                print("Detected full model (without adapters)")
                # Load directly using pipeline with device_map="auto"
                self.pipeline = pipeline(
                    "text-generation",
                    model=str(self.model_path),
                    device_map="auto",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
                )
            
            print("Model loaded successfully")

        except Exception as e:
            print(f"\nError loading model: {str(e)}")
            self.print_detailed_troubleshooting()
            raise

    def print_detailed_troubleshooting(self):
        """Print detailed troubleshooting steps"""
        print("\nTROUBLESHOOTING GUIDE:")
        print("1. Verify model files exist in:", self.model_path.absolute())
        print("2. Required files:")
        print("   - For full models: config.json, pytorch_model.bin, tokenizer files")
        print("   - For PEFT models: adapter_config.json, adapter_model.bin")
        print("3. If problems persist, try:")
        print("   - Re-downloading the model files")
        print("   - Using absolute path for model directory")
        print("   - Checking CUDA availability with torch.cuda.is_available()")

    def generate_title(self, job_tags, job_skills):
        """Generate job title based on tags and skills"""
        try:
            prompt = (f"درخواست: بر اساس برچسب‌ها و مهارت‌ها، عنوان شغل مناسب را پیشنهاد دهید.\n"
                     f"برچسب‌ها: {job_tags}\n"
                     f"مهارت‌ها: {job_skills}\n"
                     f"عنوان شغل:")
            
            # Generate text using pipeline
            outputs = self.pipeline(
                prompt,
                max_new_tokens=50,
                num_beams=4,
                temperature=0.5,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                eos_token_id=self.pipeline.tokenizer.eos_token_id,
                pad_token_id=self.pipeline.tokenizer.eos_token_id
            )
            
            generated_text = outputs[0]['generated_text']
            
            # Extract and clean the job title
            generated_title = generated_text.split("عنوان شغل:")[-1].split("\n")[0].strip()
            for char in ['؛', ':', ')', '(', '"', "'", '.', '،']:
                generated_title = generated_title.replace(char, '')
                
            return generated_title.strip()

        except Exception as e:
            print(f"Error generating title: {str(e)}")
            return None

