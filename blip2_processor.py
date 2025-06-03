import os
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from tqdm import tqdm
import json
from pathlib import Path

class BLIP2Processor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        self.model.to(device)
        
    def process_image(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16)
            
            # Generate image caption
            generated_ids = self.model.generate(
                pixel_values=inputs.pixel_values,
                max_length=50,
                num_beams=5,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1.0,
            )
            
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            # Get image-text matching score
            with torch.no_grad():
                outputs = self.model(**inputs)
                score = outputs.logits.mean().item()
            
            return {
                "caption": caption,
                "score": score,
                "image_path": str(image_path)
            }
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None

    def process_directory(self, directory_path, output_file):
        results = []
        directory = Path(directory_path)
        
        # Process all jpg files in the directory
        image_files = list(directory.glob("**/*.jpg"))
        
        for image_path in tqdm(image_files, desc=f"Processing {directory.name}"):
            result = self.process_image(image_path)
            if result:
                results.append(result)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    # Initialize processor
    processor = BLIP2Processor()
    
    # Process positive samples
    positive_results = processor.process_directory(
        "positive",
        "positive_results.json"
    )
    
    # Process negative samples
    negative_results = processor.process_directory(
        "negative",
        "negative_results.json"
    )
    
    print(f"Processed {len(positive_results)} positive samples")
    print(f"Processed {len(negative_results)} negative samples")

if __name__ == "__main__":
    main() 