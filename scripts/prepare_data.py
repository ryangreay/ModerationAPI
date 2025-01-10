import pandas as pd
import json
import os
from typing import List, Tuple
from datasets import load_dataset
from huggingface_hub import hf_hub_download

def generate_slang_variations(text: str) -> List[str]:
    """Generate variations of slang terms"""
    if not isinstance(text, str):
        return []
    variations = [text]
    # Add common variations (e.g., different spellings, capitalizations)
    variations.extend([
        text.upper(),
        text.capitalize(),
        text.replace(" ", ""),
        text.replace(" ", "_"),
    ])
    return variations

def create_training_data():
    """Create and save training data"""
    try:
        # Create a basic set of slang terms since we're having issues with the dataset
        print("Creating training dataset...")
        
        # Define our own GenZ slang terms
        genz_slang = [
            ("no cap", "being honest or truthful"),
            ("bussin", "really good or amazing"),
            ("skibidi", "silly or weird behavior"),
            ("fr fr", "for real, for real"),
            ("sus", "suspicious or questionable"),
            ("slay", "doing something really well"),
            ("bet", "okay or agreement"),
            ("cap", "lying or fake"),
            ("mid", "mediocre or not good"),
            ("finna", "going to or about to"),
            ("lowkey", "secretive or subtle"),
            ("highkey", "obvious or clearly"),
            ("hits different", "especially good"),
            ("based", "being yourself or agreeable"),
            ("ratio", "getting more likes than someone"),
            ("rent free", "constantly thinking about something"),
            ("main character", "being the center of attention"),
            ("living rent free", "constantly in someone's thoughts"),
            ("periodt", "period, end of discussion"),
            ("tea", "gossip or drama"),
        ]
        
        # Convert GenZ slang to training examples
        genz_training_data = []
        
        for slang, example in genz_slang:
            # Add slang variations
            variations = generate_slang_variations(slang)
            for var in variations:
                if var:
                    genz_training_data.append((var, 1))  # 1 indicates harmful
            
            # Add example as context
            if example:
                genz_training_data.append((f"This is {slang}, it means {example}", 1))
                genz_training_data.append((f"That's so {slang}", 1))
        
        print(f"Processed {len(genz_training_data)} GenZ examples")
        
        # Define additional content categories
        slang_terms = {
            "inappropriate_terms": [
                "stupid",
                "dumb",
                "hate",
                "idiot",
                "loser",
                "shut up",
                "whatever",
                "don't care",
                "boring",
            ],
            "safe_terms": [
                "hello",
                "good morning",
                "thank you",
                "please help",
                "great job",
                "excellent work",
                "well done",
                "nice to meet you",
                "have a good day",
                "how are you",
                "I understand",
                "that makes sense",
                "could you explain",
                "interesting point",
                "good question",
            ]
        }
        
        # Create training examples
        training_data = []
        
        # Add GenZ training data
        training_data.extend(genz_training_data)
        
        # Process inappropriate terms
        for term in slang_terms["inappropriate_terms"]:
            variations = generate_slang_variations(term)
            for var in variations:
                training_data.append((var, 1))
        
        # Process safe terms
        for term in slang_terms["safe_terms"]:
            training_data.append((term, 0))  # 0 indicates safe
        
        # Create additional context examples
        context_examples = [
            # Safe examples with context
            ("I really enjoyed the class today", 0),
            ("Can you help me with this math problem?", 0),
            ("Great presentation! Well done!", 0),
            ("I learned a lot from this lesson", 0),
            ("What did you learn in school today?", 0),
            ("The homework was challenging but interesting", 0),
            ("I'm excited about the science project", 0),
            ("Could you explain this concept again?", 0),
            ("I appreciate your help", 0),
            ("That's a great question", 0),
            ("Let's work together on this", 0),
            ("I'll try my best to understand", 0),
            ("The teacher explained it well", 0),
            ("This assignment is interesting", 0),
            ("I made progress today", 0),
        ]
        
        training_data.extend(context_examples)
        
        # Convert to DataFrame
        df = pd.DataFrame(training_data, columns=['text', 'label'])
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Create data directory if it doesn't exist
        os.makedirs("data", exist_ok=True)
        
        # Save to CSV
        df.to_csv("data/training_data.csv", index=False)
        
        # Save categories for reference
        with open("data/categories.json", "w") as f:
            json.dump(slang_terms, f, indent=2)
        
        print(f"Created {len(df)} training examples")
        print("Data saved to data/training_data.csv")
        print("Categories saved to data/categories.json")
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    create_training_data() 