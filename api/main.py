from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import sqlite3
from datetime import datetime, timedelta
import secrets
import logging

app = FastAPI(title="K-12 Content Moderation API")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Constants
DEFAULT_DAILY_LIMIT = 1000  # Default requests per day
DEFAULT_KEY_NAME = "default"

class TextRequest(BaseModel):
    text: str

class ModerationResponse(BaseModel):
    is_harmful: bool
    confidence: float
    categories: dict

class APIKey(BaseModel):
    key: str
    name: str
    created_at: str
    daily_limit: int
    requests_today: int

def init_db():
    """Initialize SQLite database with usage tracking"""
    conn = sqlite3.connect('api_keys.db')
    c = conn.cursor()
    
    # Create api_keys table with daily limit
    c.execute('''
        CREATE TABLE IF NOT EXISTS api_keys
        (key TEXT PRIMARY KEY, 
         name TEXT, 
         created_at TEXT,
         daily_limit INTEGER DEFAULT 1000)
    ''')
    
    # Create usage tracking table
    c.execute('''
        CREATE TABLE IF NOT EXISTS api_usage
        (key TEXT,
         date TEXT,
         requests INTEGER DEFAULT 0,
         PRIMARY KEY (key, date),
         FOREIGN KEY (key) REFERENCES api_keys(key))
    ''')
    
    conn.commit()
    conn.close()

def get_today_usage(api_key: str) -> int:
    """Get the number of requests made today for this key"""
    today = datetime.now().date().isoformat()
    conn = sqlite3.connect('api_keys.db')
    c = conn.cursor()
    
    # Create today's record if it doesn't exist
    c.execute('''
        INSERT OR IGNORE INTO api_usage (key, date, requests)
        VALUES (?, ?, 0)
    ''', (api_key, today))
    
    # Get current usage
    c.execute('''
        SELECT requests FROM api_usage
        WHERE key = ? AND date = ?
    ''', (api_key, today))
    
    result = c.fetchone()
    conn.close()
    return result[0] if result else 0

def increment_usage(api_key: str):
    """Increment the usage counter for today"""
    today = datetime.now().date().isoformat()
    conn = sqlite3.connect('api_keys.db')
    c = conn.cursor()
    
    c.execute('''
        UPDATE api_usage
        SET requests = requests + 1
        WHERE key = ? AND date = ?
    ''', (api_key, today))
    
    conn.commit()
    conn.close()

def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> bool:
    """Verify if API key exists and hasn't exceeded limits"""
    conn = sqlite3.connect('api_keys.db')
    c = conn.cursor()
    
    # Get key info
    c.execute('SELECT key, name, daily_limit FROM api_keys WHERE key = ?', (api_key,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    
    key, name, daily_limit = result
    
    # Default key has no limits
    if name == DEFAULT_KEY_NAME:
        return True
    
    # Check usage limits
    current_usage = get_today_usage(api_key)
    if current_usage >= daily_limit:
        raise HTTPException(
            status_code=429,
            detail=f"Daily limit of {daily_limit} requests exceeded. Current usage: {current_usage}"
        )
    
    return True

# Load the model and tokenizer
model_path = os.getenv("MODEL_PATH", "models/fine_tuned_bert")
try:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
except Exception as e:
    logger.error(f"Error loading model: {e}")
    # Fall back to base BERT model if fine-tuned model isn't available
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    # Create a default API key if none exists
    conn = sqlite3.connect('api_keys.db')
    c = conn.cursor()
    c.execute('SELECT COUNT(*) FROM api_keys')
    if c.fetchone()[0] == 0:
        default_key = secrets.token_urlsafe(32)
        c.execute(
            'INSERT INTO api_keys (key, name, created_at, daily_limit) VALUES (?, ?, ?, -1)',
            (default_key, DEFAULT_KEY_NAME, datetime.now().isoformat())
        )
        conn.commit()
        logger.info(f"Created default API key: {default_key}")
    conn.close()

@app.post("/api/moderate", response_model=ModerationResponse)
async def moderate_content(request: TextRequest, api_key: str = Security(API_KEY_HEADER)):
    try:
        # Verify API key and limits
        verify_api_key(api_key)
        
        # Increment usage (except for default key)
        if api_key != DEFAULT_KEY_NAME:
            increment_usage(api_key)
        
        # Tokenize and prepare input
        inputs = tokenizer(request.text, return_tensors="pt", truncation=True, max_length=512)
        
        # Get model prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            harmful_prob = probabilities[0][1].item()
        
        # Define confidence threshold
        CONFIDENCE_THRESHOLD = 0.7
        
        # Prepare response
        response = {
            "is_harmful": harmful_prob > CONFIDENCE_THRESHOLD,
            "confidence": harmful_prob,
            "categories": {
                "inappropriate_slang": harmful_prob > 0.8,
                "potentially_harmful": harmful_prob > CONFIDENCE_THRESHOLD,
                "safe": harmful_prob < CONFIDENCE_THRESHOLD
            }
        }
        
        return response
    
    except Exception as e:
        logger.error(f"Error in moderation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/keys", response_model=APIKey)
async def create_api_key(name: str, daily_limit: int = DEFAULT_DAILY_LIMIT, api_key: str = Security(API_KEY_HEADER)):
    """Create a new API key (only allowed with default key)"""
    # Verify it's the default key making this request
    conn = sqlite3.connect('api_keys.db')
    c = conn.cursor()
    c.execute('SELECT name FROM api_keys WHERE key = ?', (api_key,))
    result = c.fetchone()
    
    if not result or result[0] != DEFAULT_KEY_NAME:
        raise HTTPException(
            status_code=403,
            detail="Only the default API key can create new keys"
        )
    
    # Create new key
    new_key = secrets.token_urlsafe(32)
    created_at = datetime.now().isoformat()
    
    c.execute(
        'INSERT INTO api_keys (key, name, created_at, daily_limit) VALUES (?, ?, ?, ?)',
        (new_key, name, created_at, daily_limit)
    )
    conn.commit()
    conn.close()
    
    return {
        "key": new_key,
        "name": name,
        "created_at": created_at,
        "daily_limit": daily_limit,
        "requests_today": 0
    }

@app.get("/api/keys/{key}/usage", response_model=APIKey)
async def get_key_usage(key: str, authenticated: bool = Depends(verify_api_key)):
    """Get usage information for an API key"""
    conn = sqlite3.connect('api_keys.db')
    c = conn.cursor()
    c.execute(
        'SELECT name, created_at, daily_limit FROM api_keys WHERE key = ?',
        (key,)
    )
    result = c.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=404, detail="API key not found")
    
    name, created_at, daily_limit = result
    requests_today = get_today_usage(key)
    
    return {
        "key": key,
        "name": name,
        "created_at": created_at,
        "daily_limit": daily_limit,
        "requests_today": requests_today
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 