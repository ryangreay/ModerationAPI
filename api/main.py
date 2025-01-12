from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime, timedelta
import secrets
import logging
from dotenv import load_dotenv
import redis

# Load environment variables
load_dotenv()

app = FastAPI(title="K-12 Content Moderation API")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Constants
DEFAULT_DAILY_LIMIT = 1000  # Default requests per day
DEFAULT_KEY_NAME = "default"

# Redis setup
redis_client = redis.from_url(os.getenv('REDIS_URL'))

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

def get_db_connection():
    """Create a database connection"""
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")

def init_db():
    """Initialize PostgreSQL database"""
    conn = get_db_connection()
    with conn.cursor() as cur:
        # Create api_keys table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                key TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                daily_limit INTEGER DEFAULT 1000
            )
        ''')
        conn.commit()
    conn.close()

def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> bool:
    """Verify if API key exists and hasn't exceeded limits"""
    
    # Check if it's the default key first
    if api_key == os.getenv('DEFAULT_API_KEY2'):
        return True

    # Check rate limit first
    try:
        current_count = redis_client.get(f"rate_limit:{api_key}")
        if current_count and int(current_count) >= DEFAULT_DAILY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail=f"Daily limit of {DEFAULT_DAILY_LIMIT} requests exceeded"
            )
    except redis.RedisError as e:
        logger.error(f"Redis error: {e}")
        raise HTTPException(status_code=500, detail="Rate limit check failed")

    # Then check if key exists in database
    try:
        conn = get_db_connection()
        with conn.cursor(cursor_factory=DictCursor) as cur:
            cur.execute('''
                SELECT key, name, daily_limit 
                FROM api_keys 
                WHERE key = %s
            ''', (api_key,))
            result = cur.fetchone()
        conn.close()

        if not result:
            raise HTTPException(
                status_code=403,
                detail="Invalid API key"
            )
        
        # Increment counter for valid keys
        if current_count is None:
            redis_client.setex(f"rate_limit:{api_key}", 86400, 1)
        else:
            redis_client.incr(f"rate_limit:{api_key}")
        
        return True
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying API key: {e}")
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )

# Load the model and tokenizer
MODEL_PATH = "/app/models/fine_tuned_bert"

try:
    logger.info(f"Attempting to load model from {MODEL_PATH}")
    logger.info(f"Directory exists: {os.path.exists(MODEL_PATH)}")
    logger.info(f"Directory contents: {os.listdir(MODEL_PATH) if os.path.exists(MODEL_PATH) else 'directory not found'}")
    
    required_files = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]
    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(MODEL_PATH, f))]
    
    if missing_files:
        logger.warning(f"Missing required model files: {missing_files}")
        logger.warning("Using base model instead")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    else:
        logger.info("All required files found, loading fine-tuned model")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH, 
            local_files_only=True,
            use_safetensors=True
        )
        logger.info("Successfully loaded fine-tuned model from volume")
    
    model.eval()
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    logger.error(f"Error type: {type(e)}")
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize database on startup"""
    init_db()
    # Create a default API key if none exists
    conn = get_db_connection()
    with conn.cursor() as cur:
        #cur.execute('DELETE FROM api_keys')
        cur.execute('SELECT COUNT(*) FROM api_keys')
        if cur.fetchone()[0] == 0:
            default_key = os.getenv('DEFAULT_API_KEY2') or secrets.token_urlsafe(32)
            cur.execute('''
                INSERT INTO api_keys (key, name, daily_limit)
                VALUES (%s, %s, -1)
            ''', (default_key, DEFAULT_KEY_NAME))
            conn.commit()
            logger.info(f"Created default API key: {default_key}")
    conn.close()

@app.post("/api/moderate", response_model=ModerationResponse)
async def moderate_content(request: TextRequest, api_key: str = Security(API_KEY_HEADER)):
    # Verify API key and limits
    verify_api_key(api_key)
    logger.info(f"API key verified for API key: {api_key}")
    
    try:
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
        logger.error(f"Error in moderation model: {e}")
        raise HTTPException(status_code=500, detail="Error processing content")

@app.post("/api/keys", response_model=APIKey)
async def create_api_key(name: str, daily_limit: int = DEFAULT_DAILY_LIMIT, api_key: str = Security(API_KEY_HEADER)):
    """Create a new API key (only allowed with default key)"""
    conn = get_db_connection()
    with conn.cursor(cursor_factory=DictCursor) as cur:
        # Verify it's the default key making this request
        cur.execute('SELECT name FROM api_keys WHERE key = %s', (api_key,))
        result = cur.fetchone()
        
        if not result or result['name'] != DEFAULT_KEY_NAME:
            raise HTTPException(
                status_code=403,
                detail="Only the default API key can create new keys"
            )
        
        # Create new key
        new_key = secrets.token_urlsafe(32)
        
        cur.execute('''
            INSERT INTO api_keys (key, name, daily_limit)
            VALUES (%s, %s, %s)
            RETURNING created_at
        ''', (new_key, name, daily_limit))
        
        created_at = cur.fetchone()['created_at']
        conn.commit()
    conn.close()
    
    return {
        "key": new_key,
        "name": name,
        "created_at": created_at.isoformat(),
        "daily_limit": daily_limit,
        "requests_today": 0
    }

@app.get("/api/keys/{key}/usage", response_model=APIKey)
async def get_key_usage(key: str, authenticated: bool = Depends(verify_api_key)):
    """Get usage information for an API key"""
    conn = get_db_connection()
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute('''
            SELECT name, created_at, daily_limit
            FROM api_keys
            WHERE key = %s
        ''', (key,))
        result = cur.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=404, detail="API key not found")
    
    # Get current usage from Redis
    current_usage = redis_client.get(f"rate_limit:{key}")
    requests_today = int(current_usage) if current_usage else 0
    
    return {
        "key": key,
        "name": result['name'],
        "created_at": result['created_at'].isoformat(),
        "daily_limit": result['daily_limit'],
        "requests_today": requests_today
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 