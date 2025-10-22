"""
FastAPI Backend for Hate Speech Detection System with CSV Processing
Provides RESTful API endpoints for classification, explanation, moderation, and CSV batch processing

Features:
- Single text classification
- Batch classification
- CSV file processing with user analytics
- Explainable predictions
- Severity analysis
- Content moderation
- User dashboard analytics
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
import logging
import sys
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from inference.explainable_classifier import (
    ExplainableTweetClassifier,
    classify_with_explanation,
    get_explanation_summary
)
from config import CLASS_LABELS, SEVERITY_LEVELS

# Import CSV processor
try:
    from csv_processor import CSVProcessor
    HAS_CSV_PROCESSOR = True
except ImportError:
    HAS_CSV_PROCESSOR = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Hate Speech Detection API with CSV Processing",
    description="Advanced hate speech detection with explainable AI, severity analysis, CSV batch processing, and user analytics dashboard",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==================== CORS MIDDLEWARE ====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== GLOBAL CLASSIFIER ====================

classifier = None

def get_classifier():
    """Lazy load classifier (loaded on first request)"""
    global classifier
    if classifier is None:
        logger.info("Loading classifier...")
        try:
            classifier = ExplainableTweetClassifier()
            logger.info("Classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load classifier: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize classifier: {str(e)}"
            )
    return classifier

# ==================== REQUEST/RESPONSE MODELS ====================

class TextInput(BaseModel):
    """Single text input"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")
    include_severity: bool = Field(default=True, description="Include severity analysis")
    include_explanation: bool = Field(default=True, description="Include explainable AI insights")
    
    @validator('text')
    def text_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()


class BatchTextInput(BaseModel):
    """Batch text input"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to classify")
    include_severity: bool = Field(default=False, description="Include severity analysis")
    include_explanation: bool = Field(default=False, description="Include explainable AI insights")
    
    @validator('texts')
    def texts_not_empty(cls, v):
        cleaned = [t.strip() for t in v if t.strip()]
        if not cleaned:
            raise ValueError('All texts cannot be empty')
        return cleaned


class ClassificationResponse(BaseModel):
    """Single classification response"""
    text: str
    prediction: str
    class_id: int
    confidence: float
    probabilities: Dict[str, float]
    severity: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    action: Optional[Dict[str, Any]] = None
    timestamp: str
    model_info: Dict[str, str]


class BatchClassificationResponse(BaseModel):
    """Batch classification response"""
    results: List[ClassificationResponse]
    summary: Dict[str, Any]
    total_processed: int
    timestamp: str


class CSVProcessingResponse(BaseModel):
    """CSV processing response"""
    status: str
    total_comments: int
    total_users: int
    summary: Dict[str, Any]
    user_analytics: Dict[str, Any]
    top_risk_users: List[Dict[str, Any]]
    timestamp: str


class UserAnalyticsResponse(BaseModel):
    """User analytics response"""
    user: str
    total_comments: int
    hate_percentage: float
    offensive_percentage: float
    risk_score: float
    risk_level: str
    most_severe_comment: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool
    csv_processor_available: bool
    version: str

# ==================== UTILITY FUNCTIONS ====================

def format_classification_response(result: Dict, include_severity: bool, include_explanation: bool) -> Dict:
    """Format classification result into API response"""
    response = {
        "text": result.get("text", ""),
        "prediction": result.get("prediction", ""),
        "class_id": result.get("class", -1),
        "confidence": result.get("confidence", 0.0),
        "probabilities": result.get("probabilities", {}),
        "timestamp": datetime.utcnow().isoformat(),
        "model_info": result.get("model_info", {})
    }
    
    # Add severity if requested and available
    if include_severity and "severity" in result:
        severity = result["severity"]
        response["severity"] = {
            "level": severity.get("severity_level"),
            "label": severity.get("severity_label"),
            "score": severity.get("severity_score"),
            "explanation": severity.get("explanation")
        }
    
    # Add explanation if requested and available
    if include_explanation and "explanation" in result:
        explanation = result["explanation"]
        response["explanation"] = {
            "methods_used": explanation.get("methods_used", []),
            "summary": get_explanation_summary(result.get("text", ""))
        }
    
    # Add action if available
    if "action" in result:
        action = result["action"]
        response["action"] = {
            "primary": action.get("primary_action"),
            "urgency": action.get("urgency"),
            "description": action.get("action_string")
        }
    
    return response


def get_top_risk_users(user_analytics: Dict, top_n: int = 10) -> List[Dict]:
    """Get top N users by risk score"""
    users = []
    for username, analytics in user_analytics.items():
        users.append({
            'user': username,
            'risk_score': analytics['risk_score'],
            'risk_level': analytics['risk_level'],
            'hate_percentage': analytics['hate_percentage'],
            'total_comments': analytics['total_comments']
        })
    
    users.sort(key=lambda x: x['risk_score'], reverse=True)
    return users[:top_n]

# ==================== BASIC API ENDPOINTS ====================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Hate Speech Detection API with CSV Processing",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "features": "Classification, Severity Analysis, CSV Processing, User Analytics"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        global classifier
        model_loaded = classifier is not None
        
        return HealthResponse(
            status="healthy" if model_loaded else "initializing",
            timestamp=datetime.utcnow().isoformat(),
            model_loaded=model_loaded,
            csv_processor_available=HAS_CSV_PROCESSOR,
            version="2.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")

# ==================== CLASSIFICATION ENDPOINTS ====================

@app.post("/classify", response_model=ClassificationResponse)
async def classify_text(input_data: TextInput):
    """
    Classify a single text for hate speech
    
    - **text**: Text to classify (1-5000 characters)
    - **include_severity**: Include severity analysis (default: true)
    - **include_explanation**: Include explainable AI insights (default: true)
    
    Returns classification, confidence, probabilities, and optional severity/explanation
    """
    try:
        clf = get_classifier()
        
        result = clf.classify_with_explanation(
            text=input_data.text,
            include_severity=input_data.include_severity,
            verbose=False
        )
        
        response = format_classification_response(
            result,
            input_data.include_severity,
            input_data.include_explanation
        )
        
        logger.info(f"Classified text: {input_data.text[:50]}... -> {response['prediction']}")
        
        return ClassificationResponse(**response)
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(input_data: BatchTextInput):
    """
    Classify multiple texts in batch
    
    - **texts**: List of texts to classify (1-100 texts)
    - **include_severity**: Include severity analysis (default: false)
    - **include_explanation**: Include explanations (default: false)
    
    Returns classifications for all texts plus summary statistics
    """
    try:
        clf = get_classifier()
        
        results = []
        predictions_count = {}
        
        for text in input_data.texts:
            try:
                result = clf.classify_with_explanation(
                    text=text,
                    include_severity=input_data.include_severity,
                    verbose=False
                )
                
                response = format_classification_response(
                    result,
                    input_data.include_severity,
                    input_data.include_explanation
                )
                
                results.append(ClassificationResponse(**response))
                
                pred = response["prediction"]
                predictions_count[pred] = predictions_count.get(pred, 0) + 1
                
            except Exception as e:
                logger.error(f"Error classifying text '{text[:30]}...': {e}")
                continue
        
        summary = {
            "total_classified": len(results),
            "predictions_breakdown": predictions_count,
            "hate_speech_count": predictions_count.get("Hate speech", 0),
            "offensive_count": predictions_count.get("Offensive language", 0),
            "neither_count": predictions_count.get("Neither", 0)
        }
        
        logger.info(f"Batch classified {len(results)} texts")
        
        return BatchClassificationResponse(
            results=results,
            summary=summary,
            total_processed=len(results),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch classification error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch classification failed: {str(e)}"
        )

# ==================== CSV PROCESSING ENDPOINTS ====================

import numpy as np
import pandas as pd

def make_json_serializable(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    else:
        return obj

@app.post("/process-csv")
async def process_csv_file(file: UploadFile = File(...)):
    """
    Process uploaded CSV file with user comments
    
    Expected CSV format:
    - name/user/username: User identifier
    - timestamp/date/time: Comment timestamp  
    - comment/text/message: Comment text
    
    Returns processing results and user analytics dashboard data
    """
    if not HAS_CSV_PROCESSOR:
        raise HTTPException(
            status_code=501,
            detail="CSV processing not available. Please ensure csv_processor.py is in the project directory."
        )
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="File must be a CSV file (.csv extension required)"
        )
    
    temp_path = None
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        logger.info(f"Processing uploaded CSV: {file.filename}")
        
        # Process CSV
        processor = CSVProcessor()
        results_df = processor.process_csv(temp_path)
        
        # Get analytics
        user_analytics = processor.get_user_analytics()
        summary = processor.generate_summary_report()
        
        logger.info(f"CSV processed: {len(results_df)} comments from {len(user_analytics)} users")
        
        # Prepare response - CONVERT TO JSON-SERIALIZABLE FORMAT
        response = {
            "status": "success",
            "total_comments": int(len(results_df)),  # Ensure int
            "total_users": int(len(user_analytics)),  # Ensure int
            "summary": make_json_serializable(summary),
            "user_analytics": make_json_serializable(user_analytics),
            "top_risk_users": make_json_serializable(get_top_risk_users(user_analytics, 10)),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return JSONResponse(content=response)
        
    except ValueError as e:
        logger.error(f"CSV validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"CSV processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process CSV: {str(e)}"
        )
    finally:
        # Cleanup temp file
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

@app.post("/process-csv/detailed")
async def process_csv_detailed(file: UploadFile = File(...)):
    """
    Process CSV and return detailed per-comment results
    
    Returns all classification results for each comment
    """
    if not HAS_CSV_PROCESSOR:
        raise HTTPException(
            status_code=501,
            detail="CSV processing not available"
        )
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400,
            detail="File must be a CSV file"
        )
    
    temp_path = None
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Process CSV
        processor = CSVProcessor()
        results_df = processor.process_csv(temp_path)
        
        # Convert to JSON
        results = results_df.to_dict('records')
        
        # Convert timestamps to strings
        for result in results:
            if 'timestamp' in result and pd.notna(result['timestamp']):
                result['timestamp'] = str(result['timestamp'])
        
        return {
            "status": "success",
            "total_comments": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"CSV processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process CSV: {str(e)}"
        )
    finally:
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except:
                pass

# ==================== INFO ENDPOINTS ====================

@app.get("/classes", response_model=Dict[int, str])
async def get_classes():
    """Get available classification classes"""
    return CLASS_LABELS


@app.get("/severity-levels", response_model=Dict[int, str])
async def get_severity_levels():
    """Get available severity levels"""
    return SEVERITY_LEVELS


@app.get("/models/info", response_model=Dict[str, Any])
async def get_model_info():
    """Get information about loaded models"""
    try:
        clf = get_classifier()
        info = clf.get_model_info()
        
        return {
            "model": info.get("model_name", "unknown"),
            "type": info.get("model_type", "unknown"),
            "metrics": info.get("metrics", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model info: {str(e)}"
        )

# ==================== ERROR HANDLERS ====================

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation Error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general errors"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# ==================== STARTUP/SHUTDOWN EVENTS ====================

@app.on_event("startup")
async def startup_event():
    """Run on API startup"""
    logger.info("=" * 80)
    logger.info("Hate Speech Detection API v2.0 - Starting Up")
    logger.info("=" * 80)
    logger.info("Features: Classification, Severity Analysis, CSV Processing, User Analytics")
    logger.info("API is ready to accept requests")
    logger.info(f"CSV Processor: {'Available' if HAS_CSV_PROCESSOR else 'Not Available'}")
    logger.info("Classifier will be loaded on first request")
    logger.info("=" * 80)


@app.on_event("shutdown")
async def shutdown_event():
    """Run on API shutdown"""
    logger.info("=" * 80)
    logger.info("Hate Speech Detection API - Shutting Down")
    logger.info("=" * 80)


# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 80)
    print("HATE SPEECH DETECTION API v2.0")
    print("With CSV Processing & User Analytics Dashboard")
    print("=" * 80)
    print("\nStarting server...")
    print("Docs available at: http://localhost:8000/docs")
    print("Health check at: http://localhost:8000/health")
    print("CSV Upload at: http://localhost:8000/docs#/default/process_csv_file_process_csv_post")
    print("=" * 80)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )