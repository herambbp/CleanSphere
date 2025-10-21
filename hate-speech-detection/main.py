"""
FastAPI Backend for Hate Speech Detection System
Provides RESTful API endpoints for classification, explanation, and moderation

Features:
- Single text classification
- Batch classification
- Explainable predictions
- Severity analysis
- Content moderation
- Appeal generation
- Health checks
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import project modules
from inference.explainable_classifier import (
    ExplainableTweetClassifier,
    classify_with_explanation,
    get_explanation_summary
)
from config import CLASS_LABELS, SEVERITY_LEVELS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Hate Speech Detection API",
    description="Advanced hate speech detection with explainable AI, severity analysis, and moderation recommendations",
    version="1.0.0",
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


class ModerationResponse(BaseModel):
    """Content moderation response"""
    input: str
    should_remove: bool
    class_id: int
    reason: str
    severity: Optional[str] = None
    action: Optional[str] = None
    confidence: float
    timestamp: str


class AppealResponse(BaseModel):
    """Appeal response"""
    original_text: str
    classification: str
    confidence: float
    appeal_message: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool
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

# ==================== API ENDPOINTS ====================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Hate Speech Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
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
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


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
        
        # Classify with requested features
        result = clf.classify_with_explanation(
            text=input_data.text,
            include_severity=input_data.include_severity,
            verbose=False
        )
        
        # Format response
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
                
                # Count predictions
                pred = response["prediction"]
                predictions_count[pred] = predictions_count.get(pred, 0) + 1
                
            except Exception as e:
                logger.error(f"Error classifying text '{text[:30]}...': {e}")
                # Continue with other texts
                continue
        
        # Calculate summary
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


@app.post("/moderate", response_model=ModerationResponse)
async def moderate_content(input_data: TextInput):
    """
    Moderate content - returns moderation decision
    
    - **text**: Content to moderate
    
    Returns moderation decision with action recommendation
    """
    try:
        clf = get_classifier()
        
        result = clf.classify_with_explanation(
            text=input_data.text,
            include_severity=True,
            verbose=False
        )
        
        # Determine if content should be removed
        class_id = result.get("class", 2)
        should_remove = class_id in [0, 1]  # Hate speech or Offensive
        
        # Get explanation summary
        explanation_summary = get_explanation_summary(input_data.text)
        
        response = ModerationResponse(
            input=input_data.text,
            should_remove=should_remove,
            class_id=class_id,
            reason=explanation_summary,
            severity=result.get("severity", {}).get("severity_label") if "severity" in result else None,
            action=result.get("action", {}).get("primary_action") if "action" in result else None,
            confidence=result.get("confidence", 0.0),
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Moderated content: {input_data.text[:50]}... -> {'REMOVE' if should_remove else 'ALLOW'}")
        
        return response
        
    except Exception as e:
        logger.error(f"Moderation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Moderation failed: {str(e)}"
        )


@app.post("/appeal", response_model=AppealResponse)
async def generate_appeal_response(input_data: TextInput):
    """
    Generate appeal response for flagged content
    
    - **text**: Flagged content
    
    Returns appeal response with explanation
    """
    try:
        clf = get_classifier()
        
        result = clf.classify_with_explanation(
            text=input_data.text,
            methods=["lime", "keywords"],
            include_severity=False,
            verbose=False
        )
        
        prediction = result.get("prediction", "unknown")
        confidence = result.get("confidence", 0.0)
        
        # Generate appeal message
        appeal_message = f"""Your content was flagged as: {prediction}

Classification confidence: {confidence:.1%}

Our AI system detected potentially harmful content. If you believe this is a mistake, please:

1. Review our community guidelines
2. Provide context for your content
3. Contact our support team with this classification ID

Thank you for your understanding."""
        
        response = AppealResponse(
            original_text=input_data.text,
            classification=prediction,
            confidence=confidence,
            appeal_message=appeal_message,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Generated appeal for: {input_data.text[:50]}...")
        
        return response
        
    except Exception as e:
        logger.error(f"Appeal generation error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Appeal generation failed: {str(e)}"
        )


@app.get("/models/info", response_model=Dict[str, Any])
async def get_model_info():
    """
    Get information about loaded models
    
    Returns model metadata and configuration
    """
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


@app.get("/classes", response_model=Dict[int, str])
async def get_classes():
    """
    Get available classification classes
    
    Returns mapping of class IDs to class names
    """
    return CLASS_LABELS


@app.get("/severity-levels", response_model=Dict[int, str])
async def get_severity_levels():
    """
    Get available severity levels
    
    Returns mapping of severity levels to labels
    """
    return SEVERITY_LEVELS

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
    logger.info("Hate Speech Detection API - Starting Up")
    logger.info("=" * 80)
    logger.info("API is ready to accept requests")
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
    print("HATE SPEECH DETECTION API")
    print("=" * 80)
    print("\nStarting server...")
    print("Docs available at: http://localhost:8000/docs")
    print("Health check at: http://localhost:8000/health")
    print("=" * 80)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )