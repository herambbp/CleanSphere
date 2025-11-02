"""
FastAPI Backend for Hate Speech Detection System with CSV Processing and Intelligent Verification
Provides RESTful API endpoints for classification, explanation, moderation, and CSV batch processing
Now includes OpenAI GPT-4o-mini verification for intelligent feedback and corrections
**NEW: Digital Footprint Scoring System for comprehensive user behavior profiling**

Features:
- Single text classification
- Batch classification
- CSV file processing with user analytics
- Explainable predictions
- Severity analysis
- Content moderation
- User dashboard analytics
- Intelligent verification mode with OpenAI GPT-4o-mini
- **NEW: Digital Footprint Scoring & Risk Profiling**
- **NEW: Temporal behavior analysis & pattern detection**
- **NEW: User comparison & historical tracking**
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
import os
import json

# OpenAI Integration
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: OpenAI library not installed. Install with: pip install openai")

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

# ==================== NEW: DIGITAL FOOTPRINT INTEGRATION ====================
# Import Digital Footprint Scoring System
try:
    from digital_footprint_scorer import (
        DigitalFootprintScorer,
        integrate_with_csv_processor
    )
    HAS_DIGITAL_FOOTPRINT = True
except ImportError:
    HAS_DIGITAL_FOOTPRINT = False
    print("Warning: digital_footprint_scorer.py not found. Digital footprint features will be disabled.")
    print("To enable: Place digital_footprint_scorer.py in the project root directory")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== OPENAI CONFIGURATION ====================

# Initialize OpenAI client (requires OPENAI_API_KEY environment variable)
openai_client = None
if HAS_OPENAI:
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized successfully")
        else:
            logger.warning("OPENAI_API_KEY not found. Intelligent mode will be disabled.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Hate Speech Detection API with Intelligent Verification & Digital Footprint Scoring",
    description="Advanced hate speech detection with explainable AI, severity analysis, CSV batch processing, user analytics dashboard, OpenAI verification, and comprehensive digital footprint scoring",
    version="4.0.0",
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

# ==================== GLOBAL INSTANCES ====================

classifier = None

# NEW: Global Digital Footprint Scorer
footprint_scorer = None
if HAS_DIGITAL_FOOTPRINT:
    try:
        footprint_scorer = DigitalFootprintScorer()
        logger.info("Digital Footprint Scorer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Digital Footprint Scorer: {e}")
        HAS_DIGITAL_FOOTPRINT = False

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

# ==================== INTELLIGENT VERIFICATION ====================

async def verify_with_gpt(text: str, model_prediction: Dict) -> Dict:
    """
    Verify model prediction using OpenAI GPT-4o-mini
    
    Args:
        text: Original text that was classified
        model_prediction: The prediction from our model
        
    Returns:
        Dict containing verification results and feedback
    """
    if not openai_client:
        return {
            "verified": False,
            "error": "OpenAI client not available",
            "requires_api_key": True
        }
    
    try:
        # Construct prompt for GPT
        prompt = f"""You are an expert content moderator tasked with verifying hate speech detection results.

Original Text: "{text}"

Our Model's Classification:
- Prediction: {model_prediction.get('prediction', 'Unknown')}
- Confidence: {model_prediction.get('confidence', 0):.2%}
- Severity: {model_prediction.get('severity', {}).get('label', 'N/A')} ({model_prediction.get('severity', {}).get('score', 0)}/100)

Please analyze this text and provide:
1. Your classification (choose one: "Hate speech", "Offensive language", or "Neither")
2. Your confidence level (0.0 to 1.0)
3. Agreement status: Do you agree with our model? (yes/no)
4. Detailed feedback explaining your reasoning
5. If you disagree, explain what the model got wrong
6. Suggested severity score (0-100)

Respond in JSON format:
{{
    "gpt_classification": "your classification here",
    "gpt_confidence": 0.0 to 1.0,
    "agrees_with_model": true/false,
    "feedback": "detailed explanation",
    "correction_reason": "explanation if disagreement",
    "suggested_severity": 0-100,
    "confidence_in_verification": 0.0 to 1.0
}}"""

        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert content moderator with deep understanding of hate speech, offensive language, and harmful content. Provide accurate, unbiased analysis."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=500
        )
        
        # Parse response
        gpt_response = json.loads(response.choices[0].message.content)
        
        # Determine final classification
        model_pred = model_prediction.get('prediction', '')
        gpt_pred = gpt_response.get('gpt_classification', '')
        agrees = gpt_response.get('agrees_with_model', False)
        
        verification_result = {
            "verified": True,
            "agrees_with_model": agrees,
            "model_prediction": model_pred,
            "gpt_prediction": gpt_pred,
            "gpt_confidence": gpt_response.get('gpt_confidence', 0.0),
            "final_prediction": model_pred if agrees else gpt_pred,
            "feedback": gpt_response.get('feedback', ''),
            "correction_reason": gpt_response.get('correction_reason', '') if not agrees else None,
            "suggested_severity": gpt_response.get('suggested_severity', 0),
            "confidence_in_verification": gpt_response.get('confidence_in_verification', 0.0),
            "is_corrected": not agrees,
            "verification_timestamp": datetime.utcnow().isoformat()
        }
        
        logger.info(f"GPT Verification: Model={model_pred}, GPT={gpt_pred}, Agrees={agrees}")
        
        return verification_result
        
    except Exception as e:
        logger.error(f"GPT verification error: {e}")
        return {
            "verified": False,
            "error": str(e),
            "message": "Verification failed"
        }


def map_prediction_to_class_id(prediction: str) -> int:
    """Map prediction string to class ID"""
    mapping = {
        "Hate speech": 0,
        "Offensive language": 1,
        "Neither": 2
    }
    return mapping.get(prediction, -1)

# ==================== REQUEST/RESPONSE MODELS ====================

class TextInput(BaseModel):
    """Single text input"""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")
    include_severity: bool = Field(default=True, description="Include severity analysis")
    include_explanation: bool = Field(default=True, description="Include explainable AI insights")
    intelligent_mode: bool = Field(default=False, description="Enable OpenAI GPT verification")
    
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
    intelligent_mode: bool = Field(default=False, description="Enable OpenAI GPT verification")
    
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
    verification: Optional[Dict[str, Any]] = None  # GPT verification results
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


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool
    csv_processor_available: bool
    openai_available: bool
    intelligent_mode_available: bool
    digital_footprint_available: bool  # NEW
    version: str


# ==================== NEW: DIGITAL FOOTPRINT MODELS ====================

class DigitalFootprintRequest(BaseModel):
    """Request model for digital footprint analysis"""
    user_id: str = Field(..., description="User identifier")
    comments: List[Dict[str, Any]] = Field(
        ...,
        description="List of comment dicts with keys: timestamp, comment, prediction, confidence"
    )
    account_age_days: Optional[int] = Field(None, description="Account age in days")
    platform_data: Optional[Dict[str, Any]] = Field(
        None,
        description="Platform metadata (followers, engagement_rate, is_verified)"
    )


class FootprintComparisonRequest(BaseModel):
    """Request model for comparing footprints"""
    user_id_1: str
    user_id_2: str
    footprint_1: Optional[Dict] = None
    footprint_2: Optional[Dict] = None


class BatchFootprintRequest(BaseModel):
    """Request model for batch footprint analysis"""
    users: Dict[str, List[Dict[str, Any]]] = Field(
        ...,
        description="Dict mapping user_id to list of their comments"
    )
    account_metadata: Optional[Dict[str, Dict]] = Field(
        None,
        description="Optional metadata for each user"
    )


class DigitalFootprintResponse(BaseModel):
    """Response model for digital footprint analysis"""
    user_id: str
    footprint_score: float
    risk_level: str
    score_breakdown: Dict[str, float]
    statistics: Dict[str, Any]
    patterns: Dict[str, Any]
    recommendations: List[str]
    account_info: Dict[str, Any]
    analysis_timestamp: str
    visualization: Optional[str] = None


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

# ==================== BASIC API ENDPOINTS ====================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint - API information"""
    return {
        "message": "Hate Speech Detection API with Intelligent Verification & Digital Footprint Scoring",
        "version": "4.0.0",
        "docs": "/docs",
        "health": "/health",
        "features": "Classification, Severity Analysis, CSV Processing, User Analytics, OpenAI Verification, Digital Footprint Scoring"
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
            openai_available=HAS_OPENAI and openai_client is not None,
            intelligent_mode_available=HAS_OPENAI and openai_client is not None,
            digital_footprint_available=HAS_DIGITAL_FOOTPRINT,
            version="4.0.0"
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
    - **intelligent_mode**: Enable OpenAI GPT-4o-mini verification (default: false)
    
    Returns classification, confidence, probabilities, and optional severity/explanation/verification
    """
    try:
        clf = get_classifier()
        
        # Get model prediction
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
        
        # If intelligent mode is enabled, verify with GPT
        if input_data.intelligent_mode:
            if not openai_client:
                response["verification"] = {
                    "error": "Intelligent mode not available. Please set OPENAI_API_KEY environment variable.",
                    "verified": False
                }
            else:
                verification = await verify_with_gpt(input_data.text, response)
                response["verification"] = verification
                
                # If GPT disagrees, update the final prediction
                if verification.get("is_corrected", False):
                    response["original_prediction"] = response["prediction"]
                    response["prediction"] = verification["final_prediction"]
                    response["class_id"] = map_prediction_to_class_id(verification["final_prediction"])
                    logger.info(f"Prediction corrected: {response['original_prediction']} -> {response['prediction']}")
        
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
    - **intelligent_mode**: Enable OpenAI GPT verification (default: false)
    
    Returns classifications for all texts plus summary statistics
    """
    try:
        clf = get_classifier()
        
        results = []
        predictions_count = {}
        corrections_count = 0
        
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
                
                # Intelligent mode verification
                if input_data.intelligent_mode and openai_client:
                    verification = await verify_with_gpt(text, response)
                    response["verification"] = verification
                    
                    if verification.get("is_corrected", False):
                        response["original_prediction"] = response["prediction"]
                        response["prediction"] = verification["final_prediction"]
                        response["class_id"] = map_prediction_to_class_id(verification["final_prediction"])
                        corrections_count += 1
                
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
            "neither_count": predictions_count.get("Neither", 0),
            "intelligent_mode_enabled": input_data.intelligent_mode,
            "corrections_made": corrections_count if input_data.intelligent_mode else None
        }
        
        logger.info(f"Batch classified {len(results)} texts (Corrections: {corrections_count})")
        
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

@app.post("/process-csv")
async def process_csv_file(
    file: UploadFile = File(...),
    intelligent_mode: bool = False
):
    """
    Process uploaded CSV file with user comments
    
    Expected CSV format:
    - name/user/username: User identifier
    - timestamp/date/time: Comment timestamp  
    - comment/text/message: Comment text
    
    Optional query parameter:
    - intelligent_mode: Enable OpenAI verification for all comments (default: false)
    
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
        
        logger.info(f"Processing uploaded CSV: {file.filename} (Intelligent Mode: {intelligent_mode})")
        
        # Process CSV
        processor = CSVProcessor()
        results_df = processor.process_csv(temp_path)
        
        # If intelligent mode, verify samples (limit to avoid excessive API calls)
        verification_summary = None
        if intelligent_mode and openai_client:
            logger.info("Running intelligent verification on sample comments...")
            # Verify top 10 high-risk comments
            high_risk = results_df[results_df['prediction'].isin(['Hate speech', 'Offensive language'])].head(10)
            
            verifications = []
            corrections = 0
            for _, row in high_risk.iterrows():
                model_pred = {
                    'prediction': row['prediction'],
                    'confidence': row['confidence'],
                    'severity': {'label': row.get('severity', 'N/A'), 'score': 0}
                }
                verification = await verify_with_gpt(row['comment'], model_pred)
                verifications.append(verification)
                if verification.get('is_corrected', False):
                    corrections += 1
            
            verification_summary = {
                "samples_verified": len(verifications),
                "corrections_made": corrections,
                "agreement_rate": (len(verifications) - corrections) / len(verifications) if verifications else 0,
                "note": "Sample verification performed on top 10 high-risk comments"
            }
        
        # Get analytics
        user_analytics = processor.get_user_analytics()
        summary = processor.generate_summary_report()
        
        logger.info(f"CSV processed: {len(results_df)} comments from {len(user_analytics)} users")
        
        # Prepare response
        response = {
            "status": "success",
            "total_comments": int(len(results_df)),
            "total_users": int(len(user_analytics)),
            "summary": make_json_serializable(summary),
            "user_analytics": make_json_serializable(user_analytics),
            "top_risk_users": make_json_serializable(get_top_risk_users(user_analytics, 10)),
            "intelligent_mode_enabled": intelligent_mode,
            "verification_summary": verification_summary,
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


# ==================== NEW: DIGITAL FOOTPRINT ENDPOINTS ====================

@app.post("/analyze-digital-footprint", response_model=DigitalFootprintResponse)
async def analyze_digital_footprint(request: DigitalFootprintRequest):
    """
    Analyze a user's digital footprint based on their comment history.
    
    **Required Fields:**
    - user_id: Unique user identifier
    - comments: List of comment objects with timestamp, comment, prediction, confidence
    
    **Optional Fields:**
    - account_age_days: Age of the account
    - platform_data: Additional platform metrics (followers, engagement, etc.)
    
    **Returns:**
    - Comprehensive footprint analysis with risk score and recommendations
    """
    if not HAS_DIGITAL_FOOTPRINT:
        raise HTTPException(
            status_code=501,
            detail="Digital Footprint Scoring not available. Please ensure digital_footprint_scorer.py is in the project directory."
        )
    
    try:
        # Convert comments to DataFrame
        comments_df = pd.DataFrame(request.comments)
        
        # Ensure timestamp is datetime
        if 'timestamp' in comments_df.columns:
            comments_df['timestamp'] = pd.to_datetime(comments_df['timestamp'])
        else:
            # Create synthetic timestamps if not provided
            comments_df['timestamp'] = pd.date_range(
                end=datetime.now(),
                periods=len(comments_df),
                freq='1H'
            )
        
        # Validate required columns
        required_cols = ['comment', 'prediction', 'confidence']
        missing_cols = [col for col in required_cols if col not in comments_df.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {missing_cols}"
            )
        
        # Calculate footprint
        footprint = footprint_scorer.calculate_footprint_score(
            user_id=request.user_id,
            comments_df=comments_df,
            account_age_days=request.account_age_days,
            platform_data=request.platform_data
        )
        
        # Add visualization
        footprint['visualization'] = footprint_scorer.visualize_footprint(footprint)
        
        logger.info(
            f"Digital footprint analyzed: {request.user_id} - "
            f"Score: {footprint['footprint_score']}, Risk: {footprint['risk_level']}"
        )
        
        return DigitalFootprintResponse(**footprint)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Digital footprint analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze digital footprint: {str(e)}"
        )


@app.get("/user/{user_id}/footprint", response_model=DigitalFootprintResponse)
async def get_user_footprint(user_id: str):
    """
    Retrieve cached digital footprint for a user.
    
    Returns the most recent footprint analysis if available.
    If no cached data exists, returns 404.
    """
    if not HAS_DIGITAL_FOOTPRINT:
        raise HTTPException(
            status_code=501,
            detail="Digital Footprint Scoring not available."
        )
    
    if user_id not in footprint_scorer.user_footprints:
        raise HTTPException(
            status_code=404,
            detail=f"No footprint data found for user: {user_id}. "
                   f"Please analyze first using /analyze-digital-footprint"
        )
    
    footprint = footprint_scorer.user_footprints[user_id]
    footprint['visualization'] = footprint_scorer.visualize_footprint(footprint)
    
    return DigitalFootprintResponse(**footprint)


@app.post("/compare-footprints")
async def compare_footprints(request: FootprintComparisonRequest):
    """
    Compare digital footprints of two users or same user across time periods.
    
    **Options:**
    1. Provide user_id_1 and user_id_2 to compare cached footprints
    2. Provide footprint_1 and footprint_2 objects directly for custom comparison
    
    **Returns:**
    - Score difference
    - Risk level change
    - Behavior trend (improving/worsening/stable)
    - Component-wise breakdown
    """
    if not HAS_DIGITAL_FOOTPRINT:
        raise HTTPException(
            status_code=501,
            detail="Digital Footprint Scoring not available."
        )
    
    try:
        # Get footprints
        if request.footprint_1 and request.footprint_2:
            footprint1 = request.footprint_1
            footprint2 = request.footprint_2
        else:
            # Retrieve from cache
            if request.user_id_1 not in footprint_scorer.user_footprints:
                raise HTTPException(
                    status_code=404,
                    detail=f"No footprint data for user: {request.user_id_1}"
                )
            if request.user_id_2 not in footprint_scorer.user_footprints:
                raise HTTPException(
                    status_code=404,
                    detail=f"No footprint data for user: {request.user_id_2}"
                )
            
            footprint1 = footprint_scorer.user_footprints[request.user_id_1]
            footprint2 = footprint_scorer.user_footprints[request.user_id_2]
        
        # Compare
        comparison = footprint_scorer.compare_footprints(footprint1, footprint2)
        
        logger.info(
            f"Footprint comparison: {comparison['user1']} vs {comparison['user2']} - "
            f"Trend: {comparison['behavior_trend']}"
        )
        
        return comparison
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Footprint comparison error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare footprints: {str(e)}"
        )


@app.post("/batch-footprint-analysis")
async def batch_footprint_analysis(request: BatchFootprintRequest):
    """
    Analyze digital footprints for multiple users in batch.
    
    **Input:**
    - users: Dict mapping user_id to their list of comments
    - account_metadata: Optional dict with account info per user
    
    **Returns:**
    - DataFrame (as JSON) with all users ranked by footprint score
    - Individual footprint details for each user
    """
    if not HAS_DIGITAL_FOOTPRINT:
        raise HTTPException(
            status_code=501,
            detail="Digital Footprint Scoring not available."
        )
    
    try:
        # Convert user data to DataFrames
        users_data = {}
        for user_id, comments in request.users.items():
            df = pd.DataFrame(comments)
            
            # Ensure timestamp
            if 'timestamp' not in df.columns:
                df['timestamp'] = pd.date_range(
                    end=datetime.now(),
                    periods=len(df),
                    freq='1H'
                )
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            users_data[user_id] = df
        
        # Batch analyze
        results_df = footprint_scorer.batch_analyze_users(
            users_data=users_data,
            account_metadata=request.account_metadata
        )
        
        # Get individual footprints
        individual_footprints = {
            user_id: footprint_scorer.user_footprints[user_id]
            for user_id in request.users.keys()
            if user_id in footprint_scorer.user_footprints
        }
        
        logger.info(f"Batch analyzed {len(request.users)} users")
        
        return {
            "summary": results_df.to_dict(orient='records'),
            "total_users": len(request.users),
            "high_risk_users": len(results_df[results_df['risk_level'].isin(['critical', 'high'])]),
            "individual_footprints": make_json_serializable(individual_footprints),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch footprint analysis error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform batch analysis: {str(e)}"
        )


@app.post("/process-csv-with-footprint")
async def process_csv_with_footprint(
    file: UploadFile = File(...),
    include_footprint: bool = True,
    intelligent_mode: bool = False
):
    """
    Enhanced CSV processing that includes digital footprint analysis.
    
    Same as /process-csv but adds comprehensive digital footprint scores
    for each user in the dataset.
    
    **Parameters:**
    - file: CSV file to process
    - include_footprint: Include digital footprint analysis (default: true)
    - intelligent_mode: Enable OpenAI verification (default: false)
    
    **Returns:**
    - All standard CSV processing results
    - Digital footprint analysis for each user
    - Enhanced risk rankings
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
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        logger.info(
            f"Processing CSV with footprint analysis: {file.filename} "
            f"(Footprint: {include_footprint}, Intelligent Mode: {intelligent_mode})"
        )
        
        # Process CSV (existing code)
        processor = CSVProcessor()
        results_df = processor.process_csv(temp_path)
        
        # Get standard analytics
        user_analytics = processor.get_user_analytics()
        summary = processor.generate_summary_report()
        
        # Add digital footprint analysis
        digital_footprints = {}
        if include_footprint and HAS_DIGITAL_FOOTPRINT:
            logger.info("Calculating digital footprints for all users...")
            digital_footprints = integrate_with_csv_processor(results_df)
            
            # Add footprint scores to user analytics
            for user_id, footprint in digital_footprints.items():
                if user_id in user_analytics:
                    user_analytics[user_id]['digital_footprint_score'] = footprint['footprint_score']
                    user_analytics[user_id]['footprint_risk_level'] = footprint['risk_level']
                    user_analytics[user_id]['top_recommendation'] = (
                        footprint['recommendations'][0] if footprint['recommendations'] else ''
                    )
        
        # Create enhanced top risk users list (sorted by footprint score)
        top_risk_users_enhanced = []
        if digital_footprints:
            for user_id, footprint in sorted(
                digital_footprints.items(),
                key=lambda x: x[1]['footprint_score'],
                reverse=True
            )[:10]:
                top_risk_users_enhanced.append({
                    'user': user_id,
                    'footprint_score': footprint['footprint_score'],
                    'risk_level': footprint['risk_level'],
                    'hate_speech_count': footprint['statistics']['hate_speech_count'],
                    'total_comments': footprint['statistics']['total_comments'],
                    'temporal_trend': footprint['score_breakdown']['temporal_trend'],
                    'top_recommendations': footprint['recommendations'][:3]
                })
        
        logger.info(
            f"CSV processed with footprints: {len(results_df)} comments, "
            f"{len(digital_footprints)} users analyzed"
        )
        
        # Prepare response
        response = {
            "status": "success",
            "total_comments": int(len(results_df)),
            "total_users": int(len(user_analytics)),
            "summary": make_json_serializable(summary),
            "user_analytics": make_json_serializable(user_analytics),
            "digital_footprints": make_json_serializable(digital_footprints) if include_footprint else {},
            "top_risk_users": top_risk_users_enhanced if digital_footprints else make_json_serializable(
                get_top_risk_users(user_analytics, 10)
            ),
            "footprint_analysis_included": include_footprint and HAS_DIGITAL_FOOTPRINT,
            "intelligent_mode_enabled": intelligent_mode,
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
        # Cleanup
        if temp_path and Path(temp_path).exists():
            try:
                Path(temp_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")


@app.get("/footprint-info")
async def get_footprint_info():
    """
    Get information about the digital footprint scoring system.
    
    Returns details about:
    - Score components and weights
    - Risk level thresholds
    - Available patterns detected
    """
    if not HAS_DIGITAL_FOOTPRINT:
        raise HTTPException(
            status_code=501,
            detail="Digital Footprint Scoring not available."
        )
    
    from digital_footprint_scorer import DigitalFootprintScorer
    
    return {
        "version": "1.0.0",
        "description": "Digital Footprint Scoring System for comprehensive user risk profiling",
        "score_components": {
            name: {
                "weight": weight,
                "description": {
                    'content_severity': "Severity of content posted by user",
                    'temporal_trend': "Behavior improvement or deterioration over time",
                    'frequency': "How often user posts problematic content",
                    'recency': "How recent is the problematic activity",
                    'consistency': "Consistency of problematic behavior vs outliers",
                    'engagement_risk': "Risk based on follower count and engagement"
                }.get(name, "")
            }
            for name, weight in DigitalFootprintScorer.WEIGHTS.items()
        },
        "risk_levels": DigitalFootprintScorer.RISK_THRESHOLDS,
        "patterns_detected": [
            "escalation_detected",
            "improvement_detected",
            "bursty_behavior",
            "time_of_day_pattern",
            "targeted_harassment"
        ],
        "total_users_analyzed": len(footprint_scorer.user_footprints),
        "cache_size": len(footprint_scorer.user_footprints)
    }


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
            "intelligent_mode_available": openai_client is not None,
            "digital_footprint_available": HAS_DIGITAL_FOOTPRINT,
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
    logger.info("Hate Speech Detection API v4.0 - Starting Up")
    logger.info("=" * 80)
    logger.info("Features: Classification, Severity Analysis, CSV Processing, User Analytics")
    logger.info(f"Intelligent Mode (OpenAI): {'Available' if openai_client else 'Not Available'}")
    logger.info(f"CSV Processor: {'Available' if HAS_CSV_PROCESSOR else 'Not Available'}")
    logger.info(f"Digital Footprint Scoring: {'Available' if HAS_DIGITAL_FOOTPRINT else 'Not Available'}")
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
    print("HATE SPEECH DETECTION API v4.0")
    print("With CSV Processing, User Analytics, Intelligent Verification & Digital Footprint Scoring")
    print("=" * 80)
    print("\nStarting server...")
    print("Docs available at: http://localhost:8000/docs")
    print("Health check at: http://localhost:8000/health")
    print(f"Intelligent Mode: {'Available' if openai_client else 'Requires OPENAI_API_KEY'}")
    print(f"Digital Footprint: {'Available' if HAS_DIGITAL_FOOTPRINT else 'Requires digital_footprint_scorer.py'}")
    print("=" * 80)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True only during development
        log_level="info"
    )