"""
Main Training Script - Hate Speech Detection Project
Orchestrates the entire training pipeline from data loading to model evaluation

PHASE 1-3: Traditional ML Models + Embeddings
PHASE 4: Severity Classification System
PHASE 5: Deep Learning Models (LSTM, BiLSTM, CNN, BERT) (NEW)
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Import all Phase 1-3 modules
from config import PROJECT_ROOT, FEATURE_COMBINATION
from utils import logger, print_section_header
from data_handler import load_and_split_data
from feature_extractor import FeatureExtractor
from embedding_trainer import train_embeddings, HAS_GENSIM
from models.traditional_ml_trainer import train_traditional_models

# Import Phase 5 modules
try:
    from models.deep_learning_trainer import train_deep_learning_models, HAS_TENSORFLOW
    HAS_DL = True
except ImportError:
    HAS_DL = False
    HAS_TENSORFLOW = False
    logger.warning("Deep learning modules not available. Phase 5 will be skipped.")

from inference.tweet_classifier import TweetClassifier, demo

# ==================== PHASE 1-3: TRADITIONAL ML TRAINING ====================

def phase1_train_traditional_ml():
    """
    Phase 1-3: Train traditional ML models with embeddings.
    
    Pipeline:
    1. Load and split data (70/15/15)
    2. Train embeddings (Word2Vec + FastText) - PHASE 3
    3. Extract features (TF-IDF + Embeddings + linguistic)
    4. Train all ML models
    5. Evaluate and save models
    """
    
    print_section_header("PHASE 1-3: TRADITIONAL ML TRAINING WITH EMBEDDINGS")
    
    try:
        # Step 1: Load and split data
        print_section_header("STEP 1: DATA LOADING & SPLITTING")
        logger.info("Loading data and creating train/val/test splits...")
        
        X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(save_split=True)
        
        logger.info(f"Data loaded successfully:")
        logger.info(f"  Train: {len(y_train)} samples")
        logger.info(f"  Val:   {len(y_val)} samples")
        logger.info(f"  Test:  {len(y_test)} samples")
        
        # Step 2: Train embeddings (PHASE 3)
        print_section_header("STEP 2: TRAINING EMBEDDINGS (PHASE 3)")
        
        if HAS_GENSIM and FEATURE_COMBINATION in ['all', 'embeddings_only']:
            logger.info("Training Word2Vec and FastText on corpus...")
            logger.info(f"Feature combination strategy: {FEATURE_COMBINATION}")
            
            embedding_trainer = train_embeddings(X_train, save=True)
            
            if embedding_trainer:
                logger.info("Embeddings trained successfully!")
                if embedding_trainer.word2vec_model:
                    logger.info(f"  Word2Vec vocabulary: {len(embedding_trainer.word2vec_model.wv):,} words")
                if embedding_trainer.fasttext_model:
                    logger.info(f"  FastText vocabulary: {len(embedding_trainer.fasttext_model.wv):,} words")
            else:
                logger.warning("Could not train embeddings. Continuing with TF-IDF only...")
        else:
            if not HAS_GENSIM:
                logger.warning("Gensim not installed. Skipping embedding training.")
                logger.warning("Install with: pip install gensim")
            logger.info(f"Using feature combination: {FEATURE_COMBINATION}")
            logger.info("Skipping embedding training (not needed for this configuration)")
        
        # Step 3: Extract features
        print_section_header("STEP 3: FEATURE EXTRACTION")
        logger.info("Extracting features from text...")
        logger.info(f"Feature combination strategy: {FEATURE_COMBINATION}")
        
        # Initialize and fit feature extractor on training data
        feature_extractor = FeatureExtractor()
        
        logger.info("Fitting feature extractor on training data...")
        X_train_features = feature_extractor.fit_transform(X_train)
        
        logger.info("Transforming validation data...")
        X_val_features = feature_extractor.transform(X_val)
        
        logger.info("Transforming test data...")
        X_test_features = feature_extractor.transform(X_test)
        
        logger.info(f"Feature extraction complete:")
        logger.info(f"  Feature dimensions: {X_train_features.shape[1]}")
        
        # Log feature breakdown
        if FEATURE_COMBINATION == 'all':
            logger.info(f"  - Char TF-IDF: 200")
            logger.info(f"  - Word TF-IDF: 300")
            if HAS_GENSIM:
                logger.info(f"  - Word2Vec: 100")
                logger.info(f"  - FastText: 100")
            logger.info(f"  - Linguistic: 21")
        elif FEATURE_COMBINATION == 'embeddings_only':
            if HAS_GENSIM:
                logger.info(f"  - Word2Vec: 100")
                logger.info(f"  - FastText: 100")
            logger.info(f"  - Linguistic: 21")
        else:  # tfidf_only
            logger.info(f"  - Char TF-IDF: 200")
            logger.info(f"  - Word TF-IDF: 300")
            logger.info(f"  - Linguistic: 21")
        
        logger.info(f"  Train features: {X_train_features.shape}")
        logger.info(f"  Val features:   {X_val_features.shape}")
        logger.info(f"  Test features:  {X_test_features.shape}")
        
        # Save feature extractor
        logger.info("Saving feature extractor...")
        feature_extractor.save()
        
        # Step 4: Train models
        print_section_header("STEP 4: MODEL TRAINING")
        logger.info("Training all traditional ML models...")
        
        trainer = train_traditional_models(
            X_train=X_train_features,
            y_train=y_train,
            X_val=X_val_features,
            y_val=y_val,
            X_test=X_test_features,
            y_test=y_test,
            save_models=True
        )
        
        # Step 5: Summary
        print_section_header("PHASE 1-3 COMPLETE")
        
        best_name, best_model = trainer.get_best_model()
        logger.info(f"Best performing model: {best_name}")
        
        comparison_df = trainer.evaluator.get_comparison_df()
        if not comparison_df.empty:
            best_row = comparison_df.iloc[0]
            logger.info(f"Best model accuracy: {best_row['Accuracy']:.4f}")
            logger.info(f"Best model F1 (macro): {best_row['F1 (Macro)']:.4f}")
        
        logger.info(f"\nAll models saved to: {PROJECT_ROOT / 'saved_models'}")
        logger.info(f"Feature extractor saved to: {PROJECT_ROOT / 'saved_features'}")
        logger.info(f"Embeddings saved to: {PROJECT_ROOT / 'saved_features'}")
        logger.info(f"Results saved to: {PROJECT_ROOT / 'results'}")
        
        return trainer
        
    except Exception as e:
        logger.error(f"Error in Phase 1-3 training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ==================== PHASE 4: SEVERITY SYSTEM TESTING ====================

def phase4_test_severity_system():
    """
    Phase 4: Test the severity classification and action recommendation system.
    
    No training needed (rule-based system).
    Tests on diverse sample tweets to validate:
    - Severity scoring (0-100)
    - Severity levels (1-5: LOW to EXTREME)
    - Action recommendations
    - Complete integration
    """
    
    print_section_header("PHASE 4: SEVERITY CLASSIFICATION SYSTEM TEST")
    
    try:
        # Import Phase 4 components
        from severity.severity_classifier import SeverityScorer
        from severity.action_recommender import ActionRecommender
        
        logger.info("Initializing severity system components...")
        
        # Initialize
        severity_scorer = SeverityScorer()
        action_recommender = ActionRecommender()
        
        logger.info("Severity system initialized successfully!")
        
        # Test samples (diverse severity levels)
        print_section_header("STEP 1: SEVERITY SCORING TEST")
        
        test_samples = [
            # EXTREME severity
            {
                'text': "I will kill you and your entire family you fucking bitch",
                'expected_severity': 5,
                'description': "EXTREME - Violence + threats + slurs"
            },
            {
                'text': "All n*ggers should be exterminated from this country",
                'expected_severity': 5,
                'description': "EXTREME - Racial slur + genocide language"
            },
            
            # SEVERE severity
            {
                'text': "You worthless scum deserve to die you f*ggot",
                'expected_severity': 4,
                'description': "SEVERE - Dehumanization + death wish + slur"
            },
            
            # HIGH severity
            {
                'text': "You're subhuman trash and everyone hates you",
                'expected_severity': 3,
                'description': "HIGH - Dehumanization + intense hate"
            },
            
            # MODERATE severity
            {
                'text': "You're a fucking idiot and nobody likes you bitch",
                'expected_severity': 2,
                'description': "MODERATE - Profanity + sexist slur"
            },
            
            # LOW severity
            {
                'text': "I disagree with you, this is so frustrating",
                'expected_severity': 1,
                'description': "LOW - Mild negativity, no hate"
            },
            
            # Edge case: Quote (context reduces severity)
            {
                'text': 'He said "I hate you" - can you believe that?',
                'expected_severity': 1,
                'description': "LOW - Quote reduces severity"
            },
            
            # Edge case: Sarcasm
            {
                'text': "Yeah right, like I'd ever kill someone /s",
                'expected_severity': 1,
                'description': "LOW - Sarcasm indicator"
            },
        ]
        
        logger.info(f"Testing severity scoring on {len(test_samples)} samples...\n")
        
        passed = 0
        failed = 0
        
        for i, sample in enumerate(test_samples, 1):
            text = sample['text']
            expected = sample['expected_severity']
            description = sample['description']
            
            # Analyze severity
            result = severity_scorer.analyze_severity(text, verbose=False)
            
            actual = result['severity_level']
            score = result['severity_score']
            label = result['severity_label']
            
            # Check if correct (allow ±1 level tolerance)
            is_correct = abs(actual - expected) <= 1
            
            if is_correct:
                passed += 1
                status = " PASS"
            else:
                failed += 1
                status = "✗ FAIL"
            
            # Print result
            print(f"{i}. [{status}] {description}")
            print(f"   Text: {text[:70]}...")
            print(f"   Expected: Level {expected}, Got: Level {actual} ({label})")
            print(f"   Score: {score}/100")
            print(f"   {result['explanation']}")
            print()
        
        logger.info(f"Severity scoring test: {passed}/{len(test_samples)} passed ({passed/len(test_samples)*100:.1f}%)")
        
        # Test action recommendations
        print_section_header("STEP 2: ACTION RECOMMENDATION TEST")
        
        logger.info("Testing action recommendations for all (class, severity) combinations...\n")
        
        # Display action matrix
        print(action_recommender.get_action_matrix())
        
        # Test critical cases
        print_section_header("STEP 3: CRITICAL CASES VALIDATION")
        
        critical_cases = [
            (0, 5, "Hate speech + EXTREME", "IMMEDIATE_BAN", "CRITICAL"),
            (0, 4, "Hate speech + SEVERE", "IMMEDIATE_BAN", "CRITICAL"),
            (1, 5, "Offensive + EXTREME", "TEMPORARY_BAN_30_DAYS", "HIGH"),
            (2, 1, "Neither + LOW", "NO_ACTION", "NONE"),
        ]
        
        logger.info("Validating critical action recommendations...\n")
        
        all_passed = True
        
        for class_id, severity, description, expected_primary, expected_urgency in critical_cases:
            rec = action_recommender.format_recommendation(class_id, severity)
            
            primary_correct = expected_primary in rec['primary_action']
            urgency_correct = rec['urgency'] == expected_urgency
            
            if primary_correct and urgency_correct:
                status = " PASS"
            else:
                status = "✗ FAIL"
                all_passed = False
            
            print(f"[{status}] {description}")
            print(f"   Action: {rec['action_string']}")
            print(f"   Urgency: {rec['urgency']}")
            print()
        
        if all_passed:
            logger.info(" All critical cases validated successfully!")
        else:
            logger.warning("⚠ Some critical cases failed validation")
        
        # Complete integration test
        print_section_header("STEP 4: COMPLETE INTEGRATION TEST")
        
        logger.info("Testing complete pipeline (severity + actions)...\n")
        
        integration_samples = [
            ("I will kill you bitch", 0),  # Simulated Hate speech
            ("You're a fucking idiot", 1),  # Simulated Offensive
            ("Good morning everyone!", 2),  # Simulated Neither
        ]
        
        for text, simulated_class in integration_samples:
            # Get severity
            severity_result = severity_scorer.analyze_severity(text, verbose=False)
            
            # Get action
            action_result = action_recommender.format_recommendation(
                simulated_class,
                severity_result['severity_level']
            )
            
            # Display complete result
            print(f"Text: {text}")
            print(f"Simulated Class: {action_result['class_name']}")
            print(f"Severity: {severity_result['severity_label']} (Level {severity_result['severity_level']}, Score {severity_result['severity_score']})")
            print(f"Action: {action_result['action_string']}")
            print(f"Urgency: {action_result['urgency']}")
            print(f"Explanation: {severity_result['explanation']}")
            print()
        
        # Summary
        print_section_header("PHASE 4 COMPLETE")
        
        logger.info(" Severity classification system test complete!")
        logger.info("\nSystem Components:")
        logger.info("  - Severity Scorer: Rule-based multi-level detection")
        logger.info("  - Action Recommender: 15 (class, severity) combinations")
        logger.info("  - 8 keyword categories detected")
        logger.info("  - Context-aware adjustments")
        logger.info("  - Explainable outputs")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in Phase 4 testing: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================== PHASE 5: DEEP LEARNING TRAINING ====================

def phase5_train_deep_learning(use_bert: bool = False):
    """
    Phase 5: Train deep learning models (LSTM, BiLSTM, CNN, BERT).
    
    Pipeline:
    1. Load data splits (use existing from Phase 1)
    2. Prepare tokenizer
    3. Train LSTM
    4. Train BiLSTM
    5. Train CNN
    6. Train BERT (optional, slow)
    7. Evaluate and compare all models
    8. Save models and results
    
    Args:
        use_bert: Whether to train BERT model (~30 min on CPU)
    """
    
    print_section_header("PHASE 5: DEEP LEARNING TRAINING")
    
    # Check if TensorFlow is available
    if not HAS_DL:
        logger.error("Deep learning modules not available!")
        logger.error("Install TensorFlow with: pip install tensorflow==2.13.0")
        return False
    
    if not HAS_TENSORFLOW:
        logger.error("TensorFlow not installed!")
        logger.error("Install with: pip install tensorflow==2.13.0")
        return False
    
    try:
        # Step 1: Load data splits
        print_section_header("STEP 1: LOADING DATA SPLITS")
        
        logger.info("Loading saved data splits from Phase 1...")
        
        from data_handler import DataSplitter
        
        try:
            X_train, X_val, X_test, y_train, y_val, y_test = DataSplitter.load_split()
            
            logger.info(f"Data loaded successfully:")
            logger.info(f"  Train: {len(y_train)} samples")
            logger.info(f"  Val:   {len(y_val)} samples")
            logger.info(f"  Test:  {len(y_test)} samples")
            
        except FileNotFoundError:
            logger.warning("Saved splits not found. Loading and splitting data...")
            X_train, X_val, X_test, y_train, y_val, y_test = load_and_split_data(save_split=True)
        
        # Step 2: Train deep learning models
        print_section_header("STEP 2: TRAINING DEEP LEARNING MODELS")
        
        if use_bert:
            logger.info("⚠️  BERT training enabled - this will take ~30 minutes on CPU")
            logger.info("Consider using GPU for faster training (5-10x speedup)")
        else:
            logger.info("BERT training disabled (faster training)")
            logger.info("Use --use-bert flag to enable BERT")
        
        # Train all models
        trainer = train_deep_learning_models(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            use_bert=use_bert,
            save_models=True
        )
        
        # Step 3: Summary
        print_section_header("PHASE 5 COMPLETE")
        
        best_name, best_model = trainer.get_best_model()
        logger.info(f" Best deep learning model: {best_name}")
        
        comparison_df = trainer.evaluator.get_comparison_df()
        if not comparison_df.empty:
            best_row = comparison_df.iloc[0]
            logger.info(f"  Accuracy: {best_row['Accuracy']:.4f}")
            logger.info(f"  F1 Score (Macro): {best_row['F1 (Macro)']:.4f}")
        
        # Compare with traditional ML
        logger.info("\n" + "=" * 70)
        logger.info("PERFORMANCE COMPARISON: TRADITIONAL ML vs DEEP LEARNING")
        logger.info("=" * 70)
        
        try:
            import json
            import pandas as pd
            from config import RESULTS_DIR, COMPARISON_FILE, DL_COMPARISON_FILE
            
            # Load traditional ML results
            metadata_path = RESULTS_DIR / 'model_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    trad_metadata = json.load(f)
                
                best_trad_name = trad_metadata['best_model']['name']
                best_trad_acc = trad_metadata['best_model']['accuracy']
                best_trad_f1 = trad_metadata['best_model']['f1_macro']
                
                # Load DL results
                if not comparison_df.empty:
                    best_dl_acc = comparison_df.iloc[0]['Accuracy']
                    best_dl_f1 = comparison_df.iloc[0]['F1 (Macro)']
                    
                    logger.info(f"\nBest Traditional ML: {best_trad_name}")
                    logger.info(f"  Accuracy: {best_trad_acc:.4f}")
                    logger.info(f"  F1 Score: {best_trad_f1:.4f}")
                    
                    logger.info(f"\nBest Deep Learning: {best_name}")
                    logger.info(f"  Accuracy: {best_dl_acc:.4f}")
                    logger.info(f"  F1 Score: {best_dl_f1:.4f}")
                    
                    improvement_acc = best_dl_acc - best_trad_acc
                    improvement_f1 = best_dl_f1 - best_trad_f1
                    
                    logger.info(f"\nImprovement:")
                    logger.info(f"  Accuracy: {improvement_acc:+.4f} ({improvement_acc*100:+.2f}%)")
                    logger.info(f"  F1 Score: {improvement_f1:+.4f} ({improvement_f1*100:+.2f}%)")
                    
                    if improvement_acc > 0:
                        logger.info("\n Deep learning models outperform traditional ML!")
                    else:
                        logger.info("\n⚠ Traditional ML performed better (consider tuning DL hyperparameters)")
                
                # Create combined comparison
                logger.info("\n" + "=" * 70)
                logger.info("COMBINED MODEL COMPARISON")
                logger.info("=" * 70)
                
                try:
                    trad_df = pd.read_csv(COMPARISON_FILE)
                    dl_df = pd.read_csv(DL_COMPARISON_FILE)
                    
                    # Add model type column
                    trad_df['Type'] = 'Traditional ML'
                    dl_df['Type'] = 'Deep Learning'
                    
                    # Combine
                    combined_df = pd.concat([trad_df, dl_df], ignore_index=True)
                    combined_df = combined_df.sort_values('Accuracy', ascending=False)
                    
                    print("\n" + combined_df.to_string(index=False))
                    
                    # Save combined comparison
                    combined_path = RESULTS_DIR / 'all_models_comparison.csv'
                    combined_df.to_csv(combined_path, index=False)
                    logger.info(f"\n Combined comparison saved to {combined_path}")
                    
                except Exception as e:
                    logger.warning(f"Could not create combined comparison: {e}")
        
        except Exception as e:
            logger.warning(f"Could not compare with traditional ML: {e}")
        
        logger.info("\n" + "=" * 70)
        logger.info(f"All DL models saved to: {PROJECT_ROOT / 'saved_models' / 'deep_learning'}")
        logger.info(f"Tokenizer saved to: {PROJECT_ROOT / 'saved_models' / 'deep_learning'}")
        logger.info(f"Results saved to: {PROJECT_ROOT / 'results'}")
        
        return trainer
        
    except Exception as e:
        logger.error(f"Error in Phase 5 training: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================== TESTING ====================

def test_classifier():
    """Test the trained classifier with sample tweets."""
    print_section_header("TESTING CLASSIFIER")
    
    try:
        logger.info("Testing classifier with sample tweets...")
        demo()
        
    except Exception as e:
        logger.error(f"Error testing classifier: {e}")
        import traceback
        traceback.print_exc()

# ==================== MAIN ====================

def main(
    skip_phase1: bool = False,
    skip_phase4: bool = False,
    run_phase5: bool = False,
    use_bert: bool = False
):
    """
    Main entry point for training pipeline.
    
    Args:
        skip_phase1: Skip Phase 1-3 (traditional ML)
        skip_phase4: Skip Phase 4 (severity testing)
        run_phase5: Run Phase 5 (deep learning)
        use_bert: Include BERT in Phase 5
    """
    
    print("=" * 80)
    print("HATE SPEECH DETECTION - TRAINING PIPELINE")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT}")
    print("=" * 80)
    
    # Check if data files exist
    from config import LABELED_DATA_CSV, COMBINED_DATA_CSV
    
    if not LABELED_DATA_CSV.exists() and not COMBINED_DATA_CSV.exists():
        logger.error("No data files found!")
        logger.error(f"Please place data files in: {LABELED_DATA_CSV.parent}")
        logger.error("Expected files:")
        logger.error(f"  - {LABELED_DATA_CSV.name}")
        logger.error(f"  - {COMBINED_DATA_CSV.name} (optional)")
        sys.exit(1)
    
    # Phase 1-3: Traditional ML
    if not skip_phase1:
        trainer_trad = phase1_train_traditional_ml()
    else:
        logger.info("Skipping Phase 1-3 (Traditional ML)")
        trainer_trad = None
    
    # Phase 4: Severity System
    if not skip_phase4:
        phase4_success = phase4_test_severity_system()
    else:
        logger.info("Skipping Phase 4 (Severity System)")
        phase4_success = None
    
    # Phase 5: Deep Learning (NEW)
    if run_phase5:
        trainer_dl = phase5_train_deep_learning(use_bert=use_bert)
    else:
        logger.info("\nSkipping Phase 5 (Deep Learning)")
        logger.info("Use --phase5 flag to enable deep learning training")
        trainer_dl = None
    
    # Test the classifier
    if not skip_phase1 or not skip_phase4:
        test_classifier()
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 80)
    
    print("\n📊 Completed Phases:")
    if not skip_phase1:
        print("   Phase 1-3: Traditional ML + Embeddings")
        if trainer_trad:
            comparison_df = trainer_trad.evaluator.get_comparison_df()
            if not comparison_df.empty:
                best = comparison_df.iloc[0]
                print(f"    Best: {best['Model']} ({best['Accuracy']:.4f} accuracy)")
    
    if not skip_phase4:
        print("   Phase 4: Severity Classification + Action Recommendations")
    
    if run_phase5 and trainer_dl:
        print("   Phase 5: Deep Learning Models (LSTM, BiLSTM, CNN" + (", BERT" if use_bert else "") + ")")
        comparison_df = trainer_dl.evaluator.get_comparison_df()
        if not comparison_df.empty:
            best = comparison_df.iloc[0]
            print(f"    Best: {best['Model']} ({best['Accuracy']:.4f} accuracy)")
    
    print("\n📁 Output Locations:")
    print(f"  Models: {PROJECT_ROOT / 'saved_models'}")
    print(f"  Features: {PROJECT_ROOT / 'saved_features'}")
    print(f"  Results: {PROJECT_ROOT / 'results'}")
    
    print("\n🚀 Next Steps:")
    print("  1. Check results in 'results' folder")
    print("  2. Review model comparisons:")
    print(f"     - Traditional ML: results/model_comparison.csv")
    if run_phase5:
        print(f"     - Deep Learning: results/dl_model_comparison.csv")
        print(f"     - Combined: results/all_models_comparison.csv")
    
    print("\n  3. Use the classifier:")
    print("     from inference.tweet_classifier import TweetClassifier")
    print("     classifier = TweetClassifier()")
    print("     result = classifier.classify_with_severity('Your tweet')")
    
    print("\n📋 Future Phases:")
    if not run_phase5:
        print("  ⏭  Phase 5: Deep Learning (use --phase5 flag)")
    print("  ⏭  Phase 6: Explainable AI (LIME, SHAP)")
    print("  ⏭  Phase 7: User Analytics and CSV processing")
    print("  ⏭  Phase 8: Model Management and A/B testing")
    
    print("=" * 80)

# ==================== USAGE EXAMPLES ====================

def usage_examples():
    """
    Examples of how to use the trained models.
    """
    
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    
    print("\n1. Classify a single tweet:")
    print("-" * 40)
    print("""
from inference.tweet_classifier import classify_tweet

result = classify_tweet("I hate when people do that")
print(result)
# Output: {'text': '...', 'prediction': 'Offensive language', 'confidence': 0.85, ...}
    """)
    
    print("\n2. Classify with severity + actions (Phase 4):")
    print("-" * 40)
    print("""
from inference.tweet_classifier import TweetClassifier

classifier = TweetClassifier()
result = classifier.classify_with_severity("I will kill you")

print(f"Class: {result['prediction']}")
print(f"Severity: {result['severity']['severity_label']}")
print(f"Action: {result['action']['action_string']}")
print(f"Urgency: {result['action']['urgency']}")
    """)
    
    print("\n3. Use deep learning model (Phase 5):")
    print("-" * 40)
    print("""
from models.deep_learning.lstm_model import LSTMModel
from models.deep_learning.text_tokenizer import TextTokenizer

# Load tokenizer and model
tokenizer = TextTokenizer.load()
model = LSTMModel.load()

# Classify text
text = "You're an idiot"
sequences = tokenizer.texts_to_padded_sequences([text])
prediction = model.predict(sequences)

print(f"Prediction: {prediction[0]}")  # 0=Hate, 1=Offensive, 2=Neither
    """)
    
    print("\n4. Batch processing with all features:")
    print("-" * 40)
    print("""
import pandas as pd
from inference.tweet_classifier import TweetClassifier

# Load your tweets
df = pd.read_csv('tweets.csv')
tweets = df['text'].tolist()

# Classify with severity
classifier = TweetClassifier()
results = []

for tweet in tweets:
    result = classifier.classify_with_severity(tweet)
    results.append({
        'text': tweet,
        'class': result['prediction'],
        'confidence': result['confidence'],
        'severity': result['severity']['severity_label'],
        'action': result['action']['primary_action']
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('classified_tweets.csv', index=False)
    """)

# ==================== RUN ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Hate Speech Detection Models')
    
    # Skip options
    parser.add_argument('--skip-phase1', action='store_true',
                       help='Skip Phase 1-3 (Traditional ML)')
    parser.add_argument('--skip-phase4', action='store_true',
                       help='Skip Phase 4 (Severity System)')
    
    # Phase 5 options
    parser.add_argument('--phase5', action='store_true',
                       help='Run Phase 5 (Deep Learning Models)')
    parser.add_argument('--use-bert', action='store_true',
                       help='Include BERT in Phase 5 (slow, ~30 min)')
    
    # Test options
    parser.add_argument('--test-only', action='store_true',
                       help='Only test the classifier (skip training)')
    parser.add_argument('--examples', action='store_true',
                       help='Show usage examples')
    
    args = parser.parse_args()
    
    if args.examples:
        usage_examples()
    elif args.test_only:
        test_classifier()
    else:
        main(
            skip_phase1=args.skip_phase1,
            skip_phase4=args.skip_phase4,
            run_phase5=args.phase5,
            use_bert=args.use_bert
        )