"""
File Logger Wrapper for main_train_enhanced.py
Add this at the top of your main_train_enhanced.py to enable file logging
All console output will also be saved to results/training.log
"""

import sys
from pathlib import Path
from datetime import datetime


class DualLogger:
    """Logger that writes to both console and file simultaneously"""
    
    def __init__(self, log_file='results/training.log'):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize log file with header
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
        
        # Store original stdout and stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create file handle
        self.log_handle = open(self.log_file, 'a', encoding='utf-8', buffering=1)
    
    def write(self, message):
        """Write to both console and file"""
        # Write to console
        self.original_stdout.write(message)
        self.original_stdout.flush()
        
        # Write to file
        self.log_handle.write(message)
        self.log_handle.flush()
    
    def flush(self):
        """Flush both outputs"""
        self.original_stdout.flush()
        self.log_handle.flush()
    
    def close(self):
        """Close file handle"""
        if hasattr(self, 'log_handle'):
            self.log_handle.close()
    
    def __enter__(self):
        """Context manager entry"""
        sys.stdout = self
        sys.stderr = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.close()


def setup_file_logging(log_file='results/training.log'):
    """
    Setup file logging for the entire script.
    Call this at the beginning of main() function.
    
    Returns:
        DualLogger instance (for cleanup later)
    """
    dual_logger = DualLogger(log_file)
    sys.stdout = dual_logger
    sys.stderr = dual_logger
    
    print(f"[LOGGING] Output will be logged to: {dual_logger.log_file.absolute()}\n")
    
    return dual_logger


def cleanup_file_logging(dual_logger):
    """
    Cleanup file logging.
    Call this at the end of main() function or in a finally block.
    """
    if dual_logger:
        sys.stdout = dual_logger.original_stdout
        sys.stderr = dual_logger.original_stderr
        dual_logger.close()
        print(f"\n[LOGGING] Training log saved to: {dual_logger.log_file.absolute()}")


# ========== USAGE IN main_train_enhanced.py ==========

"""
OPTION 1: Add to the main() function (simple):

def main(...):
    # Setup logging at the very start
    dual_logger = setup_file_logging('results/training.log')
    
    try:
        # All your existing code here
        print_section_header("LOADING ALL DATASETS FROM data/raw")
        # ... rest of training code ...
        
    finally:
        # Cleanup logging at the end
        cleanup_file_logging(dual_logger)


OPTION 2: Use context manager (cleaner):

def main(...):
    with DualLogger('results/training.log'):
        # All your existing code here
        print_section_header("LOADING ALL DATASETS FROM data/raw")
        # ... rest of training code ...
    
    # Log is automatically saved when context exits


OPTION 3: Wrap the entire script (at the bottom):

if __name__ == "__main__":
    # Parse arguments first
    import argparse
    parser = argparse.ArgumentParser(...)
    args = parser.parse_args()
    
    # Setup logging
    if not args.usage and not args.list_datasets:
        with DualLogger('results/training.log'):
            main(
                skip_phase1=args.skip_phase1,
                skip_phase4=args.skip_phase4,
                run_phase5=args.phase5,
                use_bert=args.use_bert,
                incremental=args.incremental and not args.retrain
            )
    else:
        # Don't log usage/list commands
        main(...)
"""


# ========== COMPLETE MODIFIED main() FUNCTION ==========

def main_with_logging(
    skip_phase1: bool = False,
    skip_phase4: bool = False,
    run_phase5: bool = False,
    use_bert: bool = False,
    incremental: bool = False
):
    """
    Main training function with file logging
    This is a drop-in replacement for the existing main() function
    """
    
    # Setup file logging
    dual_logger = setup_file_logging('results/training.log')
    
    try:
        # [PASTE YOUR ENTIRE EXISTING main() FUNCTION CODE HERE]
        # Everything from print_section_header to the final print statements
        
        # Example structure:
        print("\n" + "=" * 80)
        print("HATE SPEECH DETECTION - ENHANCED TRAINING PIPELINE")
        print("=" * 80)
        
        # ... all your existing training code ...
        
        print("\n" + "=" * 80)
        print("[SUCCESS] TRAINING PIPELINE COMPLETE")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    finally:
        # Cleanup logging
        cleanup_file_logging(dual_logger)


# ========== INTEGRATION INSTRUCTIONS ==========

"""
TO ADD LOGGING TO YOUR EXISTING main_train_enhanced.py:

STEP 1: Add this import at the top (after other imports):
--------
from datetime import datetime

STEP 2: Copy the DualLogger class and helper functions to the top of your file
--------
(Copy the DualLogger class, setup_file_logging, and cleanup_file_logging)

STEP 3: Modify the bottom __main__ section:
--------
OLD CODE:
    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser(...)
        args = parser.parse_args()
        
        if args.usage:
            print_usage()
        elif args.list_datasets:
            # list datasets code
        else:
            main(
                skip_phase1=args.skip_phase1,
                ...
            )

NEW CODE:
    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser(...)
        args = parser.parse_args()
        
        if args.usage:
            print_usage()
        elif args.list_datasets:
            # list datasets code
        else:
            # Enable logging for training
            with DualLogger('results/training.log'):
                main(
                    skip_phase1=args.skip_phase1,
                    skip_phase4=args.skip_phase4,
                    run_phase5=args.phase5,
                    use_bert=args.use_bert,
                    incremental=args.incremental and not args.retrain
                )

THAT'S IT! Now all console output will also be saved to results/training.log


FEATURES:
- All print() statements automatically logged
- All logger.info/warning/error logged
- Error tracebacks logged
- No code changes needed in existing functions
- Log file includes timestamps
- Console output unchanged (still shows everything)
- Log saved to: results/training.log


RESULT:
When you run training, you'll see:
  [LOGGING] Output will be logged to: E:/Projects/.../results/training.log
  
And at the end:
  [LOGGING] Training log saved to: E:/Projects/.../results/training.log

The log file will contain EVERYTHING that appeared in the console.
"""