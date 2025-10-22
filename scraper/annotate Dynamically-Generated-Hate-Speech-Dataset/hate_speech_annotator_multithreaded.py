#!/usr/bin/env python3
"""
Multithreaded Hate Speech Dataset Annotation System
Using OpenAI GPT-4o-mini API with full safety features

Features:
- Multithreaded processing (10-15x faster)
- Thread-safe checkpoint system
- Rate limit protection with token counting
- Live annotation display in terminal
- Graceful shutdown (Ctrl+C saves progress)
- Automatic resume from checkpoints
- Comprehensive error handling
- Config file for API keys
"""

import pandas as pd
import json
import time
import threading
import queue
import signal
import sys
import os
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import deque
from dataclasses import dataclass
import configparser
from pathlib import Path

# OpenAI official SDK
from openai import OpenAI
import tiktoken

# Setup logging - ONLY to file, not console (to avoid flickering with Rich display)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(threadName)-12s] %(levelname)-8s %(message)s',
    handlers=[
        logging.FileHandler('annotator_debug.log')
    ]
)
logger = logging.getLogger(__name__)
logger.info("="*70)
logger.info("Annotator started")
logger.info("="*70)

# Rich library for beautiful terminal display
try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.info("Rich library not available, using simple text output")

# Allow disabling Rich display via environment variable
# Set DISABLE_RICH_DISPLAY=1 to see logs in console instead of dashboard
USE_RICH_DISPLAY = RICH_AVAILABLE and os.getenv('DISABLE_RICH_DISPLAY') != '1'

if not USE_RICH_DISPLAY and RICH_AVAILABLE:
    logger.info("Rich display disabled via DISABLE_RICH_DISPLAY=1")
    # Add console handler for logging when Rich is disabled
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)


@dataclass
class AnnotationResult:
    """Result from a single annotation batch"""
    batch_id: int
    start_idx: int
    end_idx: int
    annotations: List[Dict]
    success: bool
    error: Optional[str] = None
    tokens_used: int = 0


class RateLimiter:
    """Thread-safe rate limiter with token tracking"""
    
    def __init__(self, requests_per_minute: int, tokens_per_minute: int, safety_margin: float = 0.8):
        self.rpm_limit = int(requests_per_minute * safety_margin)
        self.tpm_limit = int(tokens_per_minute * safety_margin)
        
        self.request_times = deque()
        self.token_times = deque()  # (timestamp, token_count)
        self.lock = threading.Lock()
        
        print(f"🛡️  Rate Limiter: {self.rpm_limit} RPM, {self.tpm_limit} TPM")
    
    def wait_if_needed(self, estimated_tokens: int) -> None:
        """Wait if we're approaching rate limits"""
        with self.lock:
            now = time.time()
            
            # Clean old entries (older than 60 seconds)
            cutoff = now - 60
            while self.request_times and self.request_times[0] < cutoff:
                self.request_times.popleft()
            while self.token_times and self.token_times[0][0] < cutoff:
                self.token_times.popleft()
            
            # Check RPM
            current_rpm = len(self.request_times)
            if current_rpm >= self.rpm_limit:
                wait_time = 60 - (now - self.request_times[0]) + 0.1
                if wait_time > 0:
                    logger.debug(f"Rate limit: RPM {current_rpm}/{self.rpm_limit}, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    return self.wait_if_needed(estimated_tokens)
            
            # Check TPM
            current_tokens = sum(tokens for _, tokens in self.token_times)
            if current_tokens + estimated_tokens >= self.tpm_limit:
                wait_time = 60 - (now - self.token_times[0][0]) + 0.1
                if wait_time > 0:
                    logger.debug(f"Rate limit: TPM {current_tokens}/{self.tpm_limit}, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                    return self.wait_if_needed(estimated_tokens)
            
            # Record this request
            self.request_times.append(now)
            self.token_times.append((now, estimated_tokens))


class ThreadSafeCheckpoint:
    """Thread-safe checkpoint manager"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.lock = threading.Lock()
        self.pending_updates = {}
        self.last_save = time.time()
        self.save_interval = 30  # Save every 30 seconds minimum
    
    def update(self, row_idx: int, data: Dict) -> None:
        """Queue an update for a specific row"""
        with self.lock:
            self.pending_updates[row_idx] = data
    
    def save(self, df: pd.DataFrame, force: bool = False) -> bool:
        """Save checkpoint if needed"""
        with self.lock:
            now = time.time()
            if not force and (now - self.last_save) < self.save_interval:
                return False
            
            if not self.pending_updates:
                return False
            
            # Apply pending updates
            for row_idx, data in self.pending_updates.items():
                for col, value in data.items():
                    df.at[row_idx, col] = value
            
            # Save to disk
            df.to_csv(self.filepath, index=False)
            self.pending_updates.clear()
            self.last_save = now
            return True


class LiveDisplay:
    """Live terminal display for annotation progress"""
    
    def __init__(self, total_rows: int, num_threads: int):
        self.total_rows = total_rows
        self.num_threads = num_threads
        self.console = Console() if USE_RICH_DISPLAY else None
        
        self.lock = threading.Lock()
        self.processed = 0
        self.hate_count = 0
        self.offensive_count = 0
        self.neither_count = 0
        self.api_calls = 0
        self.errors = 0
        self.tokens_used = 0
        
        self.recent_annotations = deque(maxlen=10)
        self.start_time = time.time()
        self.thread_status = {i: "Idle" for i in range(num_threads)}
    
    def update_stats(self, hate: int = 0, offensive: int = 0, neither: int = 0, 
                     api_call: bool = False, error: bool = False, tokens: int = 0):
        """Update statistics"""
        with self.lock:
            self.hate_count += hate
            self.offensive_count += offensive
            self.neither_count += neither
            self.processed += (hate + offensive + neither)
            if api_call:
                self.api_calls += 1
            if error:
                self.errors += 1
            self.tokens_used += tokens
    
    def add_annotation(self, text: str, category: int, row_idx: int):
        """Add recent annotation to display"""
        with self.lock:
            label = ["HATE", "OFFENSIVE", "NEITHER"][category]
            self.recent_annotations.append((row_idx, label, text[:50]))
    
    def update_thread_status(self, thread_id: int, status: str):
        """Update thread status"""
        with self.lock:
            self.thread_status[thread_id] = status
    
    def generate_display(self) -> Table:
        """Generate rich table for display"""
        if not RICH_AVAILABLE:
            return None
        
        with self.lock:
            elapsed = time.time() - self.start_time
            rate = self.processed / elapsed if elapsed > 0 else 0
            eta = (self.total_rows - self.processed) / rate if rate > 0 else 0
            
            # Main stats table
            table = Table(show_header=True, header_style="bold magenta", expand=True)
            table.add_column("Metric", style="cyan", width=25)
            table.add_column("Value", style="green", width=20)
            
            progress_pct = (self.processed / self.total_rows * 100) if self.total_rows > 0 else 0
            table.add_row("Progress", f"{self.processed:,} / {self.total_rows:,} ({progress_pct:.1f}%)")
            table.add_row("Rate", f"{rate:.1f} rows/sec")
            table.add_row("ETA", f"{eta/60:.1f} minutes")
            table.add_row("", "")
            table.add_row("Hate Speech", f"{self.hate_count:,} ({self.hate_count/max(1,self.processed)*100:.1f}%)")
            table.add_row("Offensive", f"{self.offensive_count:,} ({self.offensive_count/max(1,self.processed)*100:.1f}%)")
            table.add_row("Neither", f"{self.neither_count:,} ({self.neither_count/max(1,self.processed)*100:.1f}%)")
            table.add_row("", "")
            table.add_row("API Calls", f"{self.api_calls:,}")
            table.add_row("Tokens Used", f"{self.tokens_used:,}")
            table.add_row("Errors", f"{self.errors:,}")
            
            return table
    
    def generate_thread_table(self) -> Table:
        """Generate thread status table"""
        if not RICH_AVAILABLE:
            return None
        
        with self.lock:
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("Thread", width=8)
            table.add_column("Status", width=30)
            
            for tid, status in self.thread_status.items():
                table.add_row(f"#{tid}", status)
            
            return table
    
    def generate_recent_table(self) -> Table:
        """Generate recent annotations table"""
        if not RICH_AVAILABLE:
            return None
        
        with self.lock:
            table = Table(show_header=True, header_style="bold yellow")
            table.add_column("Row", width=8)
            table.add_column("Label", width=12)
            table.add_column("Text Preview", width=50)
            
            for row_idx, label, text in list(self.recent_annotations)[::-1]:
                style = "red" if label == "HATE" else "yellow" if label == "OFFENSIVE" else "green"
                table.add_row(str(row_idx), label, text, style=style)
            
            return table
    
    def print_simple(self):
        """Simple text output if rich not available"""
        with self.lock:
            elapsed = time.time() - self.start_time
            rate = self.processed / elapsed if elapsed > 0 else 0
            
            print(f"\r[{self.processed}/{self.total_rows}] "
                  f"H:{self.hate_count} O:{self.offensive_count} N:{self.neither_count} "
                  f"| API:{self.api_calls} Err:{self.errors} | {rate:.1f} rows/s", 
                  end="", flush=True)


class MultithreadedAnnotator:
    """Main multithreaded annotation system"""
    
    def __init__(self, config_path: str = "config.ini"):
        # Load configuration
        self.config = self.load_config(config_path)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.config['api_key'])
        self.model = self.config['model']
        
        # Initialize tokenizer for accurate token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except:
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Thread management
        self.num_threads = self.config['num_threads']
        self.batch_size = self.config['batch_size']
        self.work_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.threads = []
        self.shutdown_event = threading.Event()
        
        # Rate limiter
        self.rate_limiter = RateLimiter(
            self.config['rpm_limit'],
            self.config['tpm_limit'],
            self.config['safety_margin']
        )
        
        # Checkpoint manager
        self.checkpoint = None
        
        # Display
        self.display = None
        
        # Setup graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from INI file"""
        config = configparser.ConfigParser()
        
        if not os.path.exists(config_path):
            print(f"❌ Config file '{config_path}' not found!")
            print(f"📄 Please create config.ini from config_template.ini")
            sys.exit(1)
        
        config.read(config_path)
        
        # Validate API key
        api_key = config.get('API', 'openai_api_key', fallback=None)
        if not api_key or api_key == 'YOUR_OPENAI_API_KEY_HERE':
            print("❌ Please set your OpenAI API key in config.ini")
            print("🔑 Get your key from: https://platform.openai.com/api-keys")
            sys.exit(1)
        
        return {
            'api_key': api_key,
            'num_threads': config.getint('PROCESSING', 'num_threads', fallback=8),
            'batch_size': config.getint('PROCESSING', 'batch_size', fallback=15),
            'checkpoint_interval': config.getint('PROCESSING', 'checkpoint_interval', fallback=1000),
            'model': config.get('PROCESSING', 'model', fallback='gpt-4o-mini'),
            'temperature': config.getfloat('PROCESSING', 'temperature', fallback=0.1),
            'rpm_limit': config.getint('RATE_LIMITS', 'requests_per_minute', fallback=500),
            'tpm_limit': config.getint('RATE_LIMITS', 'tokens_per_minute', fallback=200000),
            'safety_margin': config.getfloat('RATE_LIMITS', 'safety_margin', fallback=0.8),
            'input_csv': config.get('FILES', 'input_csv', fallback='input_dataset.csv'),
            'output_csv': config.get('FILES', 'output_csv', fallback='annotated_dataset.csv')
        }
    
    def signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully"""
        print("\n\n🛑 Shutdown signal received (Ctrl+C)")
        print("💾 Saving all progress and shutting down gracefully...")
        self.shutdown_event.set()
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        try:
            return len(self.tokenizer.encode(text))
        except:
            return len(text) // 4  # Rough estimate
    
    def create_annotation_prompt(self, texts_with_context: List[Dict]) -> str:
        """Create batch annotation prompt with clear examples"""
        prompt = """You are an expert hate speech classifier for academic research. Analyze each text and classify it into EXACTLY ONE category.

**CATEGORY DEFINITIONS & EXAMPLES:**

**Category 0: HATE SPEECH**
Definition: Content expressing hate, violence, dehumanization, or calls for harm against people based on protected characteristics (race, ethnicity, religion, nationality, gender, sexual orientation, disability, caste).

Examples:
- "All [religious group] are terrorists and should be eliminated"
- "Women belong in the kitchen, they're too stupid for anything else"
- "[Ethnic group] are subhuman and need to be cleansed"
- "Gay people are disgusting abominations that deserve to die"
- "Dalits are worthless lowlives who don't deserve respect"

**Category 1: OFFENSIVE LANGUAGE**
Definition: Profanity, insults, vulgar language, or aggressive tone WITHOUT targeting protected groups. May be rude or inappropriate but not hate speech.

Examples:
- "You're such an idiot, shut the f*** up"
- "This movie is absolute garbage, what a waste of time"
- "Go to hell you moron"
- "That's the dumbest thing I've ever heard"
- "F*** this traffic, I'm so pissed off"

**Category 2: NEITHER**
Definition: Neutral, informative, positive, or general negative content without hate or profanity. Includes news, facts, opinions, casual conversation.

Examples:
- "The weather is nice today"
- "I disagree with this policy decision"
- "Research shows that climate change is real"
- "I don't like pineapple on pizza"
- "The stock market dropped 2% today"

**CLASSIFICATION RULES:**
1. If attacking/dehumanizing protected groups → Category 0 (Hate Speech)
2. If vulgar/insulting but not targeting groups → Category 1 (Offensive)
3. If neither of above → Category 2 (Neither)
4. When in doubt between 0 and 1, check: Does it target a protected group? If YES → 0, if NO → 1
5. When in doubt between 1 and 2, check: Is there profanity or insults? If YES → 1, if NO → 2

**TEXTS TO CLASSIFY:**

"""
        
        for i, item in enumerate(texts_with_context):
            prompt += f"\n--- Text {i+1} ---\n"
            prompt += f"ID: {item['id']}\n"
            prompt += f"Text: {item['text']}\n"
            if item.get('context'):
                prompt += f"Context: {item['context']}\n"
        
        prompt += """
**RESPONSE FORMAT:**
Return ONLY a valid JSON array. No explanations, no markdown, just the JSON array.

[
  {"text_id": 1, "category": 0, "confidence": 0.95},
  {"text_id": 2, "category": 1, "confidence": 0.87},
  {"text_id": 3, "category": 2, "confidence": 0.92}
]

Remember:
- Category 0 = Hate Speech (targets protected groups)
- Category 1 = Offensive Language (vulgar but not targeting groups)
- Category 2 = Neither (neutral/informative)

Return ONLY the JSON array above."""
        
        return prompt
    
    def annotate_batch(self, texts_with_context: List[Dict], thread_id: int) -> AnnotationResult:
        """Annotate a batch of texts"""
        batch_id = texts_with_context[0]['id']
        start_idx = texts_with_context[0]['row_idx']
        end_idx = texts_with_context[-1]['row_idx']
        
        logger.debug(f"Thread {thread_id} annotating batch {batch_id} (rows {start_idx}-{end_idx})")
        
        try:
            # Create prompt
            prompt = self.create_annotation_prompt(texts_with_context)
            
            # Estimate tokens
            estimated_tokens = self.count_tokens(prompt) + 500  # +500 for response
            logger.debug(f"Thread {thread_id} batch {batch_id}: estimated {estimated_tokens} tokens")
            
            # Wait for rate limit
            logger.debug(f"Thread {thread_id} batch {batch_id}: checking rate limits")
            self.rate_limiter.wait_if_needed(estimated_tokens)
            logger.debug(f"Thread {thread_id} batch {batch_id}: rate limit OK, calling API")
            
            # Update display
            if self.display:
                self.display.update_thread_status(thread_id, f"API call (batch {batch_id})")
            
            # Call OpenAI API with timeout
            logger.debug(f"Thread {thread_id} batch {batch_id}: starting API call")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a hate speech classification expert. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config['temperature'],
                max_tokens=2000,
                timeout=60  # 60 second timeout
            )
            logger.debug(f"Thread {thread_id} batch {batch_id}: API call completed")
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            logger.debug(f"Thread {thread_id} batch {batch_id}: got response, parsing JSON")
            
            # Extract JSON
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            annotations = json.loads(response_text)
            logger.debug(f"Thread {thread_id} batch {batch_id}: successfully parsed {len(annotations)} annotations")
            
            # Get actual tokens used
            tokens_used = response.usage.total_tokens if hasattr(response, 'usage') else estimated_tokens
            
            return AnnotationResult(
                batch_id=batch_id,
                start_idx=start_idx,
                end_idx=end_idx,
                annotations=annotations,
                success=True,
                tokens_used=tokens_used
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Thread {thread_id} batch {batch_id}: JSON parse error: {e}")
            logger.error(f"Response was: {response_text[:200]}")
            # Return a special failure result for JSON parse errors
            return AnnotationResult(
                batch_id=batch_id,
                start_idx=start_idx,
                end_idx=end_idx, 
                annotations=[{"text_id": i+1, "category": 2, "confidence": 0.0} for i in range(len(texts_with_context))],
                success=False,
                error=f"JSON parse error: {str(e)}",
                tokens_used=estimated_tokens # Use estimated tokens since we got the response but couldn't parse it
            )
        
        except Exception as e:
            logger.error(f"Thread {thread_id} batch {batch_id}: API error: {e}", exc_info=True)
            return AnnotationResult(
                batch_id=batch_id,
                start_idx=start_idx,
                end_idx=end_idx,
                annotations=[{"text_id": i+1, "category": 2, "confidence": 0.0} for i in range(len(texts_with_context))],
                success=False,
                error=str(e)
            )
    
    def worker_thread(self, thread_id: int):
        """Worker thread function"""
        logger.info(f"Thread {thread_id} started")
        
        while not self.shutdown_event.is_set():
            try:
                # Get work from queue (with timeout to check shutdown)
                try:
                    work_item = self.work_queue.get(timeout=1.0)
                except queue.Empty:
                    if self.display:
                        self.display.update_thread_status(thread_id, "Waiting...")
                    continue
                
                if work_item is None:  # Poison pill
                    logger.info(f"Thread {thread_id} received poison pill, shutting down")
                    break
                
                batch_id = work_item['batch_id']
                logger.debug(f"Thread {thread_id} processing batch {batch_id} ({len(work_item['texts'])} texts)")
                
                # Process batch
                if self.display:
                    self.display.update_thread_status(thread_id, f"Processing batch {batch_id}")
                
                result = self.annotate_batch(work_item['texts'], thread_id)
                
                logger.debug(f"Thread {thread_id} completed batch {batch_id}, success={result.success}")
                
                # Put result in queue
                self.result_queue.put(result)
                logger.debug(f"Thread {thread_id} put result for batch {batch_id} in result queue")
                
                # Mark task as done
                self.work_queue.task_done()
                
                if self.display:
                    self.display.update_thread_status(thread_id, "Idle")
                
            except Exception as e:
                logger.error(f"Thread {thread_id} error: {e}", exc_info=True)
                continue
        
        logger.info(f"Thread {thread_id} stopped")
        if self.display:
            self.display.update_thread_status(thread_id, "Stopped")
    
    def prepare_context_string(self, row: pd.Series, exclude_cols: List[str]) -> str:
        """Prepare context string from row"""
        context_parts = []
        for col in row.index:
            if col not in exclude_cols and pd.notna(row[col]) and row[col] != '':
                context_parts.append(f"{col}={row[col]}")
        return " | ".join(context_parts) if context_parts else ""
    
    def process_dataset(self, input_csv: str = None, output_csv: str = None, sample_size: int = None):
        """Main processing function"""
        # Use config defaults if not provided
        input_csv = input_csv or self.config['input_csv']
        output_csv = output_csv or self.config['output_csv']
        
        print("=" * 70)
        print("🚀 MULTITHREADED HATE SPEECH ANNOTATION SYSTEM")
        print("=" * 70)
        print(f"Model: {self.model}")
        print(f"Threads: {self.num_threads}")
        print(f"Batch Size: {self.batch_size}")
        print(f"Rate Limits: {self.config['rpm_limit']} RPM, {self.config['tpm_limit']} TPM")
        print("=" * 70)
        
        # Load dataset
        print(f"\n📂 Loading dataset: {input_csv}")
        df = pd.read_csv(input_csv)
        original_size = len(df)
        
        if sample_size:
            df = df.head(sample_size)
            print(f"🧪 Sample mode: {len(df)} rows (of {original_size})")
        else:
            print(f"📊 Total rows: {len(df):,}")
        
        # Setup output columns
        if 'count' not in df.columns:
            df.insert(0, 'count', range(len(df)))
        for col in ['hate_speech', 'offensive_language', 'neither', 'class']:
            if col not in df.columns:
                df[col] = 0 if col != 'class' else -1
        if 'tweet' not in df.columns:
            text_col = 'text' if 'text' in df.columns else df.columns[1]
            df['tweet'] = df[text_col]
        
        text_col = 'text' if 'text' in df.columns else 'tweet'
        
        # Check for existing checkpoint
        checkpoint_path = output_csv.replace('.csv', '_checkpoint.csv')
        start_idx = 0
        
        if os.path.exists(checkpoint_path):
            print(f"\n♻️  Found checkpoint: {checkpoint_path}")
            checkpoint_df = pd.read_csv(checkpoint_path)
            completed_rows = (checkpoint_df['class'] != -1).sum()
            if completed_rows > 0:
                print(f"✓ Resuming from row {completed_rows}")
                start_idx = completed_rows
                df.iloc[:start_idx] = checkpoint_df.iloc[:start_idx]
        
        # Initialize checkpoint manager
        self.checkpoint = ThreadSafeCheckpoint(checkpoint_path)
        
        # Initialize display
        self.display = LiveDisplay(len(df) - start_idx, self.num_threads)
        
        # Start worker threads
        print(f"\n🔄 Starting {self.num_threads} worker threads...")
        for i in range(self.num_threads):
            t = threading.Thread(target=self.worker_thread, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)
        
        # Create work items
        print(f"📦 Creating work batches...")
        batch_id = 0
        for i in range(start_idx, len(df), self.batch_size):
            if self.shutdown_event.is_set():
                break
                
            batch_end = min(i + self.batch_size, len(df))
            batch = df.iloc[i:batch_end]
            
            texts_with_context = []
            for idx, row in batch.iterrows():
                context = self.prepare_context_string(
                    row,
                    exclude_cols=['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet', text_col]
                )
                texts_with_context.append({
                    'id': batch_id + 1,
                    'row_idx': idx,
                    'text': str(row[text_col]),
                    'context': context
                })
            
            self.work_queue.put({
                'batch_id': batch_id,
                'texts': texts_with_context
            })
            batch_id += 1
        
        total_batches = batch_id
        print(f"✓ Created {total_batches} batches\n")
        
        # Process results with live display
        processed_batches = 0
        last_log_time = time.time()
        
        logger.info(f"Starting result processing loop, expecting {total_batches} batches")
        
        if USE_RICH_DISPLAY:
            with Live(self.display.generate_display(), refresh_per_second=2, console=self.display.console) as live:
                while processed_batches < total_batches and not self.shutdown_event.is_set():
                    # Log every 10 seconds
                    if time.time() - last_log_time > 10:
                        logger.info(f"Progress: {processed_batches}/{total_batches} batches, "
                                  f"work_queue: {self.work_queue.qsize()}, "
                                  f"result_queue: {self.result_queue.qsize()}")
                        last_log_time = time.time()
                    
                    try:
                        logger.debug(f"Waiting for result (processed {processed_batches}/{total_batches})")
                        result = self.result_queue.get(timeout=1.0)
                        logger.debug(f"Got result for batch {result.batch_id}")
                    except queue.Empty:
                        # Update display even when waiting
                        layout = Layout()
                        layout.split_column(
                            Layout(Panel(self.display.generate_display(), title="📊 Statistics"), size=15),
                            Layout(Panel(self.display.generate_thread_table(), title="🧵 Threads"), size=self.num_threads + 3),
                            Layout(Panel(self.display.generate_recent_table(), title="📝 Recent Annotations"), size=13)
                        )
                        live.update(layout)
                        continue
                    
                    logger.debug(f"Processing result for batch {result.batch_id} (rows {result.start_idx}-{result.end_idx})")
                    
                    # Apply results to dataframe
                    for j, annotation in enumerate(result.annotations):
                        row_idx = result.start_idx + j
                        if row_idx >= len(df):
                            break
                        
                        category = annotation.get('category', 2)
                        
                        # Prepare update data
                        update_data = {
                            'hate_speech': 1 if category == 0 else 0,
                            'offensive_language': 1 if category == 1 else 0,
                            'neither': 1 if category == 2 else 0,
                            'class': category
                        }
                        
                        # Queue checkpoint update
                        self.checkpoint.update(row_idx, update_data)
                        
                        # Apply to dataframe immediately
                        for col, val in update_data.items():
                            df.at[row_idx, col] = val
                        
                        # Update display stats
                        if category == 0:
                            self.display.update_stats(hate=1)
                        elif category == 1:
                            self.display.update_stats(offensive=1)
                        else:
                            self.display.update_stats(neither=1)
                        
                        # Add to recent annotations
                        self.display.add_annotation(str(df.at[row_idx, text_col]), category, row_idx)
                    
                    # Update API call stats
                    self.display.update_stats(api_call=True, error=not result.success, tokens=result.tokens_used)
                    
                    # Save checkpoint periodically
                    self.checkpoint.save(df)
                    
                    processed_batches += 1
                    logger.debug(f"Completed processing batch {result.batch_id}, total processed: {processed_batches}/{total_batches}")
                    
                    # Update live display
                    layout = Layout()
                    layout.split_column(
                        Layout(Panel(self.display.generate_display(), title="📊 Statistics"), size=15),
                        Layout(Panel(self.display.generate_thread_table(), title="🧵 Threads"), size=self.num_threads + 3),
                        Layout(Panel(self.display.generate_recent_table(), title="📝 Recent Annotations"), size=13)
                    )
                    live.update(layout)
        
        else:
            # Fallback simple display
            while processed_batches < total_batches and not self.shutdown_event.is_set():
                try:
                    result = self.result_queue.get(timeout=1.0)
                except queue.Empty:
                    self.display.print_simple()
                    continue
                
                # Apply results (same as above)
                for j, annotation in enumerate(result.annotations):
                    row_idx = result.start_idx + j
                    if row_idx >= len(df):
                        break
                    
                    category = annotation.get('category', 2)
                    update_data = {
                        'hate_speech': 1 if category == 0 else 0,
                        'offensive_language': 1 if category == 1 else 0,
                        'neither': 1 if category == 2 else 0,
                        'class': category
                    }
                    self.checkpoint.update(row_idx, update_data)
                    for col, val in update_data.items():
                        df.at[row_idx, col] = val
                    
                    if category == 0:
                        self.display.update_stats(hate=1)
                    elif category == 1:
                        self.display.update_stats(offensive=1)
                    else:
                        self.display.update_stats(neither=1)
                
                self.display.update_stats(api_call=True, error=not result.success, tokens=result.tokens_used)
                self.checkpoint.save(df)
                processed_batches += 1
                self.display.print_simple()
        
        # Shutdown threads
        print("\n\n🛑 Stopping worker threads...")
        for _ in range(self.num_threads):
            self.work_queue.put(None)  # Poison pills
        
        for t in self.threads:
            t.join(timeout=5.0)
        
        # Final checkpoint save
        print("💾 Saving final checkpoint...")
        self.checkpoint.save(df, force=True)
        
        # Save final output
        print(f"💾 Saving final output: {output_csv}")
        output_cols = ['count', 'hate_speech', 'offensive_language', 'neither', 'class', 'tweet']
        df[output_cols].to_csv(output_csv, index=False)
        
        # Print final summary
        self.print_final_summary(df)
        
        # Cleanup checkpoint if completed
        if not self.shutdown_event.is_set() and processed_batches >= total_batches:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
                print("✓ Checkpoint file removed (processing complete)")
        
        print("\n✅ Processing complete!")
        return df
    
    def print_final_summary(self, df: pd.DataFrame):
        """Print final summary"""
        print("\n" + "=" * 70)
        print("🎉 FINAL SUMMARY")
        print("=" * 70)
        
        total = (df['class'] != -1).sum()
        hate = (df['class'] == 0).sum()
        offensive = (df['class'] == 1).sum()
        neither = (df['class'] == 2).sum()
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Processed:            {total:,} rows")
        print(f"  Hate Speech:          {hate:,} ({hate/total*100:.2f}%)")
        print(f"  Offensive Language:   {offensive:,} ({offensive/total*100:.2f}%)")
        print(f"  Neither:              {neither:,} ({neither/total*100:.2f}%)")
        
        if self.display:
            print(f"\n🔧 Processing Statistics:")
            print(f"  API Calls:            {self.display.api_calls:,}")
            print(f"  Tokens Used:          {self.display.tokens_used:,}")
            print(f"  Errors:               {self.display.errors:,}")
            
            elapsed = time.time() - self.display.start_time
            print(f"  Time Elapsed:         {elapsed/60:.1f} minutes")
            print(f"  Processing Rate:      {total/elapsed:.1f} rows/sec")
        
        print("=" * 70)


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multithreaded Hate Speech Annotator')
    parser.add_argument('--config', default='config.ini', help='Config file path')
    parser.add_argument('--input', help='Input CSV file (overrides config)')
    parser.add_argument('--output', help='Output CSV file (overrides config)')
    parser.add_argument('--sample', type=int, help='Sample size for testing')
    
    args = parser.parse_args()
    
    try:
        annotator = MultithreadedAnnotator(args.config)
        annotator.process_dataset(
            input_csv=args.input,
            output_csv=args.output,
            sample_size=args.sample
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("💾 All progress has been saved to checkpoint")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()