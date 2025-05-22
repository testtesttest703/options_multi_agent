import os
import json
import logging
import time
import datetime
import gc  # For garbage collection

logger = logging.getLogger(__name__)

def load_options_data(file_path):
    """
    Load options data from a JSON file with enhanced logging
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dict containing the options data and ticker symbol
    """
    start_time = time.time()
    logger.info(f"Loading options data from {file_path}...")
    
    # Check file size
    file_size = os.path.getsize(file_path)
    logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
    
    # Track memory usage
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        logger.info(f"Memory usage before loading: {mem_before:.2f} MB")
    except ImportError:
        logger.warning("psutil not available, memory usage tracking disabled")
    
    try:
        with open(file_path, 'r') as f:
            # Read first 100 characters to verify file format
            preview = f.read(100)
            f.seek(0)  # Reset to beginning of file
            logger.info(f"Preview of file: {preview}...")
            
            # Load with progress updates for large files
            if file_size > 10*1024*1024:  # If larger than 10MB
                logger.info("Large file detected, loading with progress updates...")
                data = ""
                chunk_size = 10*1024*1024  # 10MB chunks
                bytes_read = 0
                
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    data += chunk
                    bytes_read += len(chunk)
                    logger.info(f"Read {bytes_read / (1024*1024):.2f} MB / {file_size / (1024*1024):.2f} MB ({bytes_read / file_size * 100:.2f}%)")
                
                logger.info("JSON parsing starting...")
                parse_start = time.time()
                options_data = json.loads(data)
                options_data = (next(iter(options_data.values())))
                parse_time = time.time() - parse_start
                logger.info(f"JSON parsing completed in {parse_time:.2f} seconds")
            else:
                # Small file, load normally
                logger.info("Small file, loading directly...")
                options_data = json.load(f)
        
        # Extract ticker from filename
        import re
        logger.info("Extracting ticker from filename...")
        ticker_match = re.search(r'([A-Z]+)_all_trading_days', os.path.basename(file_path))
        ticker = ticker_match.group(1) if ticker_match else "UNKNOWN"
        
        # Log data structure
        logger.info(f"Data loaded successfully. Type: {type(options_data)}")
        if isinstance(options_data, dict):
            num_dates = len(options_data)
            logger.info(f"Number of dates in data: {num_dates}")
            sample_dates = list(options_data.keys())[:5]
            logger.info(f"Sample dates: {sample_dates}")
            
            if sample_dates:
                sample_date = sample_dates[0]
                date_data = options_data[sample_date]
                logger.info(f"Structure for {sample_date}: {type(date_data)}")
                if isinstance(date_data, dict):
                    sub_keys = list(date_data.keys())[:5]
                    logger.info(f"Nested structure: {sub_keys}")
        
        # Track memory after loading
        try:
            if 'process' in locals():
                mem_after = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory usage after loading: {mem_after:.2f} MB (increase: {mem_after - mem_before:.2f} MB)")
        except Exception as e:
            logger.warning(f"Failed to track memory: {e}")
            
        elapsed_time = time.time() - start_time
        logger.info(f"Data loaded successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Returning data for ticker: {ticker}")
        
        # Run garbage collection
        gc.collect()
        logger.info("Garbage collection completed")
        
        return options_data, ticker
    
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON: {e}")
        logger.error(f"Error at position: {e.pos}")
        # Try to show the problematic part
        with open(file_path, 'r') as f:
            f.seek(max(0, e.pos - 50))
            logger.error(f"Context: {f.read(100)}")
        raise
    except Exception as e:
        logger.error(f"Error loading options data: {e}")
        raise
