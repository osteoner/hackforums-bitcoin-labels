"""Original file: Cryptocurrency_home/hacker_graph/btc_extract_valide.py (dated 2025-07-10), code unchanged.

Steps 1 + 2 of the pipeline in ONE script (there is no separate 02_validate_onchain.py):
  - extracts BTC address strings from CrimeBB `posts` and `contracts` with BTC_PATTERN;
  - validates each string on-chain with the Blockstream API (https://blockstream.info/api/address/<addr>,
    valid = HTTP 200, is_transaction = chain_stats.tx_count > 0);
  - writes `crypto` (address, is_transaction) and `crypto_post_mapping` rows into the local CrimeBB
    PostgreSQL database `crimebb2`. Candidates = crypto rows with is_transaction = true.
Differences from the paper text: the paper names Blockchair/Etherscan and query dates 11-12 May 2025;
this code uses Blockstream. Only BTC is handled here (LTC/ETH/DASH/XMR validation code not included).
Requires access to the CrimeBB HackForums database (Cambridge Cybercrime Centre DUA)."""
import pandas as pd
import psycopg2
import re
import requests
from datetime import datetime
from collections import defaultdict
import swifter
import concurrent.futures
from functools import partial
import os
from tqdm import tqdm
import logging
import json
import time
import signal
import sys

# Bitcoin address regex pattern
BTC_PATTERN = r'(?:bc1[0-9a-z]{25,59}|[13][a-km-zA-HJ-NP-Z1-9]{25,34})'

class CryptoAddressExtractor:
    def __init__(self, db_config, max_workers=None, log_file="crypto_extractor.log", checkpoint_file="checkpoint.json"):
        """Initialize the extractor with database configuration."""
        self.db_config = db_config
        self.conn = None
        self.cursor = None
        # Track user addresses and post addresses
        self.user_address_map = defaultdict(lambda: defaultdict(list))
        self.post_address_map = defaultdict(list)
        # Set max workers for thread pools (default to CPU count if None)
        self.max_workers = max_workers if max_workers else os.cpu_count()
        # Tracking and logging
        self.log_file = log_file
        self.checkpoint_file = checkpoint_file
        # Store all extracted addresses
        self.all_extracted_addresses = set()
        # Store validation results
        self.address_validation_map = {}
        # Store post and contract data
        self.posts_with_addresses = []
        self.contracts_with_addresses = []
        
        self.stats = {
            "start_time": None,
            "posts_processed": 0,
            "posts_total": 0,
            "valid_post_addresses": 0,
            "contracts_processed": 0,
            "contracts_total": 0,
            "valid_contract_addresses": 0,
            "user_mappings_processed": 0,
            "user_mappings_total": 0,
            "addresses_validated": 0,
            "valid_addresses": 0,
            "current_stage": "initializing",
            "stage_progress_pct": 0,
            "last_updated": datetime.now().isoformat(),
            "completed": False
        }
        self._setup_logging()
        self._setup_signal_handlers()
        
    def _setup_logging(self):
        """Configure logging to file and console."""
        self.logger = logging.getLogger('CryptoExtractor')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful termination."""
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)
        
    def _handle_signal(self, sig, frame):
        """Handle termination signals."""
        self.logger.warning(f"Received signal {sig}. Saving checkpoint and exiting...")
        self._save_checkpoint()
        sys.exit(0)
        
    def _save_checkpoint(self):
        """Save current progress to checkpoint file."""
        self.stats["last_updated"] = datetime.now().isoformat()
        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
            self.logger.info(f"Checkpoint saved to {self.checkpoint_file}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _load_checkpoint(self):
        """Load progress from checkpoint file if it exists."""
        try:
            if os.path.exists(self.checkpoint_file):
                with open(self.checkpoint_file, 'r') as f:
                    loaded_stats = json.load(f)
                    self.stats.update(loaded_stats)
                self.logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False
            
    def _update_progress(self, stage, progress=None, increment=None, total=None):
        """Update progress statistics and save checkpoint."""
        self.stats["current_stage"] = stage
        self.stats["last_updated"] = datetime.now().isoformat()
        
        if total is not None:
            if stage == "posts":
                self.stats["posts_total"] = total
            elif stage == "contracts":
                self.stats["contracts_total"] = total
            elif stage == "user_mappings":
                self.stats["user_mappings_total"] = total
        
        if increment is not None:
            if stage == "posts":
                self.stats["posts_processed"] += increment
            elif stage == "contracts":
                self.stats["contracts_processed"] += increment
            elif stage == "user_mappings":
                self.stats["user_mappings_processed"] += increment
            elif stage == "valid_post_addresses":
                self.stats["valid_post_addresses"] += increment
            elif stage == "valid_contract_addresses":
                self.stats["valid_contract_addresses"] += increment
            elif stage == "addresses_validated":
                self.stats["addresses_validated"] += increment
            elif stage == "valid_addresses":
                self.stats["valid_addresses"] += increment
        
        if progress is not None:
            self.stats["stage_progress_pct"] = progress
            
        # Save checkpoint every 5 minutes or at key milestones
        if not hasattr(self, "last_checkpoint") or \
           (datetime.now() - datetime.fromisoformat(self.stats.get("last_checkpoint", self.stats["start_time"]))).total_seconds() > 300:
            self._save_checkpoint()
            self.stats["last_checkpoint"] = datetime.now().isoformat()
            
    def print_status(self):
        """Print current status to console."""
        elapsed = datetime.now() - datetime.fromisoformat(self.stats["start_time"])
        elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
        
        status = f"""
=== CryptoAddressExtractor Status ===
Running for: {elapsed_str}
Current stage: {self.stats['current_stage']} ({self.stats['stage_progress_pct']:.1f}%)

Posts: {self.stats['posts_processed']}/{self.stats['posts_total']} processed
Valid post addresses: {self.stats['valid_post_addresses']}

Contracts: {self.stats['contracts_processed']}/{self.stats['contracts_total']} processed
Valid contract addresses: {self.stats['valid_contract_addresses']}

Address validation: {self.stats['addresses_validated']} processed, {self.stats['valid_addresses']} valid

User mappings: {self.stats['user_mappings_processed']}/{self.stats['user_mappings_total']} processed

Last updated: {self.stats['last_updated']}
======================================
"""
        self.logger.info(status)
    
    def connect_to_db(self):
        """Establish database connection."""
        self.conn = psycopg2.connect(**self.db_config)
        self.cursor = self.conn.cursor()
    
    def close_db_connection(self):
        """Close database connection."""
        if self.conn:
            self.conn.commit()
            self.cursor.close()
            self.conn.close()
            print("Database connection closed.")
    
    def fetch_posts_data(self):
        """Fetch posts data from database."""
        self.cursor.execute("SELECT id, creator, content, created_on FROM posts")
        post_data = self.cursor.fetchall()
        colnames = [desc[0] for desc in self.cursor.description]
        return pd.DataFrame(post_data, columns=colnames)
    
    def fetch_contracts_data(self):
        """Fetch contracts data from database."""
        self.cursor.execute("""
            SELECT id, maker, accepter, type, maker_payment_info, accepter_payment_info, started_date
            FROM contracts
            WHERE maker_payment_info IS NOT NULL OR accepter_payment_info IS NOT NULL
        """)
        contract_data = self.cursor.fetchall()
        colnames = [desc[0] for desc in self.cursor.description]
        return pd.DataFrame(contract_data, columns=colnames)
    
    def extract_crypto_addresses(self, text):
        """Extract cryptocurrency addresses from text."""
        if not isinstance(text, str):
            return []
        return re.findall(BTC_PATTERN, text)
    
    def validate_btc_address(self, address):
        """Validate Bitcoin address using Blockstream API."""
        try:
            url = f"https://blockstream.info/api/address/{address}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                n_tx = data.get('chain_stats', {}).get('tx_count', 0)
                return True, n_tx > 0
        except Exception:
            return False, False
        return False, False
    
    def batch_validate_addresses(self, addresses):
        """Validate a batch of addresses in parallel."""
        address_validation_map = {}
        total_addresses = len(addresses)
        
        self.logger.info(f"Beginning validation of {total_addresses} addresses")
        self._update_progress("validation", total=total_addresses)
        
        validated_count = 0
        valid_addresses = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_address = {executor.submit(self.validate_btc_address, address): address for address in addresses}
            
            # Use tqdm for progress tracking
            for future in tqdm(concurrent.futures.as_completed(future_to_address), 
                              total=total_addresses,
                              desc="Validating addresses"):
                address = future_to_address[future]
                validated_count += 1
                
                # Update progress every 10 addresses or 5% of total, whichever is greater
                update_interval = max(10, int(total_addresses * 0.05))
                if validated_count % update_interval == 0:
                    progress_pct = (validated_count / total_addresses) * 100
                    self._update_progress("validation", progress=progress_pct, 
                                         increment=update_interval)
                    self.logger.info(f"Validated {validated_count}/{total_addresses} addresses ({progress_pct:.1f}%)")
                
                try:
                    valid, is_tx = future.result()
                    address_validation_map[address] = (valid, is_tx)
                    if valid:
                        valid_addresses += 1
                        self._update_progress("valid_addresses", increment=1)
                except Exception as e:
                    self.logger.error(f"Error validating address {address}: {e}")
                    address_validation_map[address] = (False, False)
                
                self._update_progress("addresses_validated", increment=1)
        
        self.logger.info(f"Address validation complete. {valid_addresses}/{total_addresses} valid addresses found.")
        return address_validation_map
    
    def get_or_create_crypto(self, address, is_transaction):
        """Insert or get crypto ID from database."""
        self.cursor.execute("SELECT id FROM crypto WHERE address = %s", (address,))
        result = self.cursor.fetchone()
        if result:
            return result[0]
        self.cursor.execute("""
            INSERT INTO crypto (address, crypto_type, label, is_transaction, created_on, updated_on)
            VALUES (%s, %s, %s, %s, NOW(), NOW())
            RETURNING id
        """, (address, 'BTC', None, is_transaction))
        return self.cursor.fetchone()[0]
    
    def extract_from_posts(self):
        """Extract BTC addresses from posts without validation."""
        # Fetch posts data
        df_posts = self.fetch_posts_data()
        posts_count = len(df_posts)
        self.logger.info(f"Extracting addresses from {posts_count} posts...")
        self._update_progress("posts", total=posts_count)
        
        # Extract BTC addresses using swifter for parallel processing
        self.logger.info("Extracting Bitcoin addresses from posts...")
        df_posts['bitcoin_addresses'] = df_posts['content'].swifter.apply(self.extract_crypto_addresses)
        
        # Filter posts with BTC addresses
        df_posts_with_btc = df_posts[df_posts['bitcoin_addresses'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
        posts_with_btc_count = len(df_posts_with_btc)
        self.logger.info(f"Found {posts_with_btc_count} posts with potential BTC addresses.")
        
        # Store these posts for later validation
        self.posts_with_addresses = df_posts_with_btc
        
        # Collect all unique addresses
        for addresses in df_posts_with_btc['bitcoin_addresses']:
            self.all_extracted_addresses.update(addresses)
        
        self.logger.info(f"Extracted {len(self.all_extracted_addresses)} unique BTC addresses from posts")
        self._update_progress("posts", progress=100, increment=posts_count)
        
        return df_posts_with_btc
    
    def extract_from_contracts(self):
        """Extract BTC addresses from contracts without validation."""
        # Fetch contracts data
        df_contracts = self.fetch_contracts_data()
        contracts_count = len(df_contracts)
        self.logger.info(f"Extracting addresses from {contracts_count} contracts...")
        self._update_progress("contracts", total=contracts_count)
        
        # Process maker and accepter payment info in parallel using swifter
        self.logger.info("Extracting Bitcoin addresses from contract payment info...")
        
        self.logger.info("Processing maker payment info...")
        df_contracts['maker_btc_addresses'] = df_contracts['maker_payment_info'].swifter.apply(
            lambda x: self.extract_crypto_addresses(x) if isinstance(x, str) else []
        )
        
        self.logger.info("Processing accepter payment info...")
        df_contracts['accepter_btc_addresses'] = df_contracts['accepter_payment_info'].swifter.apply(
            lambda x: self.extract_crypto_addresses(x) if isinstance(x, str) else []
        )
        
        # Store these contracts for later validation
        self.contracts_with_addresses = df_contracts
        
        # Collect all unique addresses
        for addresses in df_contracts['maker_btc_addresses']:
            self.all_extracted_addresses.update(addresses)
        for addresses in df_contracts['accepter_btc_addresses']:
            self.all_extracted_addresses.update(addresses)
        
        additional_addresses = len(self.all_extracted_addresses) - len(self.all_extracted_addresses)
        self.logger.info(f"Extracted {additional_addresses} additional unique BTC addresses from contracts")
        self._update_progress("contracts", progress=100, increment=contracts_count)
        
        return df_contracts
    
    def validate_all_addresses(self):
        """Validate all extracted addresses in batches."""
        self.logger.info(f"Validating all {len(self.all_extracted_addresses)} unique extracted addresses...")
        self._update_progress("validation", progress=0)
        
        # Validate all addresses in one batch
        self.address_validation_map = self.batch_validate_addresses(self.all_extracted_addresses)
        
        # Count valid addresses
        valid_count = sum(1 for valid, _ in self.address_validation_map.values() if valid)
        self.logger.info(f"Found {valid_count} valid BTC addresses out of {len(self.all_extracted_addresses)}")
        
        return self.address_validation_map
    
    def process_posts_with_validation(self):
        """Process posts with validated addresses."""
        if not hasattr(self, 'posts_with_addresses') or self.posts_with_addresses.empty:
            self.logger.info("No posts with addresses to process.")
            return []
            
        self.logger.info(f"Processing {len(self.posts_with_addresses)} posts with validated addresses...")
        valid_posts = []
        batch_size = 100
        
        for batch_start in range(0, len(self.posts_with_addresses), batch_size):
            batch_end = min(batch_start + batch_size, len(self.posts_with_addresses))
            batch = self.posts_with_addresses.iloc[batch_start:batch_end]
            batch_count = len(batch)
            
            for _, row in tqdm(batch.iterrows(), total=batch_count, 
                              desc=f"Processing posts batch {batch_start//batch_size + 1}"):
                post_id = row['id']
                user_id = row['creator']
                created_on = row['created_on']
                addresses = row['bitcoin_addresses']
                
                valid_addresses_in_post = 0
                for address in addresses:
                    valid, is_tx = self.address_validation_map.get(address, (False, False))
                    if not valid:
                        continue
                        
                    # Store in database
                    crypto_id = self.get_or_create_crypto(address, is_tx)
                    
                    # Debug information
                    self.logger.debug(f"Inserting crypto_post_mapping: crypto_id={crypto_id}, post_id={post_id}")
                    
                    try:
                        self.cursor.execute("""
                            INSERT INTO crypto_post_mapping (crypto_id, post_id, detected_on)
                            VALUES (%s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (crypto_id, post_id, created_on))
                        
                        # Track user and address relationships
                        self.user_address_map[user_id][address].append(created_on)
                        self.post_address_map[address].append(created_on)
                        valid_posts.append((post_id, address))
                        valid_addresses_in_post += 1
                    except Exception as e:
                        self.logger.error(f"Error inserting post mapping: {e}")
                        continue
                
                if valid_addresses_in_post > 0:
                    self._update_progress("valid_post_addresses", increment=valid_addresses_in_post)
            
            # Commit after each batch to ensure data is saved
            try:
                self.conn.commit()
                self.logger.info(f"Committed batch {batch_start//batch_size + 1} of post processing")
            except Exception as e:
                self.logger.error(f"Error committing batch: {e}")
                self.conn.rollback()
            
            # Update progress after each batch
            processed_so_far = min(batch_end, len(self.posts_with_addresses))
            progress_pct = (processed_so_far / len(self.posts_with_addresses)) * 100
            self._update_progress("post_processing", progress=progress_pct)
            
            # Save checkpoint after each batch
            self._save_checkpoint()
            
        self.logger.info(f"Successfully processed {len(valid_posts)} valid BTC addresses in posts.")
        return valid_posts
    
    def process_contracts_with_validation(self):
        """Process contracts with validated addresses."""
        if not hasattr(self, 'contracts_with_addresses') or self.contracts_with_addresses.empty:
            self.logger.info("No contracts with addresses to process.")
            return []
            
        self.logger.info(f"Processing {len(self.contracts_with_addresses)} contracts with validated addresses...")
        valid_contracts = []
        batch_size = 100
        
        for batch_start in range(0, len(self.contracts_with_addresses), batch_size):
            batch_end = min(batch_start + batch_size, len(self.contracts_with_addresses))
            batch = self.contracts_with_addresses.iloc[batch_start:batch_end]
            batch_count = len(batch)
            
            for _, row in tqdm(batch.iterrows(), total=batch_count,
                              desc=f"Processing contracts batch {batch_start//batch_size + 1}"):
                contract_id = row['id']
                maker_id = row['maker']
                accepter_id = row['accepter']
                contract_type = row['type']
                created_on = row['started_date']
                
                valid_addresses_in_contract = 0
                
                # Process maker addresses
                for address in row['maker_btc_addresses']:
                    valid, is_tx = self.address_validation_map.get(address, (False, False))
                    if not valid:
                        continue
                        
                    try:
                        crypto_id = self.get_or_create_crypto(address, is_tx)
                        
                        # Debug information
                        self.logger.debug(f"Inserting crypto_contract_mapping: crypto_id={crypto_id}, contract_id={contract_id}, type={contract_type}")
                        
                        self.cursor.execute("""
                            INSERT INTO crypto_contract_mapping (crypto_id, contract_id, contract_type, detected_on)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (crypto_id, contract_id, contract_type, created_on))
                        
                        self.user_address_map[maker_id][address].append(created_on)
                        self.post_address_map[address].append(created_on)
                        valid_contracts.append((contract_id, address, "maker"))
                        valid_addresses_in_contract += 1
                    except Exception as e:
                        self.logger.error(f"Error processing maker address: {e}")
                        continue
                
                # Process accepter addresses
                for address in row['accepter_btc_addresses']:
                    valid, is_tx = self.address_validation_map.get(address, (False, False))
                    if not valid:
                        continue
                        
                    try:
                        crypto_id = self.get_or_create_crypto(address, is_tx)
                        self.cursor.execute("""
                            INSERT INTO crypto_contract_mapping (crypto_id, contract_id, contract_type, detected_on)
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT DO NOTHING
                        """, (crypto_id, contract_id, contract_type, created_on))
                        
                        self.user_address_map[accepter_id][address].append(created_on)
                        self.post_address_map[address].append(created_on)
                        valid_contracts.append((contract_id, address, "accepter"))
                        valid_addresses_in_contract += 1
                    except Exception as e:
                        self.logger.error(f"Error processing accepter address: {e}")
                        continue
                
                if valid_addresses_in_contract > 0:
                    self._update_progress("valid_contract_addresses", increment=valid_addresses_in_contract)
            
            # Commit after each batch to ensure data is saved
            try:
                self.conn.commit()
                self.logger.info(f"Committed batch {batch_start//batch_size + 1} of contract processing")
            except Exception as e:
                self.logger.error(f"Error committing batch: {e}")
                self.conn.rollback()
                
            # Update progress after each batch
            processed_so_far = min(batch_end, len(self.contracts_with_addresses))
            progress_pct = (processed_so_far / len(self.contracts_with_addresses)) * 100
            self._update_progress("contract_processing", progress=progress_pct)
            
            # Save checkpoint after each batch
            self._save_checkpoint()
        
        self.logger.info(f"Successfully processed {len(valid_contracts)} valid BTC addresses in contracts.")
        return valid_contracts
    
    def _process_user_batch(self, batch):
        """Process a batch of user-address data."""
        results = []
        for user_id, address, timestamps in batch:
            crypto_id = self.get_or_create_crypto(address, True)
            first_shared = min(timestamps)
            last_shared = max(timestamps)
            occurrence = len(timestamps)
            results.append((crypto_id, user_id, occurrence, first_shared, last_shared))
        return results
    
    def update_user_mappings(self):
        """Update crypto_user_mapping table with collected data."""
        user_count = len(self.user_address_map)
        self.logger.info(f"Updating user mappings for {user_count} users...")
        self._update_progress("user_mappings", total=user_count)
        
        # Prepare data for batch processing
        user_address_data = []
        for user_id, addresses in self.user_address_map.items():
            for address, timestamps in addresses.items():
                # Skip invalid addresses
                valid, _ = self.address_validation_map.get(address, (False, False))
                if not valid:
                    continue
                user_address_data.append((user_id, address, timestamps))
        
        total_mappings = len(user_address_data)
        self.logger.info(f"Found {total_mappings} valid user-address mappings to process")
        
        if total_mappings == 0:
            self.logger.warning("No valid user-address mappings to update")
            return
        
        # Split data into batches for efficient DB operations
        batch_size = 500  # Smaller batch size to ensure transactions complete
        batches = [user_address_data[i:i + batch_size] for i in range(0, len(user_address_data), batch_size)]
        
        for i, batch in enumerate(tqdm(batches, desc="Updating user mappings")):
            try:
                results = self._process_user_batch(batch)
                
                # Perform bulk insert/update
                if results:
                    args_str = ','.join(self.cursor.mogrify("(%s,%s,%s,%s,%s)", x).decode('utf-8') for x in results)
                    if args_str:
                        self.cursor.execute("""
                            INSERT INTO crypto_user_mapping 
                            (crypto_id, username, occurrence_count, first_shared_on, last_shared_on)
                            VALUES """ + args_str + """
                            ON CONFLICT (crypto_id, username) DO UPDATE
                            SET occurrence_count = EXCLUDED.occurrence_count,
                                first_shared_on = LEAST(crypto_user_mapping.first_shared_on, EXCLUDED.first_shared_on),
                                last_shared_on = GREATEST(crypto_user_mapping.last_shared_on, EXCLUDED.last_shared_on)
                        """)
                
                # Commit after each batch
                self.conn.commit()
                self.logger.info(f"Committed batch {i+1}/{len(batches)} of user mappings")
                
                # Update progress
                processed_so_far = min((i+1) * batch_size, total_mappings)
                progress_pct = (processed_so_far / total_mappings) * 100
                self._update_progress("user_mappings", progress=progress_pct, increment=len(batch))
                
                # Save checkpoint after each batch
                self._save_checkpoint()
                
            except Exception as e:
                self.logger.error(f"Error updating user mappings batch {i+1}: {e}")
                self.conn.rollback()
                # Continue with the next batch
    
    def run(self):
        """Run the complete extraction and processing pipeline."""
        try:
            # Initialize stats and attempt to load checkpoint
            self.stats["start_time"] = datetime.now().isoformat()
            self.stats["last_checkpoint"] = datetime.now().isoformat()
            checkpoint_loaded = self._load_checkpoint()
            
            self.logger.info(f"Starting CryptoAddressExtractor with {self.max_workers} workers")
            if checkpoint_loaded:
                self.logger.info(f"Resuming from checkpoint: {self.stats['current_stage']} at {self.stats['stage_progress_pct']:.1f}%")
            
            self.connect_to_db()
            
            # Background progress reporting thread
            def report_progress():
                while not self.stats["completed"]:
                    time.sleep(30)  # Report every 30 seconds
            
            # Start progress reporting in background
            progress_thread = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            progress_future = progress_thread.submit(report_progress)
            
            # MODIFIED APPROACH: First extract all addresses without validation
            
            # 1. Extract from posts
            self.stats["current_stage"] = "extracting_posts"
            self.extract_from_posts()
            
            # 2. Extract from contracts
            self.stats["current_stage"] = "extracting_contracts"
            self.extract_from_contracts()
            
            # 3. Validate all extracted addresses at once
            self.stats["current_stage"] = "validating_addresses"
            self.validate_all_addresses()
            
            # Commit after validation to ensure we don't lose the validation results
            self.conn.commit()
            self.logger.info("Committed validation results to database")
            
            # 4. Process posts with validated addresses
            self.stats["current_stage"] = "processing_posts"
            self.process_posts_with_validation()
            
            # Commit after processing posts
            self.conn.commit()
            self.logger.info("Committed post processing results to database")
            
            # 5. Process contracts with validated addresses
            self.stats["current_stage"] = "processing_contracts"
            self.process_contracts_with_validation()
            
            # Commit after processing contracts
            self.conn.commit()
            self.logger.info("Committed contract processing results to database")
            
            # 6. Update user mappings
            self.stats["current_stage"] = "user_mappings"
            self.update_user_mappings()
            
            # Commit changes
            self.conn.commit()
            self.logger.info("Committed user mapping results to database")
            
            # Update completion status
            self.stats["completed"] = True
            self.stats["current_stage"] = "completed"
            self.stats["stage_progress_pct"] = 100
            
            # Final checkpoint
            self._save_checkpoint()
            
            end_time = datetime.now()
            elapsed = (end_time - datetime.fromisoformat(self.stats["start_time"])).total_seconds()
            self.logger.info(f"BTC address extraction and database population completed successfully in {elapsed:.2f} seconds.")
            
            # Wait for progress thread to finish
            progress_thread.shutdown(wait=False)
            
        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
            if self.conn:
                self.conn.rollback()
            # Save checkpoint on error
            self.stats["current_stage"] = "error"
            self._save_checkpoint()
        finally:
            self.close_db_connection()


if __name__ == "__main__":
    # Database configuration
    db_config = {
        "dbname": "crimebb2",
        "host": "localhost",
        "port": "5432"
    }
    

    # extractor = CryptoAddressExtractor(db_config, max_workers=4)
    extractor = CryptoAddressExtractor(db_config)
    extractor.run()
