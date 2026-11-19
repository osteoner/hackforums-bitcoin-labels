"""Original file: Cryptocurrency_home/crypto_labellisation/label2.py (dated 2025-07-14).
Only change: DB user/password, the Ollama GPU host and the PGDATA path now come from environment
variables (PGUSER, PGPASSWORD, OLLAMA_HOST_NODE, PGDATA) instead of hard-coded values.

Covers steps 3 + 4 (there is no separate 03_prepare_contexts.py): fetch_posts_for_address() /
prepare_content_for_analysis() build the context, classify_illicit() / identify_illicit_goods() /
categorize_legitimate() run the LLM. Model qwen3:30b via Ollama, temperature 0.3, format=json.
Differences from the paper text: posts are taken newest first (ORDER BY created_on DESC; the paper
says earliest appearances first); screening is a Yes/No illicit prompt followed by a category prompt
(the paper shows one combined prompt); an address is screened illicit if >= 1 model votes Yes."""
from numpy import add
import psycopg2
import pandas as pd
import re
import json
import requests
import time, os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from requests.exceptions import ConnectionError, ConnectTimeout


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Checkpoint configuration
CHECKPOINT_FILE = "classification_checkpoint.json"
logger = logging.getLogger(__name__)

# Custom exceptions for connection errors
class LLMConnectionError(Exception):
    """Raised when LLM connection is lost and cannot be recovered"""
    pass

class CriticalConnectionError(Exception):
    """Raised when critical connections fail and processing should stop"""
    pass

@dataclass
class ClassificationResult:
    address: str
    label: str
    vote_count: int
    llm_votes: List[str]
    legitimate_category: Optional[str] = None
    illicit_category: Optional[str] = None
    supporting_posts_count: int = 0
    thread_name: str = "Unknown Thread"
    confidences: List[float] = None
    average_confidence: float = 0.0


class OllamaLLM:
    """Wrapper for Ollama LLM API calls"""
    
    def __init__(self, model_name: str, base_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3  # Stop after 3 consecutive connection failures
    
    def test_connection(self) -> bool:
        """Test if the Ollama server is reachable"""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=10)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate(self, prompt: str, max_retries: int = 3) -> str:
        """Generate response from LLM with retry logic and connection monitoring"""
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                    },
                    "format": "json"
                }
                
                response = requests.post(self.api_url, json=payload, timeout=60)
                response.raise_for_status()
                
                result = response.json()
                
                # Reset failure counter on success
                self.consecutive_failures = 0
                return result.get('response', '').strip()
                
            except (ConnectionError, ConnectTimeout, requests.exceptions.ConnectionError) as e:
                self.consecutive_failures += 1
                logger.error(f"Connection error for {self.model_name} (attempt {attempt + 1}): {e}")
                
                # Check if this is a critical connection failure
                if "Connection refused" in str(e) or "Failed to establish a new connection" in str(e):
                    logger.critical(f"Critical connection error for {self.model_name}: {e}")
                    
                    # Test if server is completely down
                    if not self.test_connection():
                        logger.critical(f"Ollama server is not reachable. Consecutive failures: {self.consecutive_failures}")
                        
                        # Stop processing if too many consecutive failures
                        if self.consecutive_failures >= self.max_consecutive_failures:
                            error_msg = f"Critical: {self.consecutive_failures} consecutive connection failures for {self.model_name}. Ollama server appears to be down."
                            logger.critical(error_msg)
                            raise LLMConnectionError(error_msg)
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.info(f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"All attempts failed for {self.model_name}")
                    
                    # If this was the last attempt and we still have connection issues
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        raise LLMConnectionError(f"Max consecutive failures reached for {self.model_name}")
                    
                    return "Error"
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {self.model_name}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"All attempts failed for {self.model_name}")
                    return "Error"
        
        return "Error"
class ContentCleaner:
    """Clean and preprocess post content for analysis"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        text = re.sub(r"\\\\", "\n", text)
        quote_pattern = r'\*\*\*QUOTE\*\*\*([\s\S]*?)\*\*\*QUOTE\*\*\*'
        link_pattern = r'\*\*\*LINK\*\*\*([\s\S]*?)\*\*\*LINK\*\*\*'
        img_pattern = r'\*\*\*IMG\*\*\*([\s\S]*?)\*\*\*IMG\*\*\*'
        citing_pattern = r'\*\*\*CITING\*\*\*([\s\S]*?)\*\*\*CITING\*\*\*'
        iframe_pattern = r'\*\*\*IFRAME\*\*\*([\s\S]*?)\*\*\*IFRAME\*\*\*'
        attachment_pattern = r'\*\*\*ATTACHMENT\*\*\*([\s\S]*?)\*\*\*ATTACHMENT\*\*\*'
        text = re.sub(quote_pattern, 'QUOTE', text)
        text = re.sub(link_pattern, 'LINK', text)
        text = re.sub(img_pattern, 'IMG', text)
        text = re.sub(citing_pattern, 'CITING', text)
        text = re.sub(iframe_pattern, 'IFRAME', text)
        text = re.sub(attachment_pattern, 'ATTACHMENT', text)


        
        url_pattern = r'http\S+'

        # Replace URLs with 'HTTPURL'
        text= re.sub(url_pattern, 'HTTPURL',text,flags=re.DOTALL)



        # Remove leading/trailing whitespace
        text = text.strip()
        text = re.sub(r'\n', ' ', text)

        # Replace \\\\ with \n for proper newline handling
        

        # Lowercase for case-insensitive processing
        text = text.lower()

        # Remove punctuation (consider using a custom list for specific needs)
        punctuation = r"[^\w\s'-]"  # Includes common punctuation, apostrophes, hyphens
        text = re.sub(punctuation, ' ', text)
        emoji_pattern = r"[^a-z0-9\s']+"  # Matches non-alphanumeric, non-whitespace, non-quote, non-hyphen characters
        text = re.sub(emoji_pattern, ' ', text)

        # Handle named entities (consider preserving or anonymizing based on privacy)
        # ... (code for named entity recognition and replacement/anonymization)

        # Handle typos and spelling errors (consider using a spell checker or dictionary)
        # ... (code for typo correction)

        # Handle HTML tags and entities (consider removing or converting)
        text = re.sub(r"<[^>]+>", " ", text)  # Remove HTML tags
        text = re.sub(r"&.+?;", " ", text)  # Remove HTML entities
        text = re.sub(r'\S{100,}', '[ENCODED_DATA]', text)
            
        return text.strip()
    
    @staticmethod
    def extract_context(content: str, address: str, context_window: int = 200) -> str:
        """Extract relevant context around the cryptocurrency address"""
        if not content or not address:
            return ""
        
        # Find all occurrences of the address
        pattern = re.escape(address)
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        
        if not matches:
            return content[:500]  # Return first part if address not found
        
        contexts = []
        for match in matches:
            start = max(0, match.start() - context_window)
            end = min(len(content), match.end() + context_window)
            context = content[start:end]
            contexts.append(context)
        
        # Combine unique contexts
        unique_contexts = list(dict.fromkeys(contexts))
        return " ... ".join(unique_contexts)

class CryptoAddressClassifier:
    """Main classifier for cryptocurrency addresses"""
    
    def __init__(self, db_config: Dict[str, str], llm_models: List[str]):
        self.db_config = db_config
        self.llms = [OllamaLLM(model) for model in llm_models]
        self.cleaner = ContentCleaner()

        # Test all LLM connections at startup
        self.verify_llm_connections()

        
        # Classification prompts
        self.illicit_prompt = """
        You are an expert Cyber Threat Intelligence (CTI) analyst. Your task is to analyze the provided hacker forum post containing a cryptocurrency address and determine if they promote trading of illicit goods or services. 

        ### Classification Guidelines:
        Carefully analyze the content of the posts and the context in which the cryptocurrency address is used.
        Consider the following categories for illicit activities:
            - Drug sales or trafficking
            - Weapons trading
            - Stolen goods
            - Fraud schemes or scams
            - Hacking or cybercrime
            - Money laundering services
            - Hacking services
            - Counterfeit goods
            - Human trafficking
            - Other illegal activities
        
        ### Input:
        **Thread Name:**
        {thread_name}

        **Post to Analyze:**
        {content}
        **Address to Analyze:**
        {address}

        Return your classification in this exact format:
        {{
        "response": "Yes" or "No",
        "justification": "Brief explanation of why these reponses were selected based on the evidence",
        "confidence": score from 0 to 1 (1 being very confident, 0 being not confident at all)
        }}
        
        """
        
        self.legitimate_prompt = """
        You are an expert Cyber Threat Intelligence (CTI) analyst. Your task is to analyze the provided hacker forum post containing a cryptocurrency address and categorize its usage context.

        ### Input:
        **Thread Name:**
        {thread_name}

        **Post to Analyze:**
        {content}
        **Address to Analyze:**
        {address}

        Question: What is the usage context of this legitimate address based on the content?

        Choose the most appropriate category:
        - Donation request
        - Knowledge sharing
        - Escrow verification
        - User identification
        - Product advertising
        - Legal services
        - Other(Respond with only a brief description of the goods/services with max 3 words.)

        Respond with only the category name.
        """
        
        self.illicit_goods_prompt = """
            You are an expert Cyber Threat Intelligence (CTI) analyst. Analyze the provided hacker forum post containing a cryptocurrency address and classify the goods/services being traded.

            ## Input Format
            - **Thread Name:** {thread_name}
            - **Post Content:** {content}  
            - **Cryptocurrency Address:** {address}

            ## Classification Categories

            **Illicit Goods/Services:**
            - **Abuse:** Illegal sexual and violent content
            - **Account selling:** Hacked social media accounts and online services
            - **Counterfeit:** Counterfeit money and forged documents
            - **Card dumps:** Stolen credit card data and financial information
            - **Drug:** Illegal narcotics and controlled substances
            - **Fraud:** Scam services, fake documents, identity theft
            - **Investment:** Financial investment schemes and opportunities
            - **Laundering:** Money laundering services and cryptocurrency mixing
            - **Malware:** Malicious software, ransomware, exploits
            - **Membership:** Access to private illegal forums and exclusive communities
            - **Service:** Illegal errand services (hacking, fraud, contract violence)
            - **Weapon:** Unauthorized weapons and firearms

            **Other Categories:**
            - **Others:** Miscellaneous illegal offerings not covered above
            - **Unclear:** Ambiguous or insufficient context to determine offering

            ## Analysis Instructions

            1. Examine the cryptocurrency address context within the post
            2. Identify the core offering being sold, requested, or promoted
            3. Match the offering to the most appropriate category
            4. Focus on the primary goods/services, not secondary discussion

            ## Output Requirements

            Respond with **only one category name** from the list above. No additional text, explanations, or descriptions.
        """
    
    def verify_llm_connections(self):
        """Verify all LLM connections are working at startup"""
        logger.info("Verifying LLM connections...")
        
        failed_models = []
        for llm in self.llms:
            if not llm.test_connection():
                failed_models.append(llm.model_name)
        
        if failed_models:
            error_msg = f"Failed to connect to LLM models: {failed_models}. Please check your Ollama server and SSH tunnel."
            logger.critical(error_msg)
            raise CriticalConnectionError(error_msg)
        
        logger.info("All LLM connections verified successfully")
    
    def handle_llm_connection_error(self, error: Exception, address: str):
        """Handle LLM connection errors and decide whether to continue or stop"""
        logger.critical(f"LLM connection error while processing address {address}: {error}")
        
        # Test if any LLM is still reachable
        working_llms = []
        for llm in self.llms:
            if llm.test_connection():
                working_llms.append(llm.model_name)
        
        if not working_llms:
            error_msg = f"All LLM connections lost. Cannot continue processing. Last address: {address}"
            logger.critical(error_msg)
            raise CriticalConnectionError(error_msg)
        else:
            logger.warning(f"Some LLMs still working: {working_llms}")
            # You could choose to continue with fewer LLMs or stop here
            # For safety, let's stop when any LLM fails
            raise CriticalConnectionError(f"LLM connection lost for address {address}. Stopping to prevent incomplete results.")

    def get_database_connection(self):
        """Establish database connection"""
        return psycopg2.connect(**self.db_config)
    
    def save_checkpoint(self, last_processed_address: str, results_so_far: List[ClassificationResult]):
        """Save current progress to checkpoint file"""
        checkpoint_data = {
            'last_processed_address': last_processed_address,
            'timestamp': time.time(),
            'processed_count': len(results_so_far),
            'results': []  # Store serialized results
        }
        
        # Serialize results for checkpoint
        for result in results_so_far:
            result_data = {
                'address': result.address,
                'label': result.label,
                'vote_count': result.vote_count,
                'llm_votes': result.llm_votes,
                'legitimate_category': result.legitimate_category,
                'illicit_category': result.illicit_category,
                'supporting_posts_count': result.supporting_posts_count,
                'thread_name': getattr(result, 'thread_name', 'Unknown Thread'),
                'confidences': getattr(result, 'confidences', []),
                'average_confidence': getattr(result, 'average_confidence', 0.0)
            }
            checkpoint_data['results'].append(result_data)
        
        try:
            with open(CHECKPOINT_FILE, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            logger.info(f"Checkpoint saved: {len(results_so_far)} addresses processed, last: {last_processed_address}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def load_checkpoint(self) -> Optional[str]:
        """Load checkpoint and return last processed address"""
        try:
            if os.path.exists(CHECKPOINT_FILE):
                with open(CHECKPOINT_FILE, 'r') as f:
                    checkpoint_data = json.load(f)
                
                last_address = checkpoint_data.get('last_processed_address')
                processed_count = checkpoint_data.get('processed_count', 0)
                timestamp = checkpoint_data.get('timestamp', 0)
                
                logger.info(f"Checkpoint loaded: {processed_count} addresses already processed")
                logger.info(f"Last processed address: {last_address}")
                logger.info(f"Checkpoint time: {time.ctime(timestamp)}")
                
                return last_address
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
        
        return None
    
    def load_existing_results(self) -> List[ClassificationResult]:
        """Load existing results from checkpoint"""
        try:
            if os.path.exists(CHECKPOINT_FILE):
                with open(CHECKPOINT_FILE, 'r') as f:
                    checkpoint_data = json.load(f)
                
                results = []
                for result_data in checkpoint_data.get('results', []):
                    result = ClassificationResult(
                        address=result_data['address'],
                        label=result_data['label'],
                        vote_count=result_data['vote_count'],
                        llm_votes=result_data['llm_votes'],
                        legitimate_category=result_data.get('legitimate_category'),
                        illicit_category=result_data.get('illicit_category'),
                        supporting_posts_count=result_data.get('supporting_posts_count', 0),
                    )
                    # Set additional attributes if they exist
                    if 'thread_name' in result_data:
                        result.thread_name = result_data['thread_name']
                    if 'confidences' in result_data:
                        result.confidences = result_data['confidences']
                    if 'average_confidence' in result_data:
                        result.average_confidence = result_data['average_confidence']
                    
                    results.append(result)
                
                logger.info(f"Loaded {len(results)} existing results from checkpoint")
                return results
        except Exception as e:
            logger.error(f"Error loading existing results: {e}")
        
        return []

    def clear_checkpoint(self):
        """Remove checkpoint file"""
        try:
            if os.path.exists(CHECKPOINT_FILE):
                os.remove(CHECKPOINT_FILE)
                logger.info("Checkpoint file cleared")
        except Exception as e:
            logger.error(f"Error clearing checkpoint: {e}")
    
    def fetch_crypto_addresses(self) -> List[Tuple[str, str]]:
        """Fetch all cryptocurrency addresses from database"""
        query = """
        SELECT DISTINCT address, label 
        FROM crypto 
        WHERE address IS NOT NULL 
        AND address != '' AND is_transaction = true
        ORDER BY address 
        """
        
        with self.get_database_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                return cur.fetchall()
    
    def fetch_posts_for_address(self, address: str, limit: int = 20) -> List[Tuple[str, str]]:
        """Fetch posts and their thread names containing the given cryptocurrency address"""
        query = """
        SELECT rp.content, COALESCE(t.name, 'Unknown Thread') AS thread_name
        FROM (
            SELECT p.content, p.thread_id
            FROM posts p
            JOIN crypto_post_mapping cpm ON p.id = cpm.post_id
            JOIN crypto c ON cpm.crypto_id = c.id
            WHERE c.address = %s
            AND p.content IS NOT NULL
            ORDER BY p.created_on DESC
            LIMIT %s
        ) AS rp
        LEFT JOIN threads t ON rp.thread_id = t.id
        GROUP BY rp.content, t.name;

        """
        
        posts = []
        try:
            with self.get_database_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (address, limit))
                    results = cur.fetchall()
                    posts = [(row[0], row[1]) for row in results if row[0]]
        except Exception as e:
            logger.error(f"Error fetching posts for address {address}: {e}")
        
        return posts
    
    def prepare_content_for_analysis(self, posts: List[Tuple[str, str]], address: str) -> Tuple[str, str]:
        """Clean and prepare post content for LLM analysis and aggregate thread names"""
        if not posts:
            return "", "Unknown Thread"
        
        cleaned_posts = []
        thread_names = set()
        
        for post_content, thread_name in posts:
            cleaned = self.cleaner.clean_text(post_content)
            context = self.cleaner.extract_context(cleaned, address)
            if context:
                cleaned_posts.append(context)
            
            # Collect unique thread names
            if thread_name and thread_name != 'Unknown Thread':
                thread_names.add(thread_name)
        
        # Combine posts with separators
        # combined_content = "\n\n---POST SEPARATOR---\n\n".join(cleaned_posts)
        combined_content = " | ".join(cleaned_posts)
        
        # Truncate if too long (keep within reasonable token limits)
        if len(combined_content) > 8000:
            combined_content = combined_content[:8000] + "...[TRUNCATED]"
        
        # Aggregate thread names
        if thread_names:
            aggregated_thread_names = " | ".join(list(thread_names))  # Limit to 5 thread names
            # if len(thread_names) > 5:
            #     aggregated_thread_names += f" (and {len(thread_names) - 5} more threads)"
        else:
            aggregated_thread_names = "Unknown Thread"
        
        return combined_content, aggregated_thread_names
    
    def classify_illicit(self, content: str, address: str, thread_name: str) -> Tuple[List[Dict], int, List[float]]:
        """Classify if address is used for illicit activities using multiple LLMs"""
        votes = []
        confidences = []
        
        if not self.llms:
            logger.error("No LLMs available for classification")
            return [], 0, []

        def get_vote(llm):
            try:
                prompt = self.illicit_prompt.format(
                    content=content, 
                    address=address, 
                    thread_name=thread_name
                )
                # print(prompt)
                response = llm.generate(prompt)
                if response == "Error":
                        logger.error(f"Error response from {llm.model_name}")
                        return {
                            'vote': 'No',
                            'confidence': 0.0,
                            'justification': 'LLM returned error',
                            'model': llm.model_name
                        }
                
                # Try to parse JSON response
                try:
                    # Clean the response to extract JSON
                    response_clean = response.strip()
                    
                    
                    # Look for JSON-like structure
                    if '{' in response_clean and '}' in response_clean:
                        start_idx = response_clean.find('{')
                        end_idx = response_clean.rfind('}') + 1
                        json_str = re.search(r'{\s*"response":.*}', response_clean, re.DOTALL).group(0)
                        # print(json_str)
                        
                        parsed_response = json.loads(json_str)
                        
                        # Extract response and confidence
                        llm_response = parsed_response.get('response', 'No').strip()
                        confidence = float(parsed_response.get('confidence', 0.0))
                        justification = parsed_response.get('justification', 'No justification provided')
                        
                        # Normalize response to Yes/No
                        if llm_response.lower() in ['yes', 'true']:
                            vote = 'Yes'
                        elif llm_response.lower() in ['no', 'false']:
                            vote = 'No'
                        else:
                            logger.warning(f"Unexpected response value from {llm.model_name}: {llm_response}")
                            vote = 'No'
                        
                        return {
                            'vote': vote,
                            'confidence': confidence,
                            'justification': justification,
                            'model': llm.model_name
                        }
                        
                    else:
                        # Fallback to string parsing
                        logger.warning(f"No JSON structure found in response from {llm.model_name}, using fallback parsing")
                        response_lower = response.lower().strip()
                        if 'yes' in response_lower:
                            vote = 'Yes'
                        elif 'no' in response_lower:
                            vote = 'No'
                        else:
                            vote = 'No'
                        
                        return {
                            'vote': vote,
                            'confidence': 0.5,  # Default confidence for non-JSON responses
                            'justification': 'Parsed from non-JSON response',
                            'model': llm.model_name
                        }
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error from {llm.model_name}: {e}")
                    logger.error(f"Raw response: {response}")
                    
                    # Fallback parsing
                    response_lower = response.lower().strip()
                    if 'yes' in response_lower:
                        vote = 'Yes'
                    elif 'no' in response_lower:
                        vote = 'No'
                    else:
                        vote = 'No'
                    
                    return {
                        'vote': vote,
                        'confidence': 0.0,  # Low confidence for parsing errors
                        'justification': 'JSON parsing failed, used fallback',
                        'model': llm.model_name
                    }
                
                except Exception as e:
                    logger.error(f"Unexpected error parsing response from {llm.model_name}: {e}")
                    return {
                        'vote': 'No',
                        'confidence': 0.0,
                        'justification': f'Error: {str(e)}',
                        'model': llm.model_name
                    }
            except LLMConnectionError as e:
                logger.critical(f"Critical LLM connection error from {llm.model_name}: {e}")
                raise  # Re-raise to be caught by the main handler

            except Exception as e:
                logger.error(f"Unexpected error getting vote from {llm.model_name}: {e}")
                return {
                    'vote': 'No',
                    'confidence': 0.0,
                    'justification': f'Error: {str(e)}',
                    'model': llm.model_name
                }
        
        # Get votes from all LLMs with connection error handling
        try:
            with ThreadPoolExecutor(max_workers=min(3, len(self.llms))) as executor:
                future_to_llm = {executor.submit(get_vote, llm): llm for llm in self.llms}
                
                for future in as_completed(future_to_llm):
                    llm = future_to_llm[future]
                    try:
                        result = future.result(timeout=120)
                        if result:
                            votes.append(result)
                            confidences.append(result.get('confidence', 0.0))
                            logger.info(f"{llm.model_name} voted: {result['vote']} (confidence: {result.get('confidence', 0.0):.2f})")
                    except LLMConnectionError as e:
                        # Critical connection error - stop processing
                        self.handle_llm_connection_error(e, address)
                    except Exception as e:
                        logger.error(f"Error getting vote from {llm.model_name}: {e}")
                        error_vote = {
                            'vote': 'No',
                            'confidence': 0.0,
                            'justification': f'Execution error: {str(e)}',
                            'model': llm.model_name
                        }
                        votes.append(error_vote)
                        confidences.append(0.0)
        
        except CriticalConnectionError:
            # Re-raise critical errors
            raise
        except Exception as e:
            logger.error(f"Error in ThreadPoolExecutor: {e}")
            # Try sequential processing as fallback
            for llm in self.llms:
                try:
                    result = get_vote(llm)
                    if result:
                        votes.append(result)
                        confidences.append(result.get('confidence', 0.0))
                except LLMConnectionError as e:
                    self.handle_llm_connection_error(e, address)
                except Exception as llm_error:
                    logger.error(f"Error processing {llm.model_name}: {llm_error}")
                    error_vote = {
                        'vote': 'No',
                        'confidence': 0.0,
                        'justification': f'Sequential processing error: {str(llm_error)}',
                        'model': llm.model_name
                    }
                    votes.append(error_vote)
                    confidences.append(0.0)
        
        yes_votes = sum(1 for vote in votes if vote.get('vote') == 'Yes')
        return votes, yes_votes, confidences
    
    def categorize_legitimate(self, content: str, address: str, thread_name: str) -> str:
        """Categorize legitimate address usage"""
        prompt = self.legitimate_prompt.format(
            content=content, 
            address=address, 
            thread_name=thread_name
        )
        
        # Use first LLM for categorization

        response = (json.loads(self.llms[0].generate(prompt)).get('content') or json.loads(self.llms[0].generate(prompt)).get('categories', ''))
        if response =='':
           response = self.llms[0].generate(prompt)
        # response = self.llms[0].generate(prompt)
        # print (f"LLM response: {response}")
        
        # Map response to valid categories
        # categories = {
        #     'donation': 'Donation request',
        #     'knowledge': 'Knowledge sharing',
        #     'escrow': 'Escrow verification',
        #     'identification': 'User identification',
        #     'advertising': 'Product advertising',
        #     'legal': 'Legal services'
        # }
        
        response_lower = response.lower()
        # for key, category in categories.items():
        #     if key in response_lower:
        #         return category
        
        # return 'Other'
        return response_lower
    
    def identify_illicit_goods(self, content: str, address: str, thread_name: str) -> str:
        """Identify goods/services traded via illicit address"""
        prompt = self.illicit_goods_prompt.format(
            content=content, 
            address=address, 
            thread_name=thread_name
        )
        
        # Use first LLM for identification
        response = (json.loads(self.llms[0].generate(prompt)).get('content') or json.loads(self.llms[0].generate(prompt)).get('categories', ''))
        if response =='':
            response = self.llms[0].generate(prompt)
        
        # Clean and truncate response
        goods = response.strip()
        # if len(goods) > 200:
        #     goods = goods[:200] + "..."
        
        return goods if goods else "Unknown"
    
    def classify_single_address(self, address: str, label: str) -> ClassificationResult:
        """Classify a single cryptocurrency address with connection error handling"""
        logger.info(f"Classifying address: {address}")
        
        try:
            # Your existing classification logic...
            # But wrap the LLM calls with connection error handling
            
            posts_with_threads = self.fetch_posts_for_address(address)
            
            if not posts_with_threads:
                logger.warning(f"No posts found for address: {address}")
                return ClassificationResult(
                    address=address,
                    label="No Posts",
                    vote_count=0,
                    llm_votes=[],
                    supporting_posts_count=0,
                    thread_name="Unknown Thread"
                )
            
            content, thread_name = self.prepare_content_for_analysis(posts_with_threads, address)
            
            # This call can raise CriticalConnectionError
            llm_votes, yes_votes, confidences = self.classify_illicit(content, address, thread_name)
            
            result = ClassificationResult(
                address=address,
                label="",
                vote_count=yes_votes,
                llm_votes=llm_votes,
                supporting_posts_count=len(posts_with_threads),
                thread_name=thread_name,
                confidences=confidences,
                average_confidence=sum(confidences) / len(confidences) if confidences else 0.0
            )
            
            if yes_votes >= 1:
                result.label = "illicit"
                try:
                    result.illicit_category = self.identify_illicit_goods(content, address, thread_name)
                except LLMConnectionError as e:
                    self.handle_llm_connection_error(e, address)
            else:
                result.label = "legitimate"
                try:
                    result.legitimate_category = self.categorize_legitimate(content, address, thread_name)
                except LLMConnectionError as e:
                    self.handle_llm_connection_error(e, address)
            
            logger.info(f"Classified {address} as {result.label}")
            return result
            
        except (CriticalConnectionError, LLMConnectionError):
            # Re-raise connection errors to stop processing
            raise
        except Exception as e:
            logger.error(f"Error classifying address {address}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return ClassificationResult(
                address=address,
                label="error",
                vote_count=0,
                llm_votes=[],
                supporting_posts_count=0,
                thread_name="Unknown Thread"
            )
    
    def classify_all_addresses(self, resume_from_address: Optional[str] = None) -> List[ClassificationResult]:
        """Classify all addresses with connection error handling"""
        addresses = self.fetch_crypto_addresses()
        addresses = [(address, None) for address in addresses]  # Use 'unknown' label for filtering
        # print(f"Number of addresses to classify: {len(addresses)}")
        logger.info(f"Found {len(addresses)} addresses to classify")
        
        # Find starting index if resuming
        start_index = 0
        if resume_from_address:
            for i, (address, label) in enumerate(addresses):
                if address == resume_from_address:
                    start_index = i 
                    break
            logger.info(f"Resuming from index {start_index}")
        
        results = []
        if resume_from_address:
            results = self.load_existing_results()
        
        try:
            for i, (address, label) in enumerate(addresses[start_index:], start_index + 1):
                logger.info(f"Processing {i}/{len(addresses)}: {address}")
                
                try:
                    result = self.classify_single_address(address, label)
                    results.append(result)
                    
                    # Save checkpoint periodically
                    if i % 10 == 0 or i == len(addresses):
                        self.save_checkpoint(address, results)
                        self.save_results_to_csv(results, f"crypto_classification_results_checkpoint_{i}.csv")
                        
                except (CriticalConnectionError, LLMConnectionError) as e:
                    logger.critical(f"Critical connection error at address {address}: {e}")
                    logger.critical("Stopping classification due to connection failure")
                    
                    # Save current progress before stopping
                    self.save_checkpoint(address, results)
                    self.save_results_to_csv(results, f"crypto_classification_results_stopped_{i}.csv")
                    
                    # Print helpful message
                    print(f"\n{'='*50}")
                    print("CLASSIFICATION STOPPED DUE TO CONNECTION ERROR")
                    print(f"{'='*50}")
                    print(f"Reason: {e}")
                    print(f"Last processed address: {address}")
                    print(f"Progress saved in checkpoint and CSV file")
                    print("Please check your Ollama server and SSH tunnel, then restart to resume")
                    print(f"{'='*50}\n")
                    
                    # Exit the program
                    sys.exit(1)
                    
                except Exception as e:
                    logger.error(f"Error classifying address {address}: {e}")
                    error_result = ClassificationResult(
                        address=address,
                        label="error",
                        vote_count=0,
                        llm_votes=[],
                        supporting_posts_count=0,
                    )
                    results.append(error_result)
                    
                    # Save checkpoint even on error
                    self.save_checkpoint(address, results)
        
        except KeyboardInterrupt:
            logger.info("Classification interrupted by user")
            # Save progress before exiting
            if results:
                last_address = results[-1].address if results else "unknown"
                self.save_checkpoint(last_address, results)
            raise
        
        return results
    
    def save_results_to_csv(self, results: List[ClassificationResult], filename: str = "crypto_classification_results.csv"):
        """Save classification results to CSV file"""
        data = []
        for result in results:
            data.append({
                'address': result.address,
                'label': result.label,
                'vote_count': result.vote_count,
                'llm_votes': json.dumps(result.llm_votes) if isinstance(result.llm_votes, list) and result.llm_votes and isinstance(result.llm_votes[0], dict) else json.dumps([v if isinstance(v, str) else str(v) for v in result.llm_votes]),
                'legitimate_category': result.legitimate_category or '',
                'illicit_category': result.illicit_category or '',
                'supporting_posts_count': result.supporting_posts_count,
                'thread_name': getattr(result, 'thread_name', ''),  # Add thread_name to CSV
                'confidences': json.dumps(getattr(result, 'confidences', [])),  # Add individual confidences
                'average_confidence': getattr(result, 'average_confidence', 0.0),  # Add average confidence
                'justifications': json.dumps([vote.get('justification', '') for vote in result.llm_votes] if isinstance(result.llm_votes, list) and result.llm_votes and isinstance(result.llm_votes[0], dict) else []),  # Add justifications
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        logger.info(f"Results saved to {filename}")
        
        # Print summary statistics
        print("\n=== CLASSIFICATION SUMMARY ===")
        print(f"Total addresses processed: {len(results)}")
        print(f"Illicit: {len([r for r in results if r.label == 'illicit'])}")
        print(f"Possibly illicit: {len([r for r in results if r.label == 'possibly_illicit'])}")
        print(f"Legitimate: {len([r for r in results if r.label == 'legitimate'])}")
        print(f"Errors: {len([r for r in results if r.label == 'error'])}")
        
        # Print confidence statistics
        illicit_results = [r for r in results if r.label == 'illicit' and hasattr(r, 'average_confidence')]
        if illicit_results:
            avg_illicit_confidence = sum(r.average_confidence for r in illicit_results) / len(illicit_results)
            print(f"Average confidence for illicit classifications: {avg_illicit_confidence:.2f}")
        
        legitimate_results = [r for r in results if r.label == 'legitimate' and hasattr(r, 'average_confidence')]
        if legitimate_results:
            avg_legitimate_confidence = sum(r.average_confidence for r in legitimate_results) / len(legitimate_results)
            print(f"Average confidence for legitimate classifications: {avg_legitimate_confidence:.2f}")

def main():
    """Main execution function with connection error handling"""
    db_config = {
        'host': 'localhost',
        'database': 'crimebb2',
        'user': os.environ.get('PGUSER', ''),
        'password': os.environ.get('PGPASSWORD', ''),
        'port': 5432
    }
    
    llm_models = [
        'qwen3:30b',
        # 'deepseek-r1:70b'
    ]
    
    try:
        # Setup SSH tunnel and database (your existing code)
        os.system(f"ssh -N -f -L 11434:localhost:11434 {os.environ['OLLAMA_HOST_NODE']}")  # was a hard-coded GPU node
        time.sleep(3)
        os.system(f"module load PostgreSQL/13.4-GCCcore-11.2.0; pg_ctl -D {os.environ['PGDATA']} start")  # was a hard-coded path
        
        # Initialize classifier - this will test connections
        classifier = CryptoAddressClassifier(db_config, llm_models)
        
        # Handle checkpoint logic
        resume_from_address = None
        if os.path.exists(CHECKPOINT_FILE):
            response = input("Existing checkpoint found. Do you want to resume from checkpoint? (y/n): ").lower()
            if response == 'y':
                resume_from_address = classifier.load_checkpoint()
                print(f"Resuming from checkpoint. Last processed: {resume_from_address}")
            else:
                classifier.clear_checkpoint()
                print("Checkpoint cleared. Starting fresh classification.")
        
        # Run classification
        results = classifier.classify_all_addresses(resume_from_address)
        
        # Save final results and cleanup
        classifier.save_results_to_csv(results)
        classifier.clear_checkpoint()
        
        print("Classification completed successfully!")
        
    except CriticalConnectionError as e:
        logger.critical(f"Critical connection error: {e}")
        print(f"\nFATAL ERROR: {e}")
        print("Please check your Ollama server and SSH tunnel configuration.")
        sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("Classification interrupted by user. Progress saved in checkpoint.")
        print("Classification interrupted. You can resume later using the checkpoint.")
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        raise

if __name__ == "__main__":
    main()