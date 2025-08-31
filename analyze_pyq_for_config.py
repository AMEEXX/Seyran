import os
import json
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
except ImportError:
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.chat_models import ChatOpenAI

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

try:
    from langchain.chains import LLMChain
except ImportError:
    from langchain_core.runnables import RunnableSequence
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

class PYQConfigAnalyzer:
    """Analyzes Previous Year Questions to generate exam configuration"""
    
    def __init__(self, pyq_path: str, config_output_path: str = "exam_config.json", vector_cache_dir: str = "vector_cache"):
        """Initialize the analyzer"""
        # Store paths
        self.pyq_path = pyq_path
        self.config_output_path = config_output_path
        self.vector_cache_dir = vector_cache_dir
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
        
        # Initialize vector store
        self.vector_store = None
        
        # Track document sources
        self.documents = []
        
        # Default configuration (fallback)
        self.default_config = {
            "sections": [
                {
                    "name": "Section A",
                    "type": "ShortAnswer",
                    "count": 2,
                    "marks_each": 5,
                    "total_marks": 10
                },
                {
                    "name": "Section B",
                    "type": "Problem",
                    "count": 2,
                    "marks_each": 6,
                    "total_marks": 12
                }
            ],
            "topic_weights": {
                "Unit1": 0.25,
                "Unit2": 0.35,
                "Unit3": 0.25,
                "Unit4": 0.15
            },
            "duration_minutes": 180,
            "total_marks": 100,
            "special_instructions": "Answer all questions."
        }
    
    def load_pyqs(self) -> bool:
        """Load and process the PYQs PDF"""
        if not os.path.exists(self.pyq_path):
            print(f"ERROR: {self.pyq_path} not found")
            return False
        
        try:
            print(f"Loading PYQs from {self.pyq_path}...")
            loader = PyPDFLoader(self.pyq_path)
            self.documents = loader.load()
            print(f"Successfully loaded PYQs: {len(self.documents)} pages")
            return True
        except Exception as e:
            print(f"ERROR loading PYQs from {self.pyq_path}: {e}")
            return False
    
    def create_vector_store(self) -> bool:
        """Create vector store from PYQ chunks with caching"""
        try:
            os.makedirs(self.vector_cache_dir, exist_ok=True)
            cache_path = os.path.join(self.vector_cache_dir, "pyq_vector_store")
            
            if os.path.exists(cache_path):
                print("Loading PYQ vector store from cache...")
                embeddings = OpenAIEmbeddings()
                self.vector_store = FAISS.load_local(cache_path, embeddings, allow_dangerous_deserialization=True)
                print("PYQ vector store loaded from cache.")
                return True
                
            print("Creating PYQ vector store...")
            embeddings = OpenAIEmbeddings()
            self.vector_store = FAISS.from_documents(self.documents, embeddings)
            print("PYQ vector store created.")
            
            print(f"Saving PYQ vector store to cache in {self.vector_cache_dir}...")
            self.vector_store.save_local(cache_path)
            print("PYQ vector store saved to cache.")
            
            return True
        except Exception as e:
            print(f"ERROR creating/loading PYQ vector store: {e}")
            self.vector_store = None
            return False
    
    def extract_exam_format(self) -> Dict[str, Any]:
        """Extract exam format information from PYQs"""
        if not self.vector_store:
            print("Vector store not initialized")
            return {}
        
        # Create prompt for format extraction
        format_extraction_template = """
        You are an educational content analyzer.
        
        Analyze the previous year question papers and extract the following information EXACTLY as it appears:
        1. Total duration of the exam (extract the exact line containing time/duration)
        2. Total marks (extract the exact line containing marks)
        3. Section-wise breakdown:
           - Number of sections
           - Name of each section
           - Type of questions in each section (ShortAnswer/Problem/LongAnswer)
           - Number of questions per section
           - Marks per question in each section
        4. Any special instructions or patterns
        5. The exact header line if present
        
        Return the information in this JSON format:
        {{
            "duration_minutes": number,
            "total_marks": number,
            "header": "string",
            "sections": [
                {{
                    "name": "string",
                    "type": "string",
                    "count": number,
                    "marks_each": number,
                    "total_marks": number
                }}
            ],
            "special_instructions": "string"
        }}
        
        Previous Year Questions content:
        {pyq_content}
        
        Return ONLY the JSON object.
        """
        
        # Get representative content from PYQs
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        pyq_docs = retriever.get_relevant_documents("exam format duration marks sections")
        pyq_content = "\n\n".join([doc.page_content for doc in pyq_docs])
        
        # Extract format
        prompt = PromptTemplate(
            input_variables=["pyq_content"],
            template=format_extraction_template
        )
        
        try:
            # Try new method first
            chain = prompt | self.llm
            response = chain.invoke({"pyq_content": pyq_content}).content
        except:
            # Fallback to old method
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.invoke({"pyq_content": pyq_content})["text"]
        
        try:
            # Clean the response first
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            format_data = json.loads(response)
            print("Successfully extracted exam format from PYQs")
            
            # Ensure we have sections
            if not format_data.get("sections"):
                format_data["sections"] = self.get_default_sections(format_data.get("total_marks", 50))
            
            return format_data
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse format from LLM response: {e}")
            print(f"Response was: {response[:200]}...")
            fallback = self.fallback_extract_duration_and_marks(pyq_content)
            if fallback:
                return {
                    "duration_minutes": fallback["duration_minutes"],
                    "total_marks": fallback["total_marks"],
                    "header": fallback["header"],
                    "sections": self.get_default_sections(fallback.get("total_marks", 50)),
                    "special_instructions": "Answer all questions."
                }
            return self.get_complete_fallback_config()
    
    def fallback_extract_duration_and_marks(self, text: str) -> dict:
        """Fallback: Extract duration and marks from text using regex."""
        # Look for lines like 'Time: 2.5 Hours Max. Marks: 50'
        pattern = re.compile(r"(Time:\s*[\d.]+\s*Hours?.*?Max\.\s*Marks:\s*\d+)", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            header = match.group(1).strip()
            # Extract numbers from the header
            num_pattern = re.compile(r"Time:\s*([\d.]+)\s*Hours?.*?Max\.\s*Marks:\s*(\d+)", re.IGNORECASE)
            num_match = num_pattern.search(header)
            if num_match:
                hours = float(num_match.group(1))
                duration_minutes = int(hours * 60)
                total_marks = int(num_match.group(2))
                print(f"[Fallback] Extracted header: {header}")
                return {"duration_minutes": duration_minutes, "total_marks": total_marks, "header": header}
        # Try alternative patterns
        pattern2 = re.compile(r"Duration:\s*([\d.]+)\s*Hours?", re.IGNORECASE)
        pattern3 = re.compile(r"Max\.\s*Marks:\s*(\d+)", re.IGNORECASE)
        hours = None
        marks = None
        m2 = pattern2.search(text)
        m3 = pattern3.search(text)
        if m2:
            hours = float(m2.group(1))
        if m3:
            marks = int(m3.group(1))
        if hours and marks:
            duration_minutes = int(hours * 60)
            header = f"Time: {hours} Hours                 Max. Marks: {marks}"
            print(f"[Fallback] Constructed header: {header}")
            return {"duration_minutes": duration_minutes, "total_marks": marks, "header": header}
        return None
    
    def get_default_sections(self, total_marks: int = 50) -> List[Dict[str, Any]]:
        """Generate default sections based on total marks"""
        if total_marks <= 30:
            return [
                {"name": "Section A", "type": "ShortAnswer", "count": 3, "marks_each": 3, "total_marks": 9},
                {"name": "Section B", "type": "Problem", "count": 3, "marks_each": 7, "total_marks": 21}
            ]
        elif total_marks <= 60:
            return [
                {"name": "Section A", "type": "ShortAnswer", "count": 5, "marks_each": 2, "total_marks": 10},
                {"name": "Section B", "type": "Problem", "count": 4, "marks_each": 5, "total_marks": 20},
                {"name": "Section C", "type": "LongAnswer", "count": 2, "marks_each": 10, "total_marks": 20}
            ]
        else:  # 60+ marks
            return [
                {"name": "Section A", "type": "ShortAnswer", "count": 6, "marks_each": 2, "total_marks": 12},
                {"name": "Section B", "type": "Problem", "count": 5, "marks_each": 6, "total_marks": 30},
                {"name": "Section C", "type": "LongAnswer", "count": 3, "marks_each": 8, "total_marks": 24},
                {"name": "Section D", "type": "Application", "count": 2, "marks_each": 12, "total_marks": 24}
            ]
    
    def get_complete_fallback_config(self) -> Dict[str, Any]:
        """Complete fallback configuration"""
        return {
            "sections": self.get_default_sections(50),
            "topic_weights": {
                "Unit1": 0.25,
                "Unit2": 0.35,
                "Unit3": 0.25,
                "Unit4": 0.15
            },
            "duration_minutes": 150,
            "total_marks": 50,
            "header": "Time: 2.5 Hours                 Max. Marks: 50",
            "special_instructions": "Answer all questions."
        }
    
    def analyze_topic_distribution(self) -> Dict[str, float]:
        """Analyze topic distribution in PYQs"""
        if not self.vector_store:
            print("Vector store not initialized")
            return {}
        
        # Create prompt for topic analysis
        topic_analysis_template = """
        You are an educational content analyzer.
        
        Analyze the previous year questions and determine the weightage/importance of each major topic/unit.
        Return the information as a JSON object where keys are topic names and values are their relative weights (should sum to 1.0).
        
        Example format:
        {{
            "Unit1": 0.25,
            "Unit2": 0.35,
            "Unit3": 0.25,
            "Unit4": 0.15
        }}
        
        Previous Year Questions content:
        {pyq_content}
        
        Return ONLY the JSON object.
        """
        
        # Get representative content from PYQs
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
        pyq_docs = retriever.get_relevant_documents("topic distribution unit weightage")
        pyq_content = "\n\n".join([doc.page_content for doc in pyq_docs])
        
        # Analyze topics
        prompt = PromptTemplate(
            input_variables=["pyq_content"],
            template=topic_analysis_template
        )
        
        try:
            # Try new method first
            chain = prompt | self.llm
            response = chain.invoke({"pyq_content": pyq_content}).content
        except:
            # Fallback to old method
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.invoke({"pyq_content": pyq_content})["text"]
        
        try:
            topic_weights = json.loads(response)
            print("Successfully analyzed topic distribution")
            return topic_weights
        except json.JSONDecodeError:
            print("ERROR: Failed to parse topic distribution from LLM response")
            return {}
    
    def generate_exam_config(self) -> Dict[str, Any]:
        """Generate exam configuration based on PYQ analysis"""
        print("Starting PYQ analysis for exam configuration...")
        
        # Load and process PYQs
        if not self.load_pyqs():
            print("Using default configuration due to PYQ loading error")
            return self.default_config
        
        # Create vector store
        if not self.create_vector_store():
            print("Using default configuration due to vector store creation error")
            return self.default_config
        
        # Extract exam format
        format_data = self.extract_exam_format()
        if not format_data:
            print("Using default configuration due to format extraction error")
            return self.default_config
        
        # Analyze topic distribution
        topic_weights = self.analyze_topic_distribution()
        if not topic_weights:
            print("Using default topic weights due to distribution analysis error")
            topic_weights = self.default_config["topic_weights"]
        
        # Combine into final config
        exam_config = {
            "sections": format_data.get("sections", []),
            "topic_weights": topic_weights,
            "duration_minutes": format_data.get("duration_minutes", 0),
            "total_marks": format_data.get("total_marks", 0),
            "header": format_data.get("header", ""),
            "special_instructions": format_data.get("special_instructions", "")
        }
        
        return exam_config
    
    def save_exam_config(self, config: Dict[str, Any], filename="exam_config.json"):
        """Save the generated exam configuration"""
        try:
            # Remove existing config file if it exists
            if os.path.exists(filename):
                os.remove(filename)
                print(f"Removed existing {filename}")
            
            # Save new config
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"Exam configuration saved to {filename}")
            return True
        except Exception as e:
            print(f"ERROR saving exam configuration: {e}")
            return False

if __name__ == "__main__":
    # Create analyzer
    analyzer = PYQConfigAnalyzer(pyq_path="pyqs.pdf")
    
    # Generate and save config
    config = analyzer.generate_exam_config()
    if config:
        if analyzer.save_exam_config(config):
            print("Successfully updated exam configuration")
        else:
            print("Failed to update exam configuration")
