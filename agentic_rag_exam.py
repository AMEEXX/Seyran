import os
import json
import random
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Import fallback handler for robust library selection
try:
    from library_fallback_handler import fallback_handler
    FALLBACK_AVAILABLE = True
except ImportError:
    FALLBACK_AVAILABLE = False
    print("⚠️  Fallback handler not available - using basic library selection")

# Load environment variables
load_dotenv()

class ExamRAGAgent:
    """Agentic RAG system for exam paper generation"""
    
    def __init__(self, config_path: str, syllabus_path: str, notes_path: str, pyq_path: str, vector_cache_dir: str, processed_data: Optional[Dict] = None):
        """Initialize the RAG agent"""
        # Store paths
        self.config_path = config_path
        self.syllabus_path = syllabus_path
        self.notes_path = notes_path
        self.pyq_path = pyq_path
        self.vector_cache_dir = vector_cache_dir
        
        # Store processed document data if provided
        self.processed_data = processed_data
        if processed_data:
            print(f"📋 Using processed document data - Subject: {processed_data.get('metadata', {}).get('subject_type', 'Unknown')}")

        # Load config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize LLM
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.6)
        
        # Create output directories if they don't exist
        os.makedirs(self.vector_cache_dir, exist_ok=True)
        os.makedirs("diagrams", exist_ok=True)
        os.makedirs("generated_exams", exist_ok=True)
        
        # Track document sources
        self.documents = {
            "syllabus": [],
            "notes": [],
            "pyq": []
        }
        
        # Initialize vector stores
        self.vector_stores = {}
        
        # Generated questions
        self.exam_questions = []
    
    def check_required_files(self) -> bool:
        """Check if all required files are present"""
        required_files = {
            "syllabus": self.syllabus_path,
            "notes": self.notes_path,
            "pyq": self.pyq_path
        }
        
        missing_files = []
        for doc_type, filepath in required_files.items():
            if not os.path.exists(filepath):
                missing_files.append(filepath)
        
        if missing_files:
            print("ERROR: The following required files are missing:")
            for file in missing_files:
                print(f"  - {file}")
            print("\nPlease ensure these files are available.")
            return False
        
        return True
    
    def load_documents(self):
        """Load all documents using PyPDFLoader"""
        if not self.check_required_files():
            return False
        
        document_mapping = {
            "syllabus": self.syllabus_path,
            "notes": self.notes_path,
            "pyq": self.pyq_path
        }
        
        # Load each document type
        for doc_type, filepath in document_mapping.items():
            try:
                print(f"Loading {filepath}...")
                loader = PyPDFLoader(filepath)
                documents = loader.load()
                
                # Add source metadata
                for doc in documents:
                    doc.metadata["source"] = doc_type
                
                self.documents[doc_type] = documents
                print(f"Successfully loaded {filepath}: {len(documents)} pages")
            except Exception as e:
                print(f"ERROR loading {filepath}: {e}")
                return False
        
        return True
    
    def create_vector_stores(self) -> bool:
        """Create vector stores from documents with caching"""
        try:
            os.makedirs(self.vector_cache_dir, exist_ok=True)
            
            embeddings = OpenAIEmbeddings()
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            
            # Create/Load Syllabus Vector Store
            syllabus_cache_path = os.path.join(self.vector_cache_dir, "syllabus_vector_store")
            if os.path.exists(syllabus_cache_path):
                print("Loading syllabus vector store from cache...")
                self.vector_stores["syllabus"] = FAISS.load_local(syllabus_cache_path, embeddings, allow_dangerous_deserialization=True)
                print("Syllabus vector store loaded from cache.")
            elif self.documents.get("syllabus"):
                print("Creating syllabus vector store...")
                syllabus_chunks = splitter.split_documents(self.documents["syllabus"])
                self.vector_stores["syllabus"] = FAISS.from_documents(syllabus_chunks, embeddings)
                print("Syllabus vector store created.")
                print(f"Saving syllabus vector store to cache in {self.vector_cache_dir}...")
                self.vector_stores["syllabus"].save_local(syllabus_cache_path)
                print("Syllabus vector store saved to cache.")
            else:
                print("No syllabus documents loaded or cached.")

            # Create/Load Notes Vector Store
            notes_cache_path = os.path.join(self.vector_cache_dir, "notes_vector_store")
            if os.path.exists(notes_cache_path):
                print("Loading notes vector store from cache...")
                self.vector_stores["notes"] = FAISS.load_local(notes_cache_path, embeddings, allow_dangerous_deserialization=True)
                print("Notes vector store loaded from cache.")
            elif self.documents.get("notes"):
                print("Creating notes vector store...")
                notes_chunks = splitter.split_documents(self.documents["notes"])
                self.vector_stores["notes"] = FAISS.from_documents(notes_chunks, embeddings)
                print("Notes vector store created.")
                print(f"Saving notes vector store to cache in {self.vector_cache_dir}...")
                self.vector_stores["notes"].save_local(notes_cache_path)
                print("Notes vector store saved to cache.")
            else:
                print("No notes documents loaded or cached.")
                
            # Load PYQ Vector Store from cache
            pyq_cache_path = os.path.join(self.vector_cache_dir, "pyq_vector_store")
            if os.path.exists(pyq_cache_path):
                print("Loading PYQ vector store from cache...")
                self.vector_stores["pyq"] = FAISS.load_local(pyq_cache_path, embeddings, allow_dangerous_deserialization=True)
                print("PYQ vector store loaded from cache.")
            elif self.documents.get("pyq"):
                 print("PYQ vector store cache not found, creating from loaded documents...")
                 pyq_chunks = splitter.split_documents(self.documents["pyq"])
                 self.vector_stores["pyq"] = FAISS.from_documents(pyq_chunks, embeddings)
                 print("PYQ vector store created.")
                 print(f"Saving PYQ vector store to cache in {self.vector_cache_dir}...")
                 self.vector_stores["pyq"].save_local(pyq_cache_path)
                 print("PYQ vector store saved to cache.")
            else:
                 print("No PYQ documents loaded and no cache found.")

            if not all(store in self.vector_stores for store in ["syllabus", "notes", "pyq"]):
                 print("ERROR: Missing required vector stores.")
                 return False
            
            print("All required vector stores created/loaded.")
            return True
        except Exception as e:
            print(f"ERROR creating/loading vector stores: {e}")
            self.vector_stores = {}
            return False
    
    def extract_syllabus_topics(self) -> List[str]:
        """Extract key topics from syllabus using processed document data or LLM"""
        
        # First, try to use processed document topics if available
        if self.processed_data and 'extracted_topics' in self.processed_data:
            processed_topics = self.processed_data['extracted_topics']
            if processed_topics and len(processed_topics) > 0:
                print(f"📋 Using {len(processed_topics)} topics from processed documents")
                return processed_topics
        
        # Fallback to vector store extraction if no processed data
        if "syllabus" not in self.vector_stores:
            print("ERROR: Syllabus vector store not found")
            return self.get_fallback_topics()
        
        # Create prompt for topic extraction
        topic_extraction_template = """
        You are an educational content analyzer.
        
        Extract the main topics and subtopics from this syllabus content.
        Focus on identifying specific technical topics that would be suitable for exam questions.
        Format your response as a JSON list of strings, where each string is a topic or subtopic.
        
        Syllabus content:
        {syllabus_content}
        
        Return ONLY a JSON array of topic strings.
        """
        
        # Get representative content from syllabus
        syllabus_retriever = self.vector_stores["syllabus"].as_retriever(search_kwargs={"k": 10})
        syllabus_docs = syllabus_retriever.get_relevant_documents("curriculum topics main units")
        syllabus_content = "\n\n".join([doc.page_content for doc in syllabus_docs])
        
        # Extract topics
        prompt = PromptTemplate(
            input_variables=["syllabus_content"],
            template=topic_extraction_template
        )
        
        try:
            # Try new method first
            chain = prompt | self.llm
            response = chain.invoke({"syllabus_content": syllabus_content}).content
        except:
            # Fallback to old method
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.invoke({"syllabus_content": syllabus_content})["text"]
        
        try:
            # Clean the response first
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            topics = json.loads(response)
            if isinstance(topics, list):
                print(f"Extracted {len(topics)} topics from syllabus")
                return topics
            else:
                print("ERROR: Topics response is not a list")
                return self.get_fallback_topics()
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse topics from LLM response: {e}")
            print(f"Response was: {response[:200]}...")
            return self.get_fallback_topics()
    
    def get_fallback_topics(self) -> List[str]:
        """Fallback topics when extraction fails - use only from processed documents"""
        
        # Always use processed data topics if available
        if self.processed_data and 'topics' in self.processed_data:
            topics = self.processed_data['topics']
            if topics:
                print(f"📋 Using {len(topics)} topics from processed documents")
                return topics
        
        # If no processed topics available, use generic placeholders
        print("⚠️ No processed topics available, using generic placeholders")
        return [
            "Topic 1 from Course Materials",
            "Topic 2 from Course Materials", 
            "Topic 3 from Course Materials",
            "Topic 4 from Course Materials",
            "Topic 5 from Course Materials",
            "Topic 6 from Course Materials",
            "Topic 7 from Course Materials",
            "Topic 8 from Course Materials",
            "Topic 9 from Course Materials",
            "Topic 10 from Course Materials"
        ]
    
    def analyze_topic_frequency(self, topics: List[str]) -> Dict[str, int]:
        """Analyze how frequently each topic appears in PYQs"""
        if "pyq" not in self.vector_stores:
            print("ERROR: PYQ vector store not found")
            return {topic: 1 for topic in topics}  # Default equal weights
        
        topic_counts = {}
        pyq_retriever = self.vector_stores["pyq"].as_retriever(search_kwargs={"k": 5})
        
        for topic in topics:
            # Find PYQ documents relevant to this topic
            relevant_docs = pyq_retriever.get_relevant_documents(topic)
            topic_counts[topic] = len(relevant_docs)
            
        print(f"Analyzed frequency for {len(topics)} topics")
        return topic_counts
    
    def select_topics_for_section(self, section: Dict[str, Any], topics: List[str], 
                                 topic_counts: Dict[str, int]) -> List[str]:
        """Select topics for a section based on weights and PYQ frequency"""
        section_type = section.get("type", "ShortAnswer")
        count = section.get("count", 5)
        
        # Sort topics by frequency
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_topic_names = [t[0] for t in sorted_topics]
        
        # Apply different selection strategies based on section type
        if section_type == "ShortAnswer":
            # Mix of high and medium frequency topics
            high_freq = sorted_topic_names[:int(len(sorted_topic_names) * 0.3)]
            medium_freq = sorted_topic_names[int(len(sorted_topic_names) * 0.3):int(len(sorted_topic_names) * 0.7)]
            selected = random.sample(high_freq, min(count // 2, len(high_freq)))
            selected += random.sample(medium_freq, min(count - len(selected), len(medium_freq)))
        
        elif section_type == "Problem":
            # Medium frequency topics
            medium_freq = sorted_topic_names[int(len(sorted_topic_names) * 0.2):int(len(sorted_topic_names) * 0.8)]
            selected = random.sample(medium_freq, min(count, len(medium_freq)))
            
        else:  # Long answers or applications
            # Mix of medium and low frequency (more challenging) topics
            medium_freq = sorted_topic_names[int(len(sorted_topic_names) * 0.3):int(len(sorted_topic_names) * 0.7)]
            low_freq = sorted_topic_names[int(len(sorted_topic_names) * 0.7):]
            selected = random.sample(medium_freq, min(count // 2, len(medium_freq)))
            selected += random.sample(low_freq, min(count - len(selected), len(low_freq)))
        
        # If we don't have enough topics, fill with random topics
        if len(selected) < count:
            remaining = list(set(topics) - set(selected))
            selected += random.sample(remaining, min(count - len(selected), len(remaining)))
        
        return selected[:count]
    
    def ai_determine_subject_type(self, topic: str, question: str = "") -> str:
        """Use processed document data only - no AI bias"""
        
        # Always use processed document data if available
        if self.processed_data and 'metadata' in self.processed_data:
            subject_type = self.processed_data['metadata'].get('subject_type')
            if subject_type:
                print(f"📋 Using processed document subject: {subject_type}")
                return subject_type
        
        # If no processed data, return generic type without bias
        print(f"📋 No processed data available, using generic subject type")
        return "Computer Science - Content Based"
    
    def ai_requires_diagram(self, question: str, topic: str) -> bool:
        """Use AI to intelligently determine if a question needs a diagram across ALL subjects"""
        
        diagram_analysis_template = """
        You are an educational expert analyzing whether a question would benefit from a visual diagram.
        
        Analyze this question and determine if a diagram would genuinely help students understand or answer it.
        
        QUESTION: {question}
        TOPIC: {topic}
        
        Consider these factors:
        1. Does the question explicitly ask for a visual representation (draw, sketch, diagram, illustrate)?
        2. Would a visual aid significantly enhance understanding of the concept?
        3. Is the concept inherently visual or structural in nature?
        4. Would students benefit from seeing relationships, processes, or structures?
        
        Examples of questions that NEED diagrams:
        - "Draw the architecture of a system"
        - "Illustrate the process flow"
        - "Show the structure of a component"
        - "Explain the cycle with a diagram"
        - "Draw the layout for a design"
        - "Illustrate the relationship between variables"
        
        Examples of questions that DON'T need diagrams:
        - "Define a concept"
        - "What is the complexity of an operation?"
        - "List the advantages of an approach"
        - "Calculate a mathematical expression"
        - "Explain a theoretical concept"
        - "Compare different methods"
        
        Return ONLY a JSON object with this exact format:
        {{
            "needs_diagram": true/false,
            "reason": "Brief explanation of why diagram is/isn't needed",
            "diagram_type": "Type of diagram that would be helpful (if needed)"
        }}
        
        Be conservative - only return true if a diagram would genuinely add educational value.
        """
        
        try:
            # Create prompt
            prompt = PromptTemplate(
                input_variables=["question", "topic"],
                template=diagram_analysis_template
            )
            
            # Generate analysis
            try:
                # Try new method first
                chain = prompt | self.llm
                response = chain.invoke({"question": question, "topic": topic}).content
            except:
                # Fallback to old method
                chain = LLMChain(llm=self.llm, prompt=prompt)
                response = chain.invoke({"question": question, "topic": topic})["text"]
            
            # Parse response
            # Clean the response first
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            analysis = json.loads(response)
            needs_diagram = analysis.get("needs_diagram", False)
            reason = analysis.get("reason", "No reason provided")
            
            if needs_diagram:
                print(f"🤖 AI: Diagram needed - {reason}")
            else:
                print(f"🤖 AI: No diagram needed - {reason}")
            
            return needs_diagram
            
        except Exception as e:
            print(f"⚠️ AI analysis failed, using fallback logic: {e}")
            # Fallback to simple keyword detection
            return self.simple_diagram_check(question, topic)
    
    def simple_diagram_check(self, question: str, topic: str) -> bool:
        """Simple fallback diagram detection"""
        question_lower = question.lower()
        
        # Explicit diagram requests
        explicit_keywords = ['diagram', 'draw', 'sketch', 'illustrate', 'flowchart', 'graph', 'chart', 'plot']
        if any(keyword in question_lower for keyword in explicit_keywords):
            return True
        
        # Questions that typically don't need diagrams
        no_diagram_keywords = ['define', 'what is', 'list', 'calculate', 'time complexity', 'advantages', 'disadvantages']
        if any(keyword in question_lower for keyword in no_diagram_keywords):
            return False
        
        return False

    def filter_content(self, content: str) -> str:
        """Filter out irrelevant example sentences and random content"""
        if not content:
            return content
        
        # List of patterns to filter out
        filter_patterns = [
            r"Consider these examples:\s*\n.*?Dogs like bones.*?\n",
            r".*Dogs like bones.*?\n",
            r".*I ate rice.*?\n", 
            r".*Gold is valuable.*?\n",
            r".*I saw some gold.*?\n",
            r".*I saw gold.*?\n",
            # Filter out simple example sentences that are not academic content
            r"\d+\.\s*[A-Z][a-z\s]*\.\s*\n",  # Simple numbered examples like "1. Dogs like bones."
        ]
        
        filtered_content = content
        for pattern in filter_patterns:
            filtered_content = re.sub(pattern, "", filtered_content, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove multiple newlines
        filtered_content = re.sub(r'\n\s*\n\s*\n', '\n\n', filtered_content)
        
        return filtered_content.strip()

    def generate_question(self, topic: str, section_type: str, marks: int, section_name: str) -> Dict[str, Any]:
        """Generate a single exam question using agentic RAG"""
        # Retrieve relevant context from each document type
        contexts = {}
        
        for doc_type, vector_store in self.vector_stores.items():
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(topic)
            if docs:
                raw_content = "\n\n".join([d.page_content for d in docs])
                # Filter out irrelevant content
                contexts[doc_type] = self.filter_content(raw_content)
        
        # Construct the prompt template based on section type
        if section_type == "ShortAnswer":
            template = """
            You are an expert exam setter creating high-quality academic questions.
            
            Generate a SHORT-ANSWER question worth {marks} marks on: "{topic}"
            
            CONTENT-BASED REQUIREMENTS:
            - Base your question ONLY on the provided course materials below
            - Use concepts, terminology, and examples from the materials
            - Match the academic level and style shown in the materials
            - DO NOT add external knowledge not present in the materials
            - IGNORE any simple example sentences like "Dogs like bones", "I ate rice", etc.
            - Focus ONLY on academic and technical content relevant to the subject
            
            QUESTION REQUIREMENTS:
            - Clear, specific, and unambiguous wording
            - Appropriate difficulty for {marks} marks (typically 2-4 key points expected)
            - Can be answered in 50-100 words or a small calculation
            - Focus on: definitions, explanations, comparisons, or simple applications
            
            VERSATILE QUESTION TYPES:
            - "Define X and explain its significance in Y"
            - "Compare and contrast A vs B"
            - "List and briefly explain the main components/steps/types of X"
            - "Calculate/Determine X given the following parameters"
            - "What is the purpose/role/function of X in Y?"
            - "Identify and explain the key characteristics of X"
            
            Return ONLY valid JSON:
            {{
              "question": "Your clear, specific question here",
              "answer": "Comprehensive model answer with key points",
              "marks": {marks},
              "topic": "{topic}",
              "diagram_needed": false
            }}
            
            COURSE MATERIALS:
            
            SYLLABUS: {syllabus_content}
            
            LECTURE NOTES: {notes_content}
            
            PREVIOUS QUESTIONS: {pyq_content}
            """
        
        elif section_type == "Problem":
            template = """
            You are an expert exam setter creating analytical problem-solving questions.
            
            Generate a PROBLEM-SOLVING question worth {marks} marks on: "{topic}"
            
            CONTENT-BASED REQUIREMENTS:
            - Base your question on formulas, methods, and examples from the provided materials
            - Use specific values, parameters, and scenarios from the course content
            - Ensure all required information for solving is either given or derivable
            - Match the complexity level shown in previous questions and notes
            - IGNORE any simple example sentences like "Dogs like bones", "I ate rice", etc.
            - Focus ONLY on academic and technical content relevant to the subject
            
            PROBLEM REQUIREMENTS:
            - Include specific numerical values, parameters, or concrete scenarios
            - Require analytical thinking, calculations, or step-by-step solutions
            - Difficulty appropriate for {marks} marks (typically 3-6 solution steps)
            - Clear problem statement with all necessary information
            
            VERSATILE PROBLEM TYPES:
            - "Calculate X given parameters A, B, C and show your working"
            - "Design/Analyze system Y with specifications Z and justify your approach"
            - "Solve for optimal X under constraints A, B using method Y"
            - "Compare performance of methods A vs B for scenario X"
            - "Derive formula/relationship for X in terms of Y and Z"
            - "Troubleshoot/Debug system X given symptoms Y and propose solution"
            
            Return ONLY valid JSON:
            {{
              "question": "Complete problem statement with all necessary data",
              "answer_outline": ["Step 1: Identify given parameters", "Step 2: Apply relevant formula/method", "Step 3: Calculate intermediate values", "Step 4: Compute final answer", "Step 5: Verify/interpret result"],
              "marks": {marks},
              "topic": "{topic}",
              "diagram_needed": false
            }}
            
            COURSE MATERIALS:
            
            SYLLABUS: {syllabus_content}
            
            LECTURE NOTES: {notes_content}
            
            PREVIOUS QUESTIONS: {pyq_content}
            """
        
        else:  # LongAnswer or Application
            template = """
            You are an expert exam setter creating comprehensive analytical questions.
            
            Generate a LONG-ANSWER question worth {marks} marks on: "{topic}"
            
            CONTENT-BASED REQUIREMENTS:
            - Draw from theories, concepts, and detailed explanations in the provided materials
            - Use specific examples, case studies, or scenarios from the course content
            - Incorporate multiple related concepts to demonstrate comprehensive understanding
            - Match the depth and analytical level expected at this academic level
            - IGNORE any simple example sentences like "Dogs like bones", "I ate rice", etc.
            - Focus ONLY on academic and technical content relevant to the subject
            
            QUESTION REQUIREMENTS:
            - Require detailed explanation, analysis, or comprehensive discussion
            - Multiple interconnected parts or aspects to address
            - Difficulty appropriate for {marks} marks (typically 6-10 key points/concepts)
            - Encourage critical thinking and application of knowledge
            
            VERSATILE QUESTION TYPES:
            - "Explain the concept of X, its components, applications, and analyze its advantages/limitations"
            - "Discuss the relationship between A and B, and evaluate their impact on system Y"
            - "Analyze problem X using theory Y, and propose solutions with justification"
            - "Compare multiple approaches for X and recommend the best with reasoning"
            - "Explain how system X works, its design principles, and discuss real-world applications"
            - "Critically evaluate the role of X in Y, discussing both theoretical and practical aspects"
            
            Return ONLY valid JSON:
            {{
              "question": "Comprehensive question requiring detailed analysis and explanation",
              "answer_outline": ["Introduction: Define key concepts and scope", "Main Analysis: Detailed explanation of core principles", "Applications: Real-world examples and use cases", "Critical Evaluation: Advantages, limitations, and comparisons", "Conclusion: Summary and broader implications"],
              "marks": {marks},
              "topic": "{topic}",
              "diagram_needed": false
            }}
            
            COURSE MATERIALS:
            
            SYLLABUS: {syllabus_content}
            
            LECTURE NOTES: {notes_content}
            
            PREVIOUS QUESTIONS: {pyq_content}
            """
        
        # Fill in the context
        prompt_inputs = {
            "topic": topic,
            "marks": marks,
            "syllabus_content": contexts.get("syllabus", "No relevant syllabus content found."),
            "notes_content": contexts.get("notes", "No relevant lecture notes found."),
            "pyq_content": contexts.get("pyq", "No relevant previous year questions found.")
        }
        
        # Create prompt
        prompt = PromptTemplate(
            input_variables=["topic", "marks", "syllabus_content", "notes_content", "pyq_content"],
            template=template
        )
        
        # Generate question
        try:
            # Try new method first
            chain = prompt | self.llm
            response = chain.invoke(prompt_inputs).content
        except:
            # Fallback to old method
            chain = LLMChain(llm=self.llm, prompt=prompt)
            response = chain.invoke(prompt_inputs)["text"]
        
        try:
            # Clean the response first
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            question_json = json.loads(response)
            question_json["section"] = section_name
            
            # Use AI to intelligently determine if this question needs a diagram
            needs_diagram = self.ai_requires_diagram(question_json["question"], topic)
            
            if needs_diagram:
                print(f"📊 Generating diagram for question: {topic}")
                subject_type = self.ai_determine_subject_type(topic, question_json["question"])
                diagram_code = self.generate_contextual_diagram_code(
                    question_json["question"], 
                    topic, 
                    subject_type
                )
                
                # Add diagram information to question
                question_json["diagram_code"] = diagram_code
                question_json["diagram_type"] = "contextual"
                question_json["diagram_library"] = "matplotlib"
                question_json["diagram_filename"] = f"{topic.replace(' ', '_').lower()}_diagram.png"
                question_json["has_diagram"] = True
            else:
                print(f"⏭️  No diagram needed for: {topic}")
                question_json["has_diagram"] = False
            
            return question_json
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse question for {topic}: {e}")
            print(f"Response was: {response[:200]}...")
            return {
                "question": f"Question about {topic} (could not generate properly)",
                "answer": "Answer not available due to generation error.",
                "marks": marks,
                "topic": topic,
                "section": section_name,
                "has_diagram": False
            }
    
    def generate_exam(self) -> List[Dict[str, Any]]:
        """Generate a full exam paper using agentic RAG"""
        print("Starting exam generation with Agentic RAG...")
        
        # Load and process documents
        if not self.load_documents():
            return []
        
        # Create vector stores
        if not self.create_vector_stores():
            return []
        
        # Extract topics from syllabus
        topics = self.extract_syllabus_topics()
        if not topics:
            print("ERROR: Failed to extract topics from syllabus")
            return []
        
        # Analyze topic frequency in PYQs
        topic_counts = self.analyze_topic_frequency(topics)
        
        # Generate questions for each section
        exam_questions = []
        
        # Add exam header and special instructions
        exam_questions.append({
            "type": "header",
            "content": self.config.get("header", ""),
            "special_instructions": self.config.get("special_instructions", "")
        })
        
        # Get sections from config
        sections = self.config.get("sections", [])
        if not sections:
            print("ERROR: No sections found in config")
            return []
            
        print(f"Generating exam with {len(sections)} sections")
        
        for section in sections:
            section_name = section["name"]
            section_type = section["type"]
            marks_each = section["marks_each"]
            count = section["count"]
            
            print(f"\nGenerating {count} questions for {section_name} ({section_type})")
            
            # Select topics for this section
            section_topics = self.select_topics_for_section(section, topics, topic_counts)
            
            # Generate each question
            for i, topic in enumerate(section_topics[:count]):  # Limit to exact count from config
                print(f"  Generating question {i+1}/{count} on '{topic}'")
                question = self.generate_question(topic, section_type, marks_each, section_name)
                exam_questions.append(question)
                print(f"  ✓ Generated: {question.get('question', '')[:50]}...")
            
            print(f"Completed section: {section_name}")
        
        self.exam_questions = exam_questions
        print(f"\nExam generation complete. Generated {len(exam_questions)-1} questions.")  # -1 for header
        return exam_questions
    
    def save_exam_json(self, filename="generated_exams/exam.json"):
        """Save the generated exam as JSON"""
        if not self.exam_questions:
            print("No exam questions to save.")
            return
        
        with open(filename, 'w') as f:
            json.dump(self.exam_questions, f, indent=2)
        
        print(f"Exam saved to {filename}")
    
    def determine_forced_library(self, question: str, topic: str, subject_type: str) -> str:
        """Determine which library to force based on question content - no subject bias"""
        question_lower = question.lower()
        topic_lower = topic.lower()
        
        # Network/Graph related - could be any subject
        if any(keyword in question_lower or keyword in topic_lower for keyword in 
               ['network', 'topology', 'graph', 'node', 'connection', 'relationship', 'hierarchy']):
            return "networkx"
        
        # State/Flow diagrams - could be any subject
        if any(keyword in question_lower or keyword in topic_lower for keyword in 
               ['state', 'flow', 'transition', 'lifecycle', 'process', 'steps']):
            return "graphviz"
        
        # Performance/Analysis/Timeline - could be any subject
        if any(keyword in question_lower or keyword in topic_lower for keyword in 
               ['performance', 'time', 'analysis', 'comparison', 'timeline', 'trends']):
            return "plotly"
        
        # Architecture/Structure/Layout - could be any subject
        if any(keyword in question_lower or keyword in topic_lower for keyword in 
               ['architecture', 'structure', 'layout', 'design', 'framework']):
            return "plotly"
        
        # Tree/Hierarchy structures - could be any subject
        if any(keyword in question_lower or keyword in topic_lower for keyword in 
               ['tree', 'hierarchy', 'classification', 'taxonomy']):
            return "networkx"
        
        # Statistical/Data Analysis - could be any subject
        if any(keyword in question_lower or keyword in topic_lower for keyword in 
               ['statistics', 'data', 'correlation', 'distribution', 'probability']):
            return "seaborn"
        
        # Default to plotly for general visualizations
        return "plotly"
    
    def get_library_specific_examples(self, library: str, subject_type: str) -> str:
        """Get library-specific examples - content neutral"""
        
        if library == "matplotlib":
            return """
# Matplotlib for basic plotting and visualization:
import matplotlib.pyplot as plt
import numpy as np

# Basic line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='Data')
plt.title('Data Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.savefig('plot.png', dpi=300, bbox_inches='tight')
plt.show()

# Bar chart
categories = ['A', 'B', 'C', 'D']
values = [10, 25, 15, 30]
plt.figure(figsize=(8, 6))
plt.bar(categories, values, color=['red', 'blue', 'green', 'orange'])
plt.title('Category Comparison')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.savefig('bar_chart.png', dpi=300, bbox_inches='tight')
plt.show()
"""
        
        elif library == "plotly":
            return """
# Plotly for interactive visualizations:
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# Interactive line plot
x = np.linspace(0, 10, 100)
y = np.sin(x)
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Data'))
fig.update_layout(title='Interactive Data Visualization',
                  xaxis_title='X-axis',
                  yaxis_title='Y-axis')
fig.write_image('interactive_plot.png')

# 3D surface plot
x, y = np.meshgrid(np.linspace(-5, 5, 20), np.linspace(-5, 5, 20))
z = np.sin(np.sqrt(x**2 + y**2))
fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
fig.update_layout(title='3D Surface Visualization')
fig.write_image('3d_surface.png')
"""
        
        elif library == "networkx":
            return """
# NetworkX for network and graph visualization:
import networkx as nx
import matplotlib.pyplot as plt

# Create a graph
G = nx.Graph()
G.add_edges_from([('A','B'), ('B','C'), ('C','D'), ('D','A'), ('A','C')])

# Draw the graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1500)
nx.draw_networkx_labels(G, pos, font_size=16)
nx.draw_networkx_edges(G, pos, edge_color='gray', width=2)
plt.title('Network Graph Visualization')
plt.axis('off')
plt.savefig('network_graph.png', dpi=300, bbox_inches='tight')
plt.show()

# Directed graph
DG = nx.DiGraph()
DG.add_edges_from([('Start','A'), ('A','B'), ('B','End')])
pos = nx.spring_layout(DG)
nx.draw_networkx(DG, pos, with_labels=True, node_color='lightgreen', 
                arrows=True, arrowsize=20, node_size=2000)
plt.title('Directed Graph')
plt.savefig('directed_graph.png', dpi=300, bbox_inches='tight')
"""
        
        elif library == "graphviz":
            return """
# Graphviz for structured diagrams:
from graphviz import Digraph
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# Create a flowchart
flowchart = Digraph('flowchart', comment='Process Flow')
flowchart.attr(rankdir='TB')
flowchart.node('start', 'Start', shape='oval')
flowchart.node('process1', 'Process 1', shape='box')
flowchart.node('decision', 'Decision?', shape='diamond')
flowchart.node('process2', 'Process 2', shape='box')
flowchart.node('end', 'End', shape='oval')

flowchart.edge('start', 'process1')
flowchart.edge('process1', 'decision')
flowchart.edge('decision', 'process2', label='Yes')
flowchart.edge('decision', 'end', label='No')
flowchart.edge('process2', 'end')

flowchart.render('flowchart.png', format='png', cleanup=True)

# State diagram
states = Digraph('states', comment='State Diagram')
states.attr(rankdir='LR')
states.node('S1', 'State 1', shape='circle')
states.node('S2', 'State 2', shape='circle')
states.node('S3', 'State 3', shape='doublecircle')
states.edge('S1', 'S2', label='condition 1')
states.edge('S2', 'S3', label='condition 2')
states.edge('S2', 'S1', label='condition 3')
states.render('states.png', format='png', cleanup=True)
"""
        
        elif library == "seaborn":
            return """
# Seaborn for statistical visualization:
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create sample data
data = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'D', 'E'],
    'Value1': [10, 25, 15, 30, 20],
    'Value2': [15, 20, 25, 35, 18]
})

# Bar plot
plt.figure(figsize=(10, 6))
sns.barplot(data=data, x='Category', y='Value1')
plt.title('Statistical Visualization')
plt.savefig('seaborn_bar.png', dpi=300, bbox_inches='tight')

# Correlation heatmap
correlation_data = np.random.randn(5, 5)
correlation_df = pd.DataFrame(correlation_data, 
                            columns=['Var1', 'Var2', 'Var3', 'Var4', 'Var5'])
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
"""
        
        else:
            # Default matplotlib example
            return """
# Default visualization with matplotlib:
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Data Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.savefig('visualization.png', dpi=300, bbox_inches='tight')
plt.show()
"""

    def generate_contextual_diagram_code(self, question: str, topic: str, subject_type: str) -> str:
        """Generate Python diagram code contextually based on the specific question - content neutral"""
        
        # Determine the best library for this question
        forced_library = self.determine_forced_library(question, topic, subject_type)
        
        # Get content-neutral examples for the chosen library
        library_examples = self.get_library_specific_examples(forced_library, subject_type)
        
        # Create the diagram generation prompt
        diagram_prompt = f"""
You are an expert Python visualization developer. Generate Python code to create a diagram that directly addresses this question.

QUESTION: {question}
TOPIC: {topic}
PREFERRED LIBRARY: {forced_library}

Requirements:
1. Create a diagram that directly relates to the question content
2. Use the {forced_library} library as the primary visualization tool
3. Make the diagram informative and educational
4. Include proper titles, labels, and legends
5. Save the diagram as a PNG file
6. Use realistic data/examples that relate to the question topic

Available library examples for reference:
{library_examples}

Generate ONLY the Python code (no explanations, no markdown formatting).
The code should be complete and runnable.
Focus on creating a diagram that helps visualize or explain the concepts in the question.
"""

        try:
            # Generate the diagram code using LLM
            try:
                # Try new method first
                response = self.llm.invoke(diagram_prompt).content
            except:
                # Fallback to old method
                response = self.llm(diagram_prompt)
            
            # Clean the response
            if isinstance(response, dict) and 'text' in response:
                response = response['text']
            
            # Remove markdown formatting if present
            if response.startswith("```python"):
                response = response[9:]
            elif response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            response = response.strip()
            
            print(f"📊 Generated diagram code using {forced_library} library")
            return response
            
        except Exception as e:
            print(f"⚠️ Diagram code generation failed: {e}")
            # Return a simple fallback diagram
            return f"""
import matplotlib.pyplot as plt
import numpy as np

# Simple visualization for: {topic}
plt.figure(figsize=(10, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, 'b-', linewidth=2)
plt.title('{topic} - Visualization')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.savefig('diagram.png', dpi=300, bbox_inches='tight')
plt.show()
"""

if __name__ == "__main__":
    # Create agent
    agent = ExamRAGAgent(config_path="exam_config.json", syllabus_path="syllabus.pdf", notes_path="notes.pdf", pyq_path="pyqs.pdf", vector_cache_dir="vector_cache")
    
    # Generate exam
    questions = agent.generate_exam()
    
    # Save exam
    if questions:
        agent.save_exam_json()
