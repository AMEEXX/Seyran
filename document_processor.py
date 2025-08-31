#!/usr/bin/env python3
"""
Document Processor for LLMQ
Extracts all information from uploaded documents and creates structured JSON with GPT instructions
"""

import json
import os
import PyPDF2
import re
from typing import Dict, List, Any
from datetime import datetime

class DocumentProcessor:
    def __init__(self, upload_folder: str = "uploads"):
        self.upload_folder = upload_folder
        self.processed_data = {}
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def analyze_content_type(self, content: str) -> str:
        """Analyze content to determine subject type using AI"""
        try:
            # Use GPT to intelligently determine the subject
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
            
            # Create a sample of content for analysis (first 3000 chars to avoid token limits)
            content_sample = content[:3000] if len(content) > 3000 else content
            
            subject_analysis_prompt = f"""
            Analyze the following academic content and determine the most appropriate subject category.
            
            Content sample:
            {content_sample}
            
            Based on the content, determine the subject category. Respond with ONLY the subject name in this format:
            "Subject Area - Specific Topic"
            
            Examples of good responses:
            - Computer Science - Artificial Intelligence
            - Computer Science - Operating Systems  
            - Computer Science - Networks
            - Computer Science - Database Systems
            - Computer Science - Algorithms
            - Computer Science - Software Engineering
            - Mathematics - Calculus
            - Physics - Quantum Mechanics
            - Electronics - Digital Logic
            - Engineering - Control Systems
            
            Be specific and accurate based on the actual content provided. Do not use generic categories.
            """
            
            response = llm.invoke(subject_analysis_prompt)
            subject_type = response.content.strip().strip('"')
            
            print(f"🎯 AI-determined subject: {subject_type}")
            return subject_type
            
        except Exception as e:
            print(f"⚠️ Error in AI subject analysis: {e}")
            # Fallback to simple keyword-based detection
            return self._fallback_subject_detection(content)
    
    def _fallback_subject_detection(self, content: str) -> str:
        """Fallback method for subject detection using basic keywords"""
        content_lower = content.lower()
        
        # Basic broad categories
        if any(keyword in content_lower for keyword in ["artificial intelligence", "machine learning", "neural network", "ai", "expert system", "knowledge base", "inference"]):
            return "Computer Science - Artificial Intelligence"
        elif any(keyword in content_lower for keyword in ["operating system", "process", "thread", "kernel", "memory management"]):
            return "Computer Science - Operating Systems"
        elif any(keyword in content_lower for keyword in ["network", "protocol", "tcp", "internet", "routing"]):
            return "Computer Science - Networks"
        elif any(keyword in content_lower for keyword in ["algorithm", "data structure", "sorting", "complexity"]):
            return "Computer Science - Algorithms"
        elif any(keyword in content_lower for keyword in ["database", "sql", "query", "dbms"]):
            return "Computer Science - Database Systems"
        elif any(keyword in content_lower for keyword in ["calculus", "algebra", "mathematics", "theorem"]):
            return "Mathematics"
        else:
            return "General Studies"
    
    def _determine_library_config(self, subject_type: str) -> Dict[str, Any]:
        """Dynamically determine library configuration based on subject type"""
        subject_lower = subject_type.lower()
        
        # AI/ML subjects
        if any(keyword in subject_lower for keyword in ["artificial intelligence", "machine learning", "ai", "ml", "neural", "deep learning"]):
            return {
                "primary": "plotly",
                "secondary": ["networkx", "seaborn"],
                "forbidden": ["matplotlib"],
                "use_cases": {
                    "neural networks": "networkx",
                    "decision trees": "plotly",
                    "performance metrics": "plotly",
                    "data visualization": "seaborn",
                    "algorithms": "networkx"
                }
            }
        
        # Operating Systems
        elif any(keyword in subject_lower for keyword in ["operating system", "os", "process", "kernel"]):
            return {
                "primary": "graphviz",
                "secondary": ["networkx", "plotly"],
                "forbidden": ["matplotlib", "seaborn"],
                "use_cases": {
                    "process states": "graphviz",
                    "scheduling": "plotly",
                    "memory layout": "plotly",
                    "file systems": "networkx",
                    "synchronization": "graphviz"
                }
            }
        
        # Networks
        elif any(keyword in subject_lower for keyword in ["network", "networking", "protocol", "tcp", "internet"]):
            return {
                "primary": "networkx",
                "secondary": ["plotly", "graphviz"],
                "forbidden": ["matplotlib", "seaborn"],
                "use_cases": {
                    "topology": "networkx",
                    "protocols": "graphviz",
                    "performance": "plotly",
                    "architecture": "networkx"
                }
            }
        
        # Algorithms & Data Structures
        elif any(keyword in subject_lower for keyword in ["algorithm", "data structure", "sorting", "complexity"]):
            return {
                "primary": "graphviz",
                "secondary": ["networkx", "plotly"],
                "forbidden": ["matplotlib", "seaborn"],
                "use_cases": {
                    "flowcharts": "graphviz",
                    "trees": "networkx",
                    "graphs": "networkx",
                    "complexity": "plotly"
                }
            }
        
        # Database Systems
        elif any(keyword in subject_lower for keyword in ["database", "sql", "dbms", "data"]):
            return {
                "primary": "graphviz",
                "secondary": ["plotly", "networkx"],
                "forbidden": ["matplotlib", "seaborn"],
                "use_cases": {
                    "er diagrams": "graphviz",
                    "schema": "graphviz",
                    "performance": "plotly",
                    "relationships": "networkx"
                }
            }
        
        # Mathematics
        elif any(keyword in subject_lower for keyword in ["mathematics", "math", "calculus", "algebra"]):
            return {
                "primary": "plotly",
                "secondary": ["seaborn"],
                "forbidden": ["matplotlib", "networkx", "graphviz"],
                "use_cases": {
                    "functions": "plotly",
                    "graphs": "plotly",
                    "statistics": "seaborn",
                    "distributions": "seaborn"
                }
            }
        
        # Default configuration for any other subject
        else:
            return {
                "primary": "plotly",
                "secondary": ["networkx", "graphviz"],
                "forbidden": ["matplotlib"],
                "use_cases": {
                    "general diagrams": "plotly",
                    "relationships": "networkx",
                    "flowcharts": "graphviz"
                }
            }
    
    def extract_topics_from_content(self, content: str) -> List[str]:
        """Extract meaningful topics from content using AI and patterns"""
        topics = []
        
        # First try AI-based topic extraction
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
            
            # Split content into chunks for processing
            content_chunks = [content[i:i+3000] for i in range(0, len(content), 3000)]
            
            for chunk in content_chunks[:5]:  # Process first 5 chunks
                topic_prompt = f"""
                Extract meaningful academic topics from this content. Return ONLY a JSON list of topic names.
                Focus on:
                - Chapter/section titles
                - Key concepts and algorithms
                - Important techniques and methods
                - Theoretical frameworks
                
                Ignore:
                - Page numbers, references, incomplete sentences
                - Mathematical formulas without context
                - Random text fragments
                
                Content:
                {chunk}
                
                Return only a JSON array like: ["Topic 1", "Topic 2", "Topic 3"]
                """
                
                response = llm.invoke(topic_prompt)
                try:
                    chunk_topics = json.loads(response.content.strip())
                    if isinstance(chunk_topics, list):
                        topics.extend([t for t in chunk_topics if len(t) > 10 and len(t) < 80])
                except:
                    pass
        except Exception as e:
            print(f"⚠️ AI topic extraction failed: {e}")
        
        # Fallback to pattern-based extraction with better patterns
        if len(topics) < 10:
            patterns = [
                r"(?:module|chapter|unit|section)\s+\d+[:\-\s]*(.+?)(?:\n|$)",
                r"^\d+\.\s*([A-Z][^:\n]{10,80})(?:\n|$)",
                r"^\d+\.\d+\s*([A-Z][^:\n]{10,80})(?:\n|$)",
                r"^([A-Z][A-Za-z\s]{15,80})(?:\n|$)",
                r"(?:introduction to|overview of|fundamentals of|principles of)\s+(.+?)(?:\n|$)"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    topic = match.strip().strip('.').strip(':')
                    # Better filtering
                    if (len(topic) > 10 and len(topic) < 80 and 
                        not re.match(r'^\d+', topic) and
                        not any(bad in topic.lower() for bad in ['page', 'figure', 'table', 'references', 'bibliography'])):
                        topics.append(topic)
        
        # Clean and deduplicate
        topics = list(set(topics))
        topics = [t for t in topics if len(t.split()) >= 2]  # At least 2 words
        
        # If still not enough topics, add some subject-specific defaults
        if len(topics) < 20:
            subject_defaults = {
                "artificial intelligence": [
                    "Search Algorithms and Heuristics",
                    "Knowledge Representation and Reasoning", 
                    "Machine Learning Fundamentals",
                    "Neural Networks and Deep Learning",
                    "Expert Systems and Knowledge Engineering",
                    "Natural Language Processing",
                    "Computer Vision and Image Processing",
                    "Planning and Decision Making",
                    "Uncertainty and Probabilistic Reasoning",
                    "Game Playing and Adversarial Search"
                ]
            }
            
            content_lower = content.lower()
            for subject, default_topics in subject_defaults.items():
                if subject in content_lower:
                    topics.extend(default_topics)
                    break
        
        return topics[:50]  # Limit to 50 topics
    
    def create_gpt_instructions(self, subject_type: str, topics: List[str]) -> Dict[str, Any]:
        """Create comprehensive GPT instructions based on content analysis"""
        
        # Dynamic library selection based on subject type
        lib_config = self._determine_library_config(subject_type)
        
        instructions = {
            "subject_type": subject_type,
            "available_topics": topics,
            "library_configuration": lib_config,
            "diagram_generation_rules": {
                "mandatory_library_usage": f"MUST use {lib_config['primary']} as primary library",
                "forbidden_libraries": f"NEVER use: {', '.join(lib_config['forbidden'])}",
                "library_selection_logic": lib_config["use_cases"],
                "code_requirements": [
                    "Generate ONLY executable Python code",
                    "Use unique timestamps in filenames",
                    "Include proper error handling",
                    "Add clear comments explaining the diagram",
                    "Ensure diagram is relevant to the question content"
                ]
            },
            "question_generation_guidelines": {
                "content_source": "Use ONLY the provided topics and content",
                "difficulty_levels": {
                    "2_marks": "Definition, short explanation, simple examples",
                    "5_marks": "Detailed explanation, comparison, analysis",
                    "10_marks": "Comprehensive discussion, evaluation, design"
                },
                "question_types": {
                    "conceptual": "Explain, define, describe",
                    "analytical": "Compare, analyze, evaluate", 
                    "practical": "Design, implement, solve"
                }
            },
            "strict_requirements": {
                "content_adherence": "Questions MUST be based on provided content only",
                "library_enforcement": f"Diagrams MUST use {lib_config['primary']} or {lib_config['secondary']}",
                "no_generic_content": "NO generic or fallback content allowed",
                "filename_uniqueness": "Every diagram must have unique filename with timestamp"
            }
        }
        
        return instructions
    
    def process_all_documents(self) -> Dict[str, Any]:
        """Process all uploaded documents and create comprehensive JSON"""
        
        # File paths
        files = {
            "syllabus": os.path.join(self.upload_folder, "syllabus.pdf"),
            "notes": os.path.join(self.upload_folder, "notes.pdf"),
            "pyqs": os.path.join(self.upload_folder, "pyqs.pdf")
        }
        
        # Extract content from all files
        content_data = {}
        all_content = ""
        
        for file_type, file_path in files.items():
            if os.path.exists(file_path):
                print(f"📄 Processing {file_type}: {file_path}")
                content = self.extract_text_from_pdf(file_path)
                content_data[file_type] = {
                    "path": file_path,
                    "content": content,
                    "length": len(content),
                    "topics": self.extract_topics_from_content(content)
                }
                all_content += content + "\n"
            else:
                print(f"⚠️  File not found: {file_path}")
                content_data[file_type] = {"error": "File not found"}
        
        # Analyze combined content
        subject_type = self.analyze_content_type(all_content)
        all_topics = []
        for file_data in content_data.values():
            if "topics" in file_data:
                all_topics.extend(file_data["topics"])
        
        # Remove duplicates and limit
        unique_topics = list(set(all_topics))[:50]
        
        # Create GPT instructions
        gpt_instructions = self.create_gpt_instructions(subject_type, unique_topics)
        
        # Compile final processed data
        processed_data = {
            "metadata": {
                "processed_at": datetime.now().isoformat(),
                "subject_type": subject_type,
                "total_topics": len(unique_topics),
                "total_content_length": len(all_content)
            },
            "documents": content_data,
            "extracted_topics": unique_topics,
            "gpt_instructions": gpt_instructions,
            "content_summary": {
                "primary_subject": subject_type,
                "key_topics": unique_topics[:20],  # Top 20 topics
                "recommended_libraries": gpt_instructions["library_configuration"]["primary"],
                "forbidden_libraries": gpt_instructions["library_configuration"]["forbidden"]
            }
        }
        
        self.processed_data = processed_data
        return processed_data
    
    def save_processed_data(self, output_path: str = "processed_documents.json") -> bool:
        """Save processed data to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.processed_data, f, indent=2, ensure_ascii=False)
            print(f"✅ Processed data saved to: {output_path}")
            return True
        except Exception as e:
            print(f"❌ Error saving processed data: {e}")
            return False
    
    def load_processed_data(self, input_path: str = "processed_documents.json") -> Dict[str, Any]:
        """Load processed data from JSON file"""
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                self.processed_data = json.load(f)
            print(f"✅ Processed data loaded from: {input_path}")
            return self.processed_data
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
            return {}

if __name__ == "__main__":
    # Test the document processor
    processor = DocumentProcessor()
    processed_data = processor.process_all_documents()
    processor.save_processed_data()
    
    print("\n📊 Processing Summary:")
    print(f"Subject: {processed_data['metadata']['subject_type']}")
    print(f"Topics found: {processed_data['metadata']['total_topics']}")
    print(f"Recommended library: {processed_data['content_summary']['recommended_libraries']}") 