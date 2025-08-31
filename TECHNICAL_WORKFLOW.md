# LLMQ Technical Workflow - Developer Handover Guide

## 🔧 **How The System Works**

### **Core Architecture Flow**
```
PDF Upload → Document Processing → Vector Embeddings → Question Generation → Diagram Generation → PDF Export
```

### **1. Document Processing Pipeline**
```python
# Entry Point: app.py @upload_files()
uploads/ (PDF files) → document_processor.py → processed_documents.json
```

**Key Files:**
- `document_processor.py`: Extracts text from PDFs, identifies topics
- `analyze_pyq_for_config.py`: Analyzes past questions to create exam structure
- `vector_cache/`: Stores FAISS vector embeddings for fast retrieval

**Process:**
1. Extract text from syllabus, notes, PYQs using PyPDF2
2. Create vector embeddings using OpenAI embeddings
3. Store in FAISS vector database with caching
4. Generate exam configuration based on PYQ analysis

### **2. Question Generation (RAG System)**
```python
# Core Engine: agentic_rag_exam.py
Vector Store → Similarity Search → Context Retrieval → LLM Generation → Questions
```

**Key Components:**
- **Vector Retrieval**: Uses FAISS for semantic similarity search
- **Context Assembly**: Combines relevant document chunks
- **LLM Prompting**: GPT-4 generates contextual questions
- **Multi-section Support**: Short Answer, Long Answer, Problem sections

**Critical Function:**
```python
def generate_exam(self):
    # 1. Load vector stores (syllabus, notes, pyqs)
    # 2. Analyze topic frequencies
    # 3. Generate questions per section
    # 4. Determine diagram needs using AI
    # 5. Return structured question list
```

### **3. Diagram Generation System**
```python
# Enhanced System: enhanced_diagram_generator.py + diagram_executor.py
Question Text → Library Selection → Code Generation → Safe Execution → Base64 Image
```

**Library Selection Logic:**
```python
ENHANCED_LIBRARIES = {
    'matplotlib': 'Statistical plots, mathematical functions',
    'seaborn': 'Advanced statistical visualizations', 
    'plotly': 'Interactive 3D visualizations',
    'networkx': 'Graph theory, network analysis',
    'graphviz': 'Hierarchical diagrams, flowcharts',
    'pyvis': 'Interactive network visualizations',
    'mermaid': 'Process diagrams, flowcharts',
    # ... 4 more libraries
}
```

**Execution Flow:**
1. **AI Decision**: Determine if diagram needed
2. **Library Selection**: Choose optimal library based on content
3. **Code Generation**: LLM creates Python visualization code
4. **Safe Execution**: Subprocess with 30s timeout
5. **Fallback System**: Multiple backup mechanisms
6. **Base64 Conversion**: Embed in question JSON

### **4. Template System & PDF Export**
```python
# Templates: templates/display_exam.html, templates/question_paper.html
Question JSON → Jinja2 Template → HTML → PDF Export
```

**Key Fix Applied:**
- **Question Numbering**: Uses `namespace` objects to fix Jinja2 scoping
- **Diagram Embedding**: Base64 images embedded directly in HTML
- **PDF Generation**: `exam_to_pdf.py` converts HTML to professional PDF

## 🚨 **Current Issues & Limitations**

### **1. Duplicate File Generation** ❌
**Problem**: Multiple diagram files created with different timestamps
**Location**: `static/diagrams/` folder
**Cause**: Filename generation happens in multiple places
**Impact**: Storage waste, confusion

**Example**: 
```
set_theory_operations_1751184200.png  # Generated in enhanced_diagram_generator
set_theory_operations_1751184203.png  # Generated during execution
```

### **2. Single Subject Limitation** ❌
**Problem**: Optimized only for Computer Science/AI topics
**Location**: `enhanced_diagram_generator.py` library selection logic
**Impact**: Limited market applicability

### **3. Performance Bottleneck** ⚠️
**Problem**: 2-3 minutes processing time for 6 questions
**Cause**: Sequential diagram generation, no parallel processing
**Impact**: Poor user experience

### **4. Basic Web Interface** ⚠️
**Problem**: Simple HTML/CSS interface, no real-time feedback
**Location**: `templates/index.html`
**Impact**: Limited user experience

### **5. No Multi-user Support** ⚠️
**Problem**: Single-user application, shared directories
**Impact**: Cannot scale to multiple concurrent users

## 🔧 **Required Improvements - Priority Order**

### **HIGH PRIORITY** (Fix Immediately)

#### **1. Fix Duplicate File Generation** (4-6 hours)
**Problem**: Files like `topic_1751184200.png` and `topic_1751184203.png` created

**Root Cause:**
```python
# In app.py - filename created here
unique_filename = enhanced_diagram_gen.create_unique_filename(topic)

# In enhanced_diagram_generator.py - filename created again
def generate_enhanced_diagram_code(self, question, topic, filename=None):
    unique_filename = filename if filename else self.create_unique_filename(topic)
```

**Solution:**
```python
# Modify app.py to pass filename consistently
diagram_code = enhanced_diagram_gen.generate_enhanced_diagram_code(
    question_text, topic, filename=unique_filename  # Pass pre-generated filename
)

# Remove duplicate filename generation in enhanced_diagram_generator.py
def generate_enhanced_diagram_code(self, question, topic, filename):
    # Use provided filename, don't generate new one
```

**Files to Modify:**
- `app.py`: Lines 220-250 (diagram generation loop)
- `enhanced_diagram_generator.py`: Line 135 (remove duplicate filename creation)

#### **2. Multi-Subject Library Selection** (12-15 hours)
**Problem**: Library selection hardcoded for CS topics

**Current Logic:**
```python
def get_best_library_for_content(self, question, topic):
    # Only CS-focused keywords
    if 'neural network' in topic.lower():
        return 'pyvis'
    elif 'algorithm' in topic.lower():
        return 'mermaid'
```

**Solution:**
```python
SUBJECT_LIBRARY_MAPPING = {
    'computer_science': {
        'neural_network': ['pyvis', 'networkx'],
        'algorithm': ['mermaid', 'graphviz'],
        'database': ['graphviz', 'matplotlib']
    },
    'mathematics': {
        'geometry': ['matplotlib', 'plotly'],
        'statistics': ['seaborn', 'plotly'],
        'calculus': ['matplotlib', 'plotly']
    },
    'physics': {
        'mechanics': ['matplotlib', 'plotly'],
        'waves': ['matplotlib', 'seaborn'],
        'circuits': ['schemdraw', 'matplotlib']
    },
    'chemistry': {
        'organic': ['rdkit', 'matplotlib'],
        'physical': ['plotly', 'seaborn']
    }
}

def detect_subject_from_content(self, documents):
    # Analyze document content to determine subject
    # Return subject classification
    
def get_best_library_for_content(self, question, topic):
    subject = self.detect_subject_from_content(self.processed_data)
    topic_keywords = self.extract_topic_keywords(topic)
    return self.select_optimal_library(subject, topic_keywords)
```

**Files to Modify:**
- `enhanced_diagram_generator.py`: Add subject detection logic
- `document_processor.py`: Add subject classification
- Create `subject_mappings.py`: Subject-specific configurations

### **MEDIUM PRIORITY** (Next Phase)

#### **3. Performance Optimization** (10-12 hours)
**Problem**: Sequential processing, no parallelization

**Current Flow:**
```python
# Sequential processing in app.py
for question in generated_questions:
    if needs_diagram:
        diagram_code = generate_diagram()  # Blocking call
        execute_diagram()  # Another blocking call
```

**Solution:**
```python
import concurrent.futures
import asyncio

async def generate_diagrams_parallel(questions):
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        tasks = []
        for question in questions:
            if question.needs_diagram:
                task = executor.submit(process_diagram, question)
                tasks.append(task)
        
        completed = await asyncio.gather(*tasks)
        return completed

def process_diagram(question):
    # Combined generation + execution in single thread
    diagram_code = generate_diagram_code(question)
    base64_image = execute_diagram_code(diagram_code)
    return base64_image
```

**Files to Modify:**
- `app.py`: Add parallel processing in upload_files()
- `enhanced_diagram_generator.py`: Make thread-safe
- `diagram_executor.py`: Add batch processing method

#### **4. Modern Web Interface** (20-25 hours)
**Problem**: Basic HTML interface, no real-time updates

**Current**: Static HTML forms
**Needed**: React.js with WebSocket real-time updates

**Implementation Plan:**
```javascript
// New frontend structure
frontend/
├── src/
│   ├── components/
│   │   ├── FileUpload.jsx
│   │   ├── ProgressTracker.jsx
│   │   └── ExamViewer.jsx
│   ├── hooks/
│   │   └── useWebSocket.js
│   └── services/
│       └── apiService.js
```

**Backend Changes:**
```python
# Add to app.py
from flask_socketio import SocketIO, emit

socketio = SocketIO(app)

@socketio.on('start_exam_generation')
def handle_exam_generation(data):
    emit('progress', {'stage': 'processing_documents', 'percent': 10})
    # ... emit progress updates throughout generation
```

**Files to Create:**
- `frontend/` directory with React app
- Add WebSocket support to `app.py`
- Create API endpoints for frontend communication

### **LOW PRIORITY** (Future Releases)

#### **5. Multi-user Support** (15-20 hours)
**Problem**: Shared directories, no session isolation

**Solution:**
```python
# User session management
class UserSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.upload_dir = f"uploads/{session_id}"
        self.cache_dir = f"vector_cache/{session_id}"
        self.diagrams_dir = f"static/diagrams/{session_id}"
    
    def cleanup_expired_sessions(self):
        # Remove sessions older than 24 hours
```

#### **6. Enhanced Error Logging** (8-10 hours)
**Current**: Print statements
**Needed**: Professional logging system

**Solution:**
```python
import logging
from datetime import datetime

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llmq.log'),
        logging.StreamHandler()
    ]
)

class ErrorTracker:
    def log_diagram_error(self, question, library, error):
        logging.error(f"Diagram generation failed: {question[:50]}... using {library}: {error}")
```

## 🛠️ **Development Environment Setup**

### **1. Clone and Setup** (30 minutes)
```bash
git clone <repository>
cd llmq
pip install -r requirements.txt
python install_optional_libraries.py
```

### **2. Environment Configuration**
```bash
# Create .env file
OPENAI_API_KEY=your_key_here

# Install Graphviz (Windows)
# Download: https://graphviz.org/download/
# Add to PATH: C:\Program Files\Graphviz\bin\
```

### **3. Test System**
```bash
python app.py
# Upload test PDFs
# Verify diagram generation works
```

## 📁 **Key Files to Understand**

### **Core Files** (Must understand first)
1. `app.py` - Main Flask application, request handling
2. `agentic_rag_exam.py` - RAG engine, question generation
3. `enhanced_diagram_generator.py` - Diagram code generation
4. `diagram_executor.py` - Safe code execution

### **Supporting Files**
5. `document_processor.py` - PDF processing
6. `analyze_pyq_for_config.py` - Exam configuration
7. `templates/display_exam.html` - Question display template

### **Configuration Files**
8. `requirements.txt` - Python dependencies
9. `processed_documents.json` - Document analysis results
10. `exam_config.json` - Generated exam structure

## 🧪 **Testing Strategy**

### **Unit Tests Needed**
```python
# test_diagram_generation.py
def test_library_selection():
    # Test library selection for different subjects
    
def test_filename_uniqueness():
    # Ensure no duplicate filenames generated
    
def test_fallback_mechanisms():
    # Test all fallback systems work
```

### **Integration Tests Needed**
```python
# test_full_workflow.py
def test_end_to_end_exam_generation():
    # Upload PDFs → Generate exam → Verify output
    
def test_parallel_diagram_generation():
    # Test multiple diagrams generated simultaneously
```

## 🚀 **Next Developer Action Plan**

### **Week 1: Setup & Understanding**
- [ ] Setup development environment
- [ ] Run full exam generation test
- [ ] Study core architecture (4 main files)
- [ ] Identify current issues firsthand

### **Week 2: Fix Critical Issues**
- [ ] Fix duplicate file generation bug
- [ ] Clean up existing duplicate files
- [ ] Test filename consistency
- [ ] Add basic logging

### **Week 3-4: Multi-Subject Support**
- [ ] Add subject detection logic
- [ ] Create subject-specific library mappings
- [ ] Test with non-CS documents
- [ ] Validate diagram relevance

### **Month 2: Performance & UI**
- [ ] Implement parallel processing
- [ ] Start React.js frontend development
- [ ] Add WebSocket real-time updates
- [ ] Performance benchmarking

## 📞 **Getting Help**

### **Code Comments**
- All critical functions have docstrings
- Complex logic is commented inline
- Architecture decisions documented

### **Debugging**
```python
# Enable debug mode
export FLASK_ENV=development
python app.py --debug

# Check logs
tail -f llmq.log
```

### **Common Issues**
1. **Graphviz not found**: Add to PATH
2. **OpenAI API errors**: Check .env file
3. **Memory issues**: Process smaller documents
4. **Import errors**: Run install_optional_libraries.py

---

**TECHNICAL HANDOVER COMPLETE**  
**Next Developer Ready to Begin**  
**Priority: Fix duplicate files → Multi-subject support → Performance** 