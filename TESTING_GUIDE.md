# ChatGPT Diagram Integration Testing Guide

This guide explains how to test the ChatGPT diagram integration feature in the LLMQ exam generation system.

## 🎯 What's Been Implemented

The system now features **ChatGPT-generated contextual diagrams** that:
- Analyze each exam question specifically
- Generate Python code using appropriate libraries (matplotlib, graphviz, networkx)
- Execute the code safely in a sandboxed environment
- Convert diagrams to base64 for web display
- Integrate seamlessly with the exam generation workflow

## 🧪 Available Test Scripts

### 1. Quick Test (`quick_diagram_test.py`)
**Purpose**: Verify basic functionality without requiring OpenAI API
```bash
python quick_diagram_test.py
```

**What it tests**:
- ✅ File structure and imports
- ✅ DiagramExecutor functionality
- ✅ Matplotlib diagram generation
- ✅ NetworkX diagram generation
- ⚠️ Graphviz (may fail if not installed)

### 2. Full Integration Test (`test_diagram_integration.py`)
**Purpose**: Complete end-to-end testing including ChatGPT
```bash
python test_diagram_integration.py
```

**Requirements**:
- OpenAI API key in `.env` file or environment
- All Python dependencies installed

**What it tests**:
- All quick test features
- ChatGPT diagram code generation
- Full integration with exam system
- HTML template integration

### 3. Interactive Demo (`demo_chatgpt_diagrams.py`)
**Purpose**: Live demonstration of ChatGPT generating contextual diagrams
```bash
python demo_chatgpt_diagrams.py
```

**Features**:
- Processes 4 sample exam questions
- Shows ChatGPT generating diagram code in real-time
- Executes and validates diagrams
- Saves results to `demo_results.json`

### 4. Enhanced Library Demo (`demo_enhanced_diagrams.py`)
**Purpose**: Comprehensive test of all supported libraries across 8 domains
```bash
python demo_enhanced_diagrams.py
```

**Features**:
- Tests 8 different academic domains
- Verifies 19+ visualization libraries
- Shows subject-specific diagram generation
- Generates detailed performance report

### 5. Windows Batch Runner (`run_tests.bat`)
**Purpose**: Easy testing on Windows
```cmd
run_tests.bat
```

## 📋 Test Results Interpretation

### Expected Results (Quick Test)
```
File Structure            ✅ PASSED
ChatGPT Integration       ✅ PASSED  
Basic Diagram Executor    ✅ PASSED
Graphviz Diagrams         ❌ FAILED (expected if graphviz not installed)
NetworkX Diagrams         ✅ PASSED
```

### Success Indicators
- **4/5 or 5/5 tests pass** = Integration is working
- **Base64 output generated** = Diagrams can be displayed in web
- **No import errors** = All dependencies are available

## 🔧 Prerequisites

### Required Files
- `agentic_rag_exam.py` - Core RAG agent with new diagram methods
- `diagram_executor.py` - Safe code execution engine
- `app.py` - Flask app with diagram integration
- `exam_config.json` - Exam configuration
- `templates/display_exam.html` - Updated template

### Python Dependencies
```bash
pip install langchain langchain-openai openai faiss-cpu matplotlib networkx flask
```

### Enhanced Diagram Libraries
```bash
# Core visualization libraries
pip install matplotlib numpy scipy pandas seaborn plotly

# Specialized diagram libraries
pip install graphviz networkx diagrams schemdraw

# Advanced libraries for specific domains
pip install bokeh pygraphviz anytree ete3 sympy statsmodels scikit-learn

# Control systems and signal processing
pip install control scapy

# Image export support
pip install kaleido psutil pycairo
```

### Library Coverage by Domain
- **Operating Systems**: matplotlib, seaborn, plotly, networkx
- **Networks/DCCN**: networkx, matplotlib, plotly, bokeh, diagrams
- **Digital Logic**: schemdraw, matplotlib, graphviz, numpy, control
- **Algorithms**: matplotlib, seaborn, networkx, anytree, plotly
- **Circuit Design**: schemdraw, matplotlib, scipy, control, plotly
- **Mathematics**: matplotlib, numpy, scipy, seaborn, sympy
- **Database Systems**: networkx, matplotlib, plotly, pandas
- **Machine Learning**: matplotlib, seaborn, networkx, sklearn

## 🚀 Testing the Full System

### Step 1: Run Quick Tests
```bash
python quick_diagram_test.py
```
**Expected**: 4/5 tests pass (Graphviz may fail)

### Step 2: Set Up OpenAI API
Create `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

### Step 3: Run Demo
```bash
python demo_chatgpt_diagrams.py
```
**Expected**: ChatGPT generates contextual diagram code

### Step 4: Test Web Integration
```bash
python app.py
```
Visit: `http://localhost:5000/test_chatgpt_diagrams`

## 🎨 How It Works

### 1. Question Analysis
ChatGPT receives:
- The exact question text
- Topic and subject context
- Available Python libraries
- Example code patterns

### 2. Code Generation
ChatGPT generates:
```python
import matplotlib.pyplot as plt
import numpy as np

# Create process scheduling timeline
processes = ['P1', 'P2', 'P3', 'P4']
burst_times = [6, 8, 4, 5]
# ... contextual diagram code ...
```

### 3. Safe Execution
- Code runs in isolated subprocess
- 30-second timeout protection
- Error handling and logging
- Base64 conversion for web display

### 4. Web Integration
```html
{% if question.diagram_success %}
<div class="diagram">
    <img src="data:image/png;base64,{{ question.diagram_base64 }}">
</div>
{% endif %}
```

## 🔍 Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install -r requirements.txt
```

**Directory Not Found**
- Ensure `static/diagrams/` exists
- Run from project root directory

**OpenAI API Errors**
- Check API key in `.env` file
- Verify API quota/billing

**Graphviz Errors**
- Install Graphviz system package
- Add to system PATH
- Or skip Graphviz tests (not critical)

### Debug Mode
Add debug prints to see generated code:
```python
print("Generated diagram code:")
print(diagram_data["diagram_code"])
```

## 🎉 Success Criteria

The integration is working correctly if:
1. ✅ Quick tests pass (4/5 minimum)
2. ✅ Demo generates at least 1 diagram successfully  
3. ✅ Web interface displays diagrams
4. ✅ No critical errors in console

## 📞 Support

If tests fail:
1. Check the error messages in console output
2. Verify all dependencies are installed
3. Ensure OpenAI API key is valid
4. Run individual test components to isolate issues

The system is designed to be robust - even if diagram generation fails, the exam generation will continue without diagrams. 