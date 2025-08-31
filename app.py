from flask import Flask, render_template, request, redirect, url_for, send_file, session, jsonify
import os
import json
from dotenv import load_dotenv
from analyze_pyq_for_config import PYQConfigAnalyzer
from agentic_rag_exam import ExamRAGAgent
import shutil

# Load environment variables from .env file
load_dotenv()
from exam_to_pdf import ExamToPDF
from datetime import datetime
from diagram_executor import DiagramExecutor
from document_processor import DocumentProcessor
from enhanced_diagram_generator import EnhancedDiagramGenerator
import re
import PyPDF2
import socket
import sys

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['GENERATED_EXAMS_FOLDER'] = 'generated_exams'
app.config['VECTOR_CACHE_FOLDER'] = 'vector_cache'
app.config['SECRET_KEY'] = 'your-secret-key'  # Change this to a secure secret key

diagram_executor = DiagramExecutor()

# Keywords that suggest diagram needs
DIAGRAM_KEYWORDS = {
    'Operating Systems': {
        'process_states': ['process state', 'process states', 'state transition', 'process lifecycle'],
        'scheduling': ['scheduling algorithm', 'round robin', 'priority scheduling', 'FCFS']
    },
    'DCCN / Networks': {
        'network_topology': ['network topology', 'network architecture', 'network design', 'OSI model', 'TCP/IP'],
        'protocol': ['protocol', 'handshake', 'connection establishment']
    },
    'Digital Logic / Control': {
        'digital_circuit': ['circuit', 'logic gate', 'flip flop', 'counter', 'register', 'combinational circuit'],
        'state_machine': ['state machine', 'state diagram', 'state transition']
    },
    'Algorithms / Architecture': {
        'algorithm_flowchart': ['algorithm', 'flowchart', 'pseudocode', 'sorting', 'searching', 'tree', 'graph'],
        'architecture': ['system architecture', 'component diagram', 'deployment diagram']
    },
    'Switching & Circuit Design': {
        'switching_circuit': ['switching circuit', 'power circuit', 'voltage', 'current', 'resistor', 'capacitor'],
        'control_system': ['control system', 'feedback', 'transfer function']
    }
}

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def analyze_content_for_diagrams(content):
    """Analyze content to determine which diagrams are needed"""
    needed_diagrams = {}
    content = content.lower()
    
    for subject, diagram_types in DIAGRAM_KEYWORDS.items():
        subject_diagrams = {}
        for diagram_type, keywords in diagram_types.items():
            if any(keyword.lower() in content for keyword in keywords):
                subject_diagrams[diagram_type] = True
        if subject_diagrams:
            needed_diagrams[subject] = subject_diagrams
    
    return needed_diagrams

def determine_subject_type(content):
    """Determine the subject type from content"""
    subject_scores = {subject: 0 for subject in DIAGRAM_KEYWORDS.keys()}
    
    for subject, keywords_dict in DIAGRAM_KEYWORDS.items():
        for keywords in keywords_dict.values():
            for keyword in keywords:
                subject_scores[subject] += len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', content.lower()))
    
    return max(subject_scores.items(), key=lambda x: x[1])[0]

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['GENERATED_EXAMS_FOLDER'], exist_ok=True)
os.makedirs(app.config['VECTOR_CACHE_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        print("🚀 Starting file upload and processing...")
        
        if 'pyqs' not in request.files or 'notes' not in request.files or 'syllabus' not in request.files:
            return render_template('index.html', error="Please upload all three files.")

        pyqs_file = request.files['pyqs']
        notes_file = request.files['notes']
        syllabus_file = request.files['syllabus']

        if pyqs_file.filename == '' or notes_file.filename == '' or syllabus_file.filename == '':
            return render_template('index.html', error="Please select all three files.")
            
        print("✅ File validation passed")
    except Exception as e:
        print(f"Error in file validation: {e}")
        return render_template('index.html', error=f"File validation error: {e}")

    # Define file paths in the upload folder
    pyqs_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pyqs.pdf')
    notes_path = os.path.join(app.config['UPLOAD_FOLDER'], 'notes.pdf')
    syllabus_path = os.path.join(app.config['UPLOAD_FOLDER'], 'syllabus.pdf')

    # Save uploaded files
    try:
        pyqs_file.save(pyqs_path)
        notes_file.save(notes_path)
        syllabus_file.save(syllabus_path)
    except Exception as e:
        return render_template('index.html', error=f"Error saving files: {e}")

    # Step 1: Process all documents and create structured JSON
    try:
        print("📄 Processing uploaded documents...")
        doc_processor = DocumentProcessor(app.config['UPLOAD_FOLDER'])
        processed_data = doc_processor.process_all_documents()
        doc_processor.save_processed_data()
        
        print(f"✅ Processed documents - Subject: {processed_data['metadata']['subject_type']}")
        print(f"📊 Found {processed_data['metadata']['total_topics']} topics")
        
    except Exception as e:
        return render_template('index.html', error=f"Error processing documents: {e}")

    # Step 2: Analyze PYQs and generate config
    try:
        config_analyzer = PYQConfigAnalyzer(
            pyq_path=pyqs_path,
            config_output_path='exam_config.json',
            vector_cache_dir=app.config['VECTOR_CACHE_FOLDER']
        )
        exam_config = config_analyzer.generate_exam_config()

        if not exam_config:
            return render_template('index.html', error="Failed to generate exam configuration.")
            
        if not config_analyzer.save_exam_config(exam_config):
            return render_template('index.html', error="Failed to save exam configuration.")
    except Exception as e:
        return render_template('index.html', error=f"Error in config analysis: {e}")

    # Step 3: Generate Exam Questions with Simple Diagram System
    try:
        print("🚀 Initializing exam generator...")
        
        # Load processed document data
        processed_data = None
        processed_data_path = 'processed_documents.json'
        if os.path.exists(processed_data_path):
            try:
                with open(processed_data_path, 'r', encoding='utf-8') as f:
                    processed_data = json.load(f)
                print(f"📋 Loaded processed document data - Subject: {processed_data.get('metadata', {}).get('subject_type', 'Unknown')}")
            except Exception as e:
                print(f"⚠️ Failed to load processed document data: {e}")
        
        exam_generator = ExamRAGAgent(
            config_path='exam_config.json',
            syllabus_path=syllabus_path,
            notes_path=notes_path,
            pyq_path=pyqs_path,
            vector_cache_dir=app.config['VECTOR_CACHE_FOLDER'],
            processed_data=processed_data
        )
        
        # 🎯 INTEGRATE ENHANCED DIAGRAM SYSTEM
        print("🔧 Integrating enhanced diagram system with advanced libraries...")
        enhanced_diagram_gen = EnhancedDiagramGenerator(processed_data_path='processed_documents.json')
        
        print("📝 Generating exam questions...")
        generated_questions = exam_generator.generate_exam()

        if not generated_questions:
            print("❌ No questions generated")
            return render_template('index.html', error="Failed to generate exam questions.")
        
        print(f"✅ Generated {len(generated_questions)} questions successfully")
        
        # Add diagrams using simple system (with progress tracking)
        print("📊 Adding diagrams with forced library system...")
        diagram_count = 0
        failed_diagrams = 0
        total_questions = len([q for q in generated_questions if q.get('type') != 'header'])
        
        for i, question in enumerate(generated_questions):
            if question.get('type') != 'header':
                q_text = question.get('question', '')
                topic = question.get('topic', 'Unknown')
                
                print(f"🔍 Processing question {i+1}/{total_questions}: {topic}")
                
                if enhanced_diagram_gen.should_generate_diagram(q_text, topic):
                    try:
                        # Create filename first, then pass it to generation
                        unique_filename = enhanced_diagram_gen.create_unique_filename(topic)
                        # Generate diagram code using the pre-created filename
                        diagram_code = enhanced_diagram_gen.generate_enhanced_diagram_code(q_text, topic, unique_filename)
                        
                        # Store diagram info
                        question['diagram_code'] = diagram_code
                        question['diagram_filename'] = unique_filename
                        question['has_diagram'] = True
                        
                        # Execute diagram immediately to ensure it's ready
                        diagram_data = {
                            "diagram_code": diagram_code,
                            "filename": unique_filename
                        }
                        
                        print(f"🎨 Executing diagram for: {topic}")
                        base64_result = diagram_executor.execute_diagram_code(diagram_data)
                        
                        if base64_result:
                            question['diagram_base64'] = base64_result
                            question['diagram_success'] = True
                            diagram_count += 1
                            print(f"✅ Successfully generated and executed diagram for: {topic}")
                        else:
                            question['diagram_success'] = False
                            print(f"⚠️ Diagram execution failed for: {topic}")
                            failed_diagrams += 1
                            
                    except Exception as e:
                        print(f"⚠️ Diagram generation/execution failed for {topic}: {e}")
                        question['has_diagram'] = False
                        question['diagram_success'] = False
                        failed_diagrams += 1
                else:
                    question['has_diagram'] = False
                    print(f"⏭️ No diagram needed for: {topic}")
        
        diagram_stats = enhanced_diagram_gen.get_library_stats()
        print(f"📈 Generated {diagram_count} diagrams with enhanced library usage: {diagram_stats}")
        if failed_diagrams > 0:
            print(f"⚠️ {failed_diagrams} diagrams failed to generate")
        
    except Exception as e:
        print(f"❌ Error generating questions: {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f"Error generating questions: {e}")

    # Determine subject type from content for legacy compatibility
    try:
        print("🔍 Analyzing content for subject type...")
        pyqs_content = extract_text_from_pdf(pyqs_path)
        notes_content = extract_text_from_pdf(notes_path)
        syllabus_content = extract_text_from_pdf(syllabus_path)
        combined_content = f"{pyqs_content}\n{notes_content}\n{syllabus_content}"
        subject_type = determine_subject_type(combined_content)
        
        # Final diagram statistics (already executed inline above)
        final_diagrams_count = sum(1 for q in generated_questions if q.get('diagram_success', False))
        total_diagram_attempts = sum(1 for q in generated_questions if q.get('has_diagram', False))
        
        print(f"✅ All diagram processing completed.")
        print(f"📊 Total diagrams attempted: {total_diagram_attempts}")
        print(f"📊 Total diagrams successful: {final_diagrams_count}")
        
        if final_diagrams_count == 0 and total_diagram_attempts > 0:
            print("⚠️ No diagrams were successfully generated. Check library installations.")
        elif final_diagrams_count > 0:
            print(f"🎉 Successfully generated {final_diagrams_count} contextual diagrams!")
        
        # Use the questions that already have diagrams executed
        updated_questions = generated_questions
        
    except Exception as e:
        print(f"⚠️ Error in final processing: {e}")
        updated_questions = generated_questions
        subject_type = "General"

    # Verify all diagrams are properly linked before saving
    try:
        print("🔍 Verifying diagram integration...")
        
        # Count questions with diagrams and verify they have base64 data
        questions_with_diagrams = [q for q in updated_questions if q.get('has_diagram', False)]
        questions_with_base64 = [q for q in updated_questions if q.get('diagram_base64')]
        
        print(f"📊 Questions with diagrams: {len(questions_with_diagrams)}")
        print(f"📊 Questions with base64 data: {len(questions_with_base64)}")
        
        # Ensure all questions with diagrams have base64 data
        for question in questions_with_diagrams:
            if not question.get('diagram_base64'):
                print(f"⚠️ Missing base64 for question: {question.get('topic', 'Unknown')}")
                # Try to regenerate the missing diagram
                if question.get('diagram_filename'):
                    diagram_path = os.path.join('static/diagrams', question['diagram_filename'])
                    if os.path.exists(diagram_path):
                        try:
                            with open(diagram_path, 'rb') as img_file:
                                import base64
                                question['diagram_base64'] = base64.b64encode(img_file.read()).decode('utf-8')
                                question['diagram_success'] = True
                                print(f"✅ Recovered base64 for: {question.get('topic')}")
                        except Exception as e:
                            print(f"❌ Failed to recover base64 for {question.get('topic')}: {e}")
        
        # Final verification
        final_questions_with_base64 = [q for q in updated_questions if q.get('diagram_base64')]
        print(f"📊 Final questions with complete diagrams: {len(final_questions_with_base64)}")
        
        # Save exam data with questions and executed diagrams
        exam_data = {
            'questions': updated_questions,
            'subject_type': subject_type,
            'has_contextual_diagrams': len(final_questions_with_base64) > 0,
            'diagram_stats': {
                'total_questions': len([q for q in updated_questions if q.get('type') != 'header']),
                'questions_with_diagrams': len(questions_with_diagrams),
                'successful_diagrams': len(final_questions_with_base64)
            }
        }

        with open(os.path.join(app.config['GENERATED_EXAMS_FOLDER'], 'exam.json'), 'w') as f:
            json.dump(exam_data, f, indent=4)

        print("🎯 Exam generation completed successfully!")
        print(f"📄 Saved exam with {len(final_questions_with_base64)} complete diagrams")
        
        # Redirect to display exam in new tab
        return redirect(url_for('display_exam'))
    except Exception as e:
        print(f"❌ Error saving exam data: {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f"Error saving exam data: {e}")

@app.route('/display_exam')
def display_exam():
    try:
        with open(os.path.join(app.config['GENERATED_EXAMS_FOLDER'], 'exam.json'), 'r') as f:
            exam_data = json.load(f)
        return render_template('display_exam.html', exam_data=exam_data)
    except Exception as e:
        return render_template('index.html', error=f"Error displaying exam: {e}")

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['GENERATED_EXAMS_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found.", 404

@app.route('/test_diagrams')
def test_diagrams():
    """Test the enhanced diagram generator with various topics"""
    try:
        enhanced_diagram_gen = EnhancedDiagramGenerator()
        
        test_topics = [
            ("Process States", "Operating system process lifecycle and state transitions"),
            ("Network Topology", "Computer network architecture and connections"),
            ("Algorithm Flowchart", "Step-by-step algorithm visualization"),
            ("Neural Network", "Artificial intelligence neural network structure"),
            ("Database Schema", "Relational database design and relationships")
        ]
        
        diagrams = {}
        for topic, description in test_topics:
            if enhanced_diagram_gen.should_generate_diagram(description, topic):
                filename = enhanced_diagram_gen.create_unique_filename(topic)
                code = enhanced_diagram_gen.generate_enhanced_diagram_code(description, topic, filename)
                
                # Execute the diagram
                diagram_data = {"diagram_code": code, "filename": filename}
                base64_result = diagram_executor.execute_diagram_code(diagram_data)
                
                if base64_result:
                    diagrams[topic] = {
                        'base64': base64_result,
                        'filename': filename,
                        'description': description
                    }
        
        return render_template('test_diagrams.html', diagrams=diagrams)
    except Exception as e:
        return f"Error testing diagrams: {e}"

@app.route('/test_chatgpt_diagrams')
def test_chatgpt_diagrams():
    """Test route to verify ChatGPT diagram generation"""
    
    # Sample questions for testing
    test_questions = [
        {
            "question": "Explain the process states in an operating system and draw a state transition diagram.",
            "topic": "Process Management",
            "marks": 5,
            "section": "Section A"
        },
        {
            "question": "Design a 4-bit binary counter using D flip-flops and show the circuit diagram.",
            "topic": "Digital Logic Design",
            "marks": 8,
            "section": "Section B"
        }
    ]
    
    # Initialize the RAG agent for diagram generation
    try:
        from agentic_rag_exam import ExamRAGAgent
        agent = ExamRAGAgent(
            config_path='exam_config.json',
            syllabus_path='uploads/syllabus.pdf',
            notes_path='uploads/notes.pdf',
            pyq_path='uploads/pyqs.pdf',
            vector_cache_dir='vector_cache'
        )
        
        # Generate diagram code for each test question
        for question in test_questions:
            subject_type = agent.determine_subject_type_from_topic(question["topic"])
            diagram_data = agent.generate_contextual_diagram_code(
                question["question"], 
                question["topic"], 
                subject_type
            )
            
            # Add diagram data to question
            question.update(diagram_data)
            question["has_diagram"] = True
        
        # Execute the diagram code
        updated_questions = diagram_executor.batch_execute_diagrams(test_questions)
        
        return render_template('display_exam.html', exam_data={
            'questions': updated_questions,
            'subject_type': 'Test',
            'has_contextual_diagrams': True
        })
        
    except Exception as e:
        return f"Error testing ChatGPT diagrams: {e}"

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear all cached data and processed documents"""
    try:
        print("🧹 Clearing cache and processed documents...")
        
        # Clear vector cache directory
        if os.path.exists(app.config['VECTOR_CACHE_FOLDER']):
            shutil.rmtree(app.config['VECTOR_CACHE_FOLDER'])
            os.makedirs(app.config['VECTOR_CACHE_FOLDER'], exist_ok=True)
            print("✅ Vector cache cleared")
        
        # Clear processed documents
        processed_files = ['processed_documents.json', 'exam_config.json']
        for file in processed_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"✅ Removed {file}")
        
        # Clear generated exams
        if os.path.exists(app.config['GENERATED_EXAMS_FOLDER']):
            for file in os.listdir(app.config['GENERATED_EXAMS_FOLDER']):
                file_path = os.path.join(app.config['GENERATED_EXAMS_FOLDER'], file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("✅ Generated exams cleared")
        
        # Clear uploaded files
        if os.path.exists(app.config['UPLOAD_FOLDER']):
            for file in os.listdir(app.config['UPLOAD_FOLDER']):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print("✅ Uploaded files cleared")
        
        # Clear all diagram files from static/diagrams
        diagrams_folder = 'static/diagrams'
        if os.path.exists(diagrams_folder):
            diagram_count = 0
            for file in os.listdir(diagrams_folder):
                if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.svg'):
                    file_path = os.path.join(diagrams_folder, file)
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                            diagram_count += 1
                        except Exception as e:
                            print(f"⚠️ Could not remove {file}: {e}")
            print(f"✅ Cleared {diagram_count} diagram files from {diagrams_folder}")
        
        print("🎉 Cache refresh completed successfully!")
        return jsonify({"success": True, "message": "Cache cleared successfully!"})
        
    except Exception as e:
        print(f"❌ Error clearing cache: {e}")
        return jsonify({"success": False, "error": str(e)})

@app.route('/test_diagram_status')
def test_diagram_status():
    """Test diagram generation capabilities and return status"""
    try:
        # Test enhanced diagram generator
        generator = EnhancedDiagramGenerator()
        executor = DiagramExecutor()
        
        # Test with a simple question
        test_question = "Draw a simple network topology diagram"
        test_topic = "Network Topology"
        
        status = {
            "generator_available": True,
            "executor_available": True,
            "libraries_status": {},
            "test_generation": False,
            "test_execution": False
        }
        
        # Test library availability
        try:
            import matplotlib.pyplot as plt
            status["libraries_status"]["matplotlib"] = "✅ Available"
        except ImportError:
            status["libraries_status"]["matplotlib"] = "❌ Missing"
        
        try:
            import networkx
            status["libraries_status"]["networkx"] = "✅ Available"
        except ImportError:
            status["libraries_status"]["networkx"] = "❌ Missing"
        
        try:
            import graphviz
            status["libraries_status"]["graphviz"] = "✅ Available"
        except ImportError:
            status["libraries_status"]["graphviz"] = "❌ Missing"
        
        try:
            import plotly
            status["libraries_status"]["plotly"] = "✅ Available"
        except ImportError:
            status["libraries_status"]["plotly"] = "❌ Missing"
        
        # Test enhanced libraries
        try:
            import mermaid_py
            status["libraries_status"]["mermaid"] = "✅ Available"
        except ImportError:
            status["libraries_status"]["mermaid"] = "❌ Missing"
        
        try:
            import pyvis
            status["libraries_status"]["pyvis"] = "✅ Available"
        except ImportError:
            status["libraries_status"]["pyvis"] = "❌ Missing"
        
        try:
            import pyecharts
            status["libraries_status"]["pyecharts"] = "✅ Available"
        except ImportError:
            status["libraries_status"]["pyecharts"] = "❌ Missing"
        
        try:
            import altair
            status["libraries_status"]["altair"] = "✅ Available"
        except ImportError:
            status["libraries_status"]["altair"] = "❌ Missing"
        
        try:
            import pydot
            status["libraries_status"]["pydot"] = "✅ Available"
        except ImportError:
            status["libraries_status"]["pydot"] = "❌ Missing"
        
        # Test diagram generation
        try:
            if generator.should_generate_diagram(test_question, test_topic):
                diagram_code = generator.generate_enhanced_diagram_code(test_question, test_topic)
                if diagram_code and "import" in diagram_code:
                    status["test_generation"] = True
        except Exception as e:
            status["test_generation"] = f"Error: {str(e)}"
        
        # Test diagram execution
        try:
            test_diagram = {
                "diagram_code": """
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(8, 6))
x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y, 'b-', linewidth=2)
plt.title('Test Diagram')
plt.savefig('test_diagram_status.png', dpi=150, bbox_inches='tight')
plt.close()
""",
                "filename": "test_diagram_status.png"
            }
            
            result = executor.execute_diagram_code(test_diagram)
            if result:
                status["test_execution"] = True
                # Clean up test file
                test_file = os.path.join("static/diagrams", "test_diagram_status.png")
                if os.path.exists(test_file):
                    os.remove(test_file)
        except Exception as e:
            status["test_execution"] = f"Error: {str(e)}"
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "generator_available": False,
            "executor_available": False
        })

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring and deployment"""
    try:
        from datetime import datetime
        
        # Basic system checks
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "environment": os.environ.get('FLASK_ENV', 'development'),
            "checks": {
                "openai_api_key": "configured" if os.environ.get('OPENAI_API_KEY') else "missing",
                "directories": {}
            }
        }
        
        # Check directory accessibility
        required_dirs = [
            app.config['UPLOAD_FOLDER'],
            app.config['GENERATED_EXAMS_FOLDER'], 
            app.config['VECTOR_CACHE_FOLDER'],
            'static/diagrams'
        ]
        
        for dir_path in required_dirs:
            try:
                os.makedirs(dir_path, exist_ok=True)
                status["checks"]["directories"][dir_path] = "accessible"
            except Exception as e:
                status["checks"]["directories"][dir_path] = f"error: {str(e)}"
                status["status"] = "unhealthy"
        
        # Check import dependencies
        critical_imports = ['openai', 'langchain', 'flask', 'matplotlib']
        status["checks"]["imports"] = {}
        
        for module in critical_imports:
            try:
                __import__(module)
                status["checks"]["imports"][module] = "available"
            except ImportError:
                status["checks"]["imports"][module] = "missing"
                status["status"] = "unhealthy"
        
        return jsonify(status), 200 if status["status"] == "healthy" else 503
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 503

if __name__ == '__main__':
    try:
        # Check for production mode
        production_mode = '--production' in sys.argv or os.environ.get('FLASK_ENV') == 'production'
        
        if production_mode:
            print("\n🚀 LLMQ Production Server Starting...")
            print("📋 Auto-reload disabled for stable exam generation")
            print("🔧 Diagram synchronization fix active")
            print("Please open your browser and go to: http://localhost:8000\n")
            
            # Run without debug mode to prevent auto-reload
            app.run(
                debug=False,           # Disable debug mode
                port=8000, 
                host='localhost', 
                threaded=True,
                use_reloader=False     # Explicitly disable reloader
            )
        else:
            print("\nServer starting...")
            print("Please open your browser and go to: http://localhost:8000\n")
            app.run(debug=True, port=8000, host='localhost', threaded=True)
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Please try running the application again.") 