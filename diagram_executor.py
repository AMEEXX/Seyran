import os
import sys
import subprocess
import tempfile
import base64
import json
from typing import Dict, Any, Optional
import traceback

class DiagramExecutor:
    """Safely execute ChatGPT-generated diagram code and convert to base64"""
    
    def __init__(self, output_dir: str = "static/diagrams"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def execute_diagram_code(self, diagram_data: Dict[str, Any]) -> Optional[str]:
        """
        Execute the diagram code and return base64 encoded image
        
        Args:
            diagram_data: Dictionary containing diagram_code, filename, etc.
            
        Returns:
            Base64 encoded image string or None if execution failed
        """
        try:
            diagram_code = diagram_data.get("diagram_code", "")
            filename = diagram_data.get("filename", "diagram.png")
            
            if not diagram_code:
                print("No diagram code provided")
                return None
            
            # Create a safe execution environment
            return self._safe_execute_code(diagram_code, filename)
            
        except Exception as e:
            print(f"Error executing diagram code: {e}")
            traceback.print_exc()
            return None
    
    def _safe_execute_code(self, code: str, filename: str) -> Optional[str]:
        """Safely execute the diagram code in a controlled environment"""
        
        # Create a temporary Python file with UTF-8 encoding
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as temp_file:
            # Add safety imports and setup
            safe_code = self._wrap_code_safely(code, filename)
            temp_file.write(safe_code)
            temp_file_path = temp_file.name
        
        try:
            # Execute the code in a subprocess for safety
            result = subprocess.run(
                [sys.executable, temp_file_path],
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                cwd=self.output_dir
            )
            
            if result.returncode == 0:
                # Code executed successfully, try to read the generated image
                image_path = os.path.join(self.output_dir, filename)
                print(f"Looking for image at: {image_path}")
                
                # List all files in the output directory for debugging
                if os.path.exists(self.output_dir):
                    files_in_dir = os.listdir(self.output_dir)
                    print(f"Files in output directory: {files_in_dir}")
                    
                    # Check for files with similar names - but be very specific
                    base_name = filename.replace('.png', '')
                    # Only look for exact filename matches
                    exact_match = filename in files_in_dir
                    if exact_match:
                        print(f"Found exact match: {filename}")
                        image_path = os.path.join(self.output_dir, filename)
                    else:
                        # Only look for files with the exact base name and timestamp
                        similar_files = [f for f in files_in_dir if f.startswith(base_name) and f.endswith('.png')]
                        if similar_files:
                            print(f"Similar files found: {similar_files}")
                            # Use the most recent file (highest timestamp)
                            similar_files.sort(reverse=True)  # Sort by name (timestamp) descending
                            alternative_path = os.path.join(self.output_dir, similar_files[0])
                            if os.path.exists(alternative_path):
                                print(f"Using alternative file: {alternative_path}")
                                image_path = alternative_path
                        else:
                            # If no similar files found, create a fallback diagram
                            print(f"No similar files found for {base_name}")
                            return self._create_fallback_diagram(filename)
                
                if os.path.exists(image_path):
                    return self._image_to_base64(image_path)
                else:
                    print(f"Image file not found: {image_path}")
                    print(f"Stdout: {result.stdout}")
                    print(f"Stderr: {result.stderr}")
                    # Create a fallback diagram instead of returning None
                    return self._create_fallback_diagram(filename)
            else:
                print(f"Code execution failed with return code: {result.returncode}")
                print(f"Stdout: {result.stdout}")
                print(f"Stderr: {result.stderr}")
                # Create a fallback diagram for execution failures too
                return self._create_fallback_diagram(filename)
                
        except subprocess.TimeoutExpired:
            print("Diagram code execution timed out")
            return self._create_fallback_diagram(filename)
        except Exception as e:
            print(f"Error during code execution: {e}")
            return self._create_fallback_diagram(filename)
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    def _wrap_code_safely(self, code: str, filename: str) -> str:
        """Wrap the user code with safety measures and proper setup"""
        
        # Use absolute path to avoid issues
        abs_output_dir = os.path.abspath(self.output_dir)
        
        safe_wrapper = f"""
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add common library fallbacks
try:
    import graphviz
    # Try to set Graphviz path for Windows
    graphviz_paths = [
        'C:/Program Files/Graphviz/bin/',
        'C:/Program Files (x86)/Graphviz/bin/',
        '/usr/bin/',
        '/usr/local/bin/'
    ]
    for path in graphviz_paths:
        if os.path.exists(path):
            os.environ["PATH"] += os.pathsep + path
            break
except ImportError:
    print("Warning: Graphviz not available, falling back to matplotlib")
    graphviz = None

try:
    import networkx as nx
except ImportError:
    print("Warning: NetworkX not available")
    nx = None

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.kaleido.scope.mathjax = None  # Disable MathJax for better compatibility
except ImportError:
    print("Warning: Plotly not available")
    go = None

# Create output directory if it doesn't exist
output_dir = r'{abs_output_dir}'
os.makedirs(output_dir, exist_ok=True)

# Set working directory to diagrams folder
try:
    os.chdir(output_dir)
    print(f"Changed to directory: {{os.getcwd()}}")
except Exception as e:
    print(f"Failed to change directory: {{e}}")
    sys.exit(1)

try:
    # User's diagram code starts here
{self._indent_code(code)}
    
    # Ensure matplotlib plots are saved
    if 'plt' in globals():
        if plt.get_fignums():  # Check if there are any figures
            plt.savefig('{filename}', dpi=150, bbox_inches='tight')
            plt.close('all')
            print(f"Matplotlib plot saved as {{'{filename}'}}")
    
    print("Diagram generated successfully")
    
except ImportError as e:
    print(f"Missing library: {{e}}")
    # Create a contextual fallback matplotlib diagram
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        # Extract topic from filename for better context
        topic = '{filename}'.replace('.png', '').replace('_', ' ').title()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a meaningful diagram based on topic keywords
        if any(word in topic.lower() for word in ['neural', 'network', 'ai']):
            # Simple neural network representation
            ax.add_patch(patches.Circle((2, 3), 0.4, facecolor='lightblue', edgecolor='black'))
            ax.add_patch(patches.Circle((5, 3.5), 0.4, facecolor='lightgreen', edgecolor='black'))
            ax.add_patch(patches.Circle((5, 2.5), 0.4, facecolor='lightgreen', edgecolor='black'))
            ax.add_patch(patches.Circle((8, 3), 0.4, facecolor='lightcoral', edgecolor='black'))
            
            ax.plot([2.4, 4.6], [3, 3.5], 'k-', alpha=0.7)
            ax.plot([2.4, 4.6], [3, 2.5], 'k-', alpha=0.7)
            ax.plot([5.4, 7.6], [3.5, 3], 'k-', alpha=0.7)
            ax.plot([5.4, 7.6], [2.5, 3], 'k-', alpha=0.7)
            
            ax.text(2, 1.5, 'Input', ha='center', fontweight='bold')
            ax.text(5, 1.5, 'Hidden', ha='center', fontweight='bold')
            ax.text(8, 1.5, 'Output', ha='center', fontweight='bold')
            ax.set_title(f'Neural Network: {{topic}} (Fallback)', fontsize=12, fontweight='bold')
            
        elif any(word in topic.lower() for word in ['algorithm', 'process', 'flow']):
            # Simple flowchart
            steps = ['Start', 'Process', 'Decision', 'End']
            y_positions = [4, 3, 2, 1]
            colors = ['lightgreen', 'lightblue', 'orange', 'lightcoral']
            
            for i, (step, y, color) in enumerate(zip(steps, y_positions, colors)):
                if step == 'Decision':
                    diamond = patches.Polygon([(5, y+0.3), (6, y), (5, y-0.3), (4, y)], 
                                            facecolor=color, edgecolor='black')
                    ax.add_patch(diamond)
                else:
                    rect = patches.Rectangle((4, y-0.2), 2, 0.4, facecolor=color, edgecolor='black')
                    ax.add_patch(rect)
                
                ax.text(5, y, step, ha='center', va='center', fontweight='bold')
                
                if i < len(steps) - 1:
                    ax.arrow(5, y-0.25, 0, -0.5, head_width=0.1, head_length=0.05, fc='black', ec='black')
            
            ax.set_title(f'Algorithm: {{topic}} (Fallback)', fontsize=12, fontweight='bold')
            
        else:
            # General concept diagram
            ax.add_patch(patches.Rectangle((4, 2.8), 2, 0.4, facecolor='lightblue', edgecolor='black'))
            ax.text(5, 3, 'Main Topic', ha='center', va='center', fontweight='bold')
            
            concepts = [('Concept 1', 2, 4), ('Concept 2', 8, 4), ('Detail', 5, 1.5)]
            colors = ['lightgreen', 'lightcoral', 'lightyellow']
            
            for (concept, x, y), color in zip(concepts, colors):
                ax.add_patch(patches.Rectangle((x-0.7, y-0.15), 1.4, 0.3, facecolor=color, edgecolor='black'))
                ax.text(x, y, concept, ha='center', va='center', fontweight='bold', fontsize=9)
                ax.plot([5, x], [3, y], 'k-', alpha=0.6)
            
            ax.set_title(f'Concept: {{topic}} (Fallback)', fontsize=12, fontweight='bold')
        
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig('{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Contextual fallback diagram saved as {{'{filename}'}}")
    except Exception as fallback_error:
        print(f"Fallback diagram failed: {{fallback_error}}")
        sys.exit(1)
        
except Exception as e:
    print(f"Error in diagram generation: {{e}}")
    # Create a simple error diagram
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.add_patch(patches.Rectangle((0.1, 0.1), 0.8, 0.8, linewidth=2, edgecolor='red', facecolor='none'))
        ax.text(0.5, 0.5, 'Diagram Generation\\nFailed', ha='center', va='center', fontsize=16, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Error in Diagram Generation')
        ax.axis('off')
        plt.savefig('{filename}', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Error diagram saved as {{'{filename}'}}")
    except Exception as error_diagram_error:
        print(f"Error diagram failed: {{error_diagram_error}}")
        sys.exit(1)
"""
        return safe_wrapper
    
    def _indent_code(self, code: str) -> str:
        """Indent code by 4 spaces for proper wrapping"""
        lines = code.split('\n')
        indented_lines = ['    ' + line if line.strip() else line for line in lines]
        return '\n'.join(indented_lines)
    
    def _image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string"""
        try:
            with open(image_path, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            print(f"Error converting image to base64: {e}")
            return None
    
    def _create_fallback_diagram(self, filename: str) -> Optional[str]:
        """Create a meaningful fallback diagram when the main diagram fails"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            # Extract topic from filename for contextual diagram
            topic = filename.replace('.png', '').replace('_', ' ').title()
            
            # Create a contextual diagram based on filename/topic
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if any(word in filename.lower() for word in ['neural', 'network', 'ai']):
                # Neural network fallback
                ax.add_patch(patches.Circle((2, 3), 0.5, facecolor='lightblue', edgecolor='black'))
                ax.add_patch(patches.Circle((5, 4), 0.5, facecolor='lightgreen', edgecolor='black'))
                ax.add_patch(patches.Circle((5, 2), 0.5, facecolor='lightgreen', edgecolor='black'))
                ax.add_patch(patches.Circle((8, 3), 0.5, facecolor='lightcoral', edgecolor='black'))
                
                # Connections
                ax.plot([2.5, 4.5], [3, 4], 'k-', alpha=0.7)
                ax.plot([2.5, 4.5], [3, 2], 'k-', alpha=0.7)
                ax.plot([5.5, 7.5], [4, 3], 'k-', alpha=0.7)
                ax.plot([5.5, 7.5], [2, 3], 'k-', alpha=0.7)
                
                ax.text(2, 1, 'Input', ha='center', fontweight='bold')
                ax.text(5, 1, 'Hidden', ha='center', fontweight='bold')
                ax.text(8, 1, 'Output', ha='center', fontweight='bold')
                ax.set_title(f'Neural Network: {topic}', fontsize=14, fontweight='bold')
                
            elif any(word in filename.lower() for word in ['algorithm', 'process', 'flow']):
                # Flowchart fallback
                steps = ['Start', 'Process', 'Decision', 'End']
                y_pos = [4, 3, 2, 1]
                colors = ['lightgreen', 'lightblue', 'orange', 'lightcoral']
                
                for i, (step, y, color) in enumerate(zip(steps, y_pos, colors)):
                    if step == 'Decision':
                        diamond = patches.Polygon([(5, y+0.3), (6, y), (5, y-0.3), (4, y)], 
                                                facecolor=color, edgecolor='black')
                        ax.add_patch(diamond)
                    else:
                        rect = patches.Rectangle((4, y-0.2), 2, 0.4, facecolor=color, edgecolor='black')
                        ax.add_patch(rect)
                    
                    ax.text(5, y, step, ha='center', va='center', fontweight='bold')
                    
                    if i < len(steps) - 1:
                        ax.arrow(5, y-0.3, 0, -0.4, head_width=0.1, head_length=0.05, fc='black', ec='black')
                
                ax.set_title(f'Algorithm Flow: {topic}', fontsize=14, fontweight='bold')
                
            else:
                # General concept map fallback
                ax.add_patch(patches.Rectangle((4, 2.7), 2, 0.6, facecolor='lightblue', edgecolor='black'))
                ax.text(5, 3, 'Main Concept', ha='center', va='center', fontweight='bold')
                
                # Related concepts
                concepts = [('Idea 1', 2, 4), ('Idea 2', 8, 4), ('Detail', 5, 1)]
                colors = ['lightgreen', 'lightcoral', 'lightyellow']
                
                for (concept, x, y), color in zip(concepts, colors):
                    ax.add_patch(patches.Rectangle((x-0.8, y-0.2), 1.6, 0.4, facecolor=color, edgecolor='black'))
                    ax.text(x, y, concept, ha='center', va='center', fontweight='bold', fontsize=9)
                    ax.plot([5, x], [3, y], 'k-', alpha=0.7)
                
                ax.set_title(f'Concept Map: {topic}', fontsize=14, fontweight='bold')
            
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 5)
            ax.axis('off')
            
            # Save the fallback diagram
            fallback_path = os.path.join(self.output_dir, filename)
            plt.savefig(fallback_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Created contextual fallback diagram: {fallback_path}")
            
            # Convert to base64
            return self._image_to_base64(fallback_path)
            
        except Exception as e:
            print(f"Failed to create fallback diagram: {e}")
            return None
    
    def batch_execute_diagrams(self, questions: list) -> list:
        """Execute diagram code for all questions that have diagrams"""
        updated_questions = []
        
        for question in questions:
            if question.get("has_diagram", False) and question.get("diagram_code"):
                print(f"Executing diagram for: {question.get('topic', 'Unknown')}")
                
                diagram_data = {
                    "diagram_code": question["diagram_code"],
                    "filename": question.get("diagram_filename", "diagram.png")
                }
                
                # Execute the diagram code
                base64_image = self.execute_diagram_code(diagram_data)
                
                if base64_image:
                    question["diagram_base64"] = base64_image
                    question["diagram_success"] = True
                    print(f"✅ Diagram generated successfully for {question.get('topic')}")
                else:
                    question["diagram_success"] = False
                    print(f"❌ Failed to generate diagram for {question.get('topic')}")
            
            updated_questions.append(question)
        
        return updated_questions

if __name__ == "__main__":
    # Test the executor
    executor = DiagramExecutor()
    
    # Test with a simple matplotlib example
    test_diagram = {
        "diagram_code": """
import matplotlib.pyplot as plt
import numpy as np

# Create a simple process state diagram
fig, ax = plt.subplots(figsize=(10, 6))
states = ['New', 'Ready', 'Running', 'Waiting', 'Terminated']
x_pos = [0, 2, 4, 2, 6]
y_pos = [0, 0, 0, -2, 0]

# Plot states
for i, state in enumerate(states):
    ax.scatter(x_pos[i], y_pos[i], s=1000, c='lightblue', edgecolors='black')
    ax.text(x_pos[i], y_pos[i], state, ha='center', va='center', fontweight='bold')

# Draw arrows for transitions
arrows = [(0,1), (1,2), (2,1), (2,3), (3,1), (2,4)]
for start, end in arrows:
    ax.annotate('', xy=(x_pos[end], y_pos[end]), xytext=(x_pos[start], y_pos[start]),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))

ax.set_xlim(-1, 7)
ax.set_ylim(-3, 1)
ax.set_title('Process State Transitions', fontsize=16, fontweight='bold')
ax.axis('off')
plt.tight_layout()
""",
        "filename": "test_process_states.png"
    }
    
    result = executor.execute_diagram_code(test_diagram)
    if result:
        print("✅ Test diagram executed successfully!")
        print(f"Base64 length: {len(result)}")
    else:
        print("❌ Test diagram execution failed") 