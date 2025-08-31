#!/usr/bin/env python3
"""
Enhanced Diagram Generator for LLMQ
Uses advanced visualization libraries including Mermaid, PyVis, PyEcharts, etc.
"""

import json
import os
import time
import re
from typing import Dict, Any, Optional
# Removed PromptTemplate and LLMChain imports to avoid curly brace parsing issues
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import from the updated location - prefer langchain_openai
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
    except ImportError:
        from langchain.chat_models import ChatOpenAI

class EnhancedDiagramGenerator:
    def __init__(self, processed_data_path: str = "processed_documents.json"):
        # Use same setup as other files
        self.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
        self.processed_data = self.load_processed_data(processed_data_path)
        self.library_counter = {}  # Track library usage for diversity
        
        # Enhanced library configuration with new libraries
        self.enhanced_libraries = {
            "matplotlib": "Standard plotting and charts",
            "networkx": "Graph and network analysis", 
            "graphviz": "Directed graphs and flowcharts",
            "plotly": "Interactive plots and dashboards",
            "seaborn": "Statistical data visualization",
            "mermaid": "Flowcharts, sequence diagrams, gantt charts",
            "pyvis": "Interactive network visualizations",
            "pyecharts": "Rich interactive charts and maps",
            "altair": "Grammar of graphics statistical visualization",
            "pydot": "DOT language graph generation",
            "bokeh": "Interactive web-ready visualizations"
        }
        
    def load_processed_data(self, path: str) -> Dict[str, Any]:
        """Load processed document data"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading processed data: {e}")
            return {}
    
    def get_best_library_for_content(self, question: str, topic: str) -> str:
        """Intelligently select the best library based on content analysis"""
        
        question_lower = question.lower()
        topic_lower = topic.lower()
        combined_text = f"{question_lower} {topic_lower}"
        
        # PRIORITY-BASED content analysis (more specific matches first)
        
        # 1. MERMAID - Flowcharts, sequences, gantt (highest priority for process diagrams)
        if any(word in combined_text for word in ['flowchart', 'sequence', 'gantt', 'workflow', 'lifecycle']):
            return "mermaid"
        
        # 2. PYECHARTS - Maps and geographical (very specific)
        if any(word in combined_text for word in ['world map', 'geographical', 'geographic', 'location map', 'world']):
            return "pyecharts"
        
        # 3. ALTAIR - Grammar of graphics (specific keywords)
        if any(word in combined_text for word in ['grammar', 'layered', 'faceted']):
            return "altair"
        
        # 4. PYDOT - DOT language (very specific)
        if any(word in combined_text for word in ['dot language', 'dot notation', 'simple graph using dot']):
            return "pydot"
        
        # 5. BOKEH - Web applications (specific web terms)
        if any(word in combined_text for word in ['web application', 'browser', 'real-time']):
            return "bokeh"
        
        # 6. NETWORKX - Network analysis (specific analysis terms)
        if any(word in combined_text for word in ['network centrality', 'graph analysis', 'network properties', 'centrality']):
            return "networkx"
        
        # 7. SEABORN - Statistical analysis (specific stats terms)
        if any(word in combined_text for word in ['statistical', 'correlation', 'heatmap', 'regression', 'distribution']):
            return "seaborn"
        
        # 8. PLOTLY - 3D and dashboards (specific interactive terms)
        if any(word in combined_text for word in ['3d', 'dashboard', 'dynamic']):
            return "plotly"
        
        # 9. GRAPHVIZ - State machines and directed graphs (specific state terms)
        if any(word in combined_text for word in ['state machine', 'finite automaton', 'decision tree', 'directed', 'transition']):
            return "graphviz"
        
        # 10. MATPLOTLIB - Standard charts (basic chart terms)
        if any(word in combined_text for word in ['standard', 'chart', 'bar chart', 'line chart', 'comparison']):
            return "matplotlib"
        
        # 11. PYVIS - Interactive networks (general network terms, lowest priority)
        if any(word in combined_text for word in ['interactive', 'network', 'topology', 'social', 'node', 'edge']):
            return "pyvis"
        
        # Fallback: Ensure library diversity by selecting least used library
        available_libs = list(self.enhanced_libraries.keys())
        least_used_lib = min(
            available_libs,
            key=lambda lib: self.library_counter.get(lib, 0)
        )
        return least_used_lib
    
    def create_unique_filename(self, topic: str) -> str:
        """Create unique filename with timestamp"""
        timestamp = str(int(time.time()))
        safe_topic = re.sub(r'[^a-zA-Z0-9_]', '_', topic.lower())
        return f"{safe_topic}_{timestamp}.png"
    
    def generate_enhanced_diagram_code(self, question: str, topic: str, filename: str = None) -> str:
        """Generate diagram code using advanced libraries"""
        
        # Sanitize inputs
        question = self._sanitize_text(question)
        topic = self._sanitize_text(topic)
        
        # Get best library for this content
        chosen_library = self.get_best_library_for_content(question, topic)
        self.library_counter[chosen_library] = self.library_counter.get(chosen_library, 0) + 1
        
        # Use provided filename or create unique one
        unique_filename = filename if filename else self.create_unique_filename(topic)
        
        # Get subject type
        if self.processed_data:
            gpt_instructions = self.processed_data.get("gpt_instructions", {})
            subject_type = gpt_instructions.get("subject_type", "General")
        else:
            subject_type = "General"
            print("⚠️ No processed data available, using default settings")
        
        print(f"🎯 USING ENHANCED library: {chosen_library} for '{topic}'")
        print(f"📁 Unique filename: {unique_filename}")
        
        # Create library-specific templates
        templates = self._get_library_templates(chosen_library, unique_filename, topic, question)
        
        # Sanitize all text to remove problematic characters that could break the prompt
        question_clean = self._sanitize_text(question).replace('{', '').replace('}', '')
        topic_clean = self._sanitize_text(topic).replace('{', '').replace('}', '')
        subject_clean = self._sanitize_text(subject_type).replace('{', '').replace('}', '')
        
        # Create enhanced prompt with sanitized inputs
        diagram_prompt = f"""
You are an advanced Python visualization expert. Generate ONLY executable Python code that creates a SPECIFIC, high-quality diagram.

MANDATORY REQUIREMENTS:
1. Use ONLY this library: {chosen_library}
2. Save file as: {unique_filename}
3. Generate working Python code only
4. NO explanations, NO markdown, NO comments except in code
5. Use only ASCII characters in code comments and strings
6. Create a diagram that SPECIFICALLY addresses the question content
7. Make the diagram professional, educational, and visually appealing

QUESTION: {question_clean}
TOPIC: {topic_clean}
SUBJECT: {subject_clean}
CHOSEN LIBRARY: {chosen_library}
FILENAME: {unique_filename}

LIBRARY DESCRIPTION: {self.enhanced_libraries.get(chosen_library, 'Advanced visualization library')}

SPECIFIC TEMPLATE FOR {chosen_library.upper()}:
{templates}

CONTENT ANALYSIS: Create a diagram that specifically illustrates the concepts in the question.
Make it educational, clear, and relevant to the topic.

Generate code using {chosen_library} that creates a professional diagram for: {question_clean}
Save as: {unique_filename}

Return ONLY the Python code:
"""
        
        try:
            # Generate code directly without PromptTemplate to avoid curly brace conflicts
            try:
                # Use direct invocation to avoid template parsing issues
                response = self.llm.invoke(diagram_prompt).content
            except Exception as e:
                print(f"⚠️ Direct invocation failed: {e}")
                # Fallback: use simple string prompt without any template parsing
                try:
                    # Try alternative invocation method
                    response = self.llm(diagram_prompt)
                    if isinstance(response, dict) and 'text' in response:
                        response = response['text']
                except Exception as e2:
                    print(f"⚠️ Alternative invocation also failed: {e2}")
                    # Final fallback: create a basic diagram
                    print("🔄 Using fallback diagram generation")
                    return self._create_fallback_diagram_code(unique_filename, topic)
            
            # Clean response
            response = response.strip()
            response = self._sanitize_text(response)
            
            if response.startswith("```python"):
                response = response[9:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            # Ensure filename is correct in the generated code
            response = self._ensure_correct_filename(response, unique_filename, chosen_library)
            
            print(f"✅ Generated {chosen_library} code for: {topic}")
            return response
            
        except Exception as e:
            print(f"❌ Error generating enhanced diagram code: {e}")
            # Return a fallback matplotlib diagram
            return self._create_fallback_diagram_code(unique_filename, topic)
    
    def _get_library_templates(self, library: str, filename: str, topic: str, question: str) -> str:
        """Get specific templates for each library"""
        
        templates = {
            "mermaid": f"""
# Mermaid diagrams using mermaid-py
from mermaid import Mermaid
mermaid = Mermaid()
diagram_code = '''
graph TD
    A[Start] --> B[Process]
    B --> C[Decision]
    C -->|Yes| D[Action]
    C -->|No| E[Alternative]
'''
mermaid.render(diagram_code, '{filename}')
""",
            
            "pyvis": f"""
# Interactive network using PyVis
from pyvis.network import Network
import networkx as nx

net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
G = nx.Graph()
G.add_edges_from([('Node1', 'Node2'), ('Node2', 'Node3')])
net.from_nx(G)
net.show('{filename.replace(".png", ".html")}')
# Convert to image for consistency
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.text(0.5, 0.5, 'Interactive Network Generated\\nView {filename.replace(".png", ".html")} for interactive version', 
         ha='center', va='center', fontsize=14)
plt.axis('off')
plt.savefig('{filename}', dpi=300, bbox_inches='tight')
plt.close()
""",

            "pyecharts": f"""
# Rich charts using PyEcharts
from pyecharts.charts import Bar, Line, Pie
from pyecharts import options as opts
import os

chart = Bar()
chart.add_xaxis(['A', 'B', 'C', 'D'])
chart.add_yaxis('Series', [1, 2, 3, 4])
chart.set_global_opts(title_opts=opts.TitleOpts(title='{topic}'))
chart.render('{filename.replace(".png", ".html")}')

# Create static version
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(['A', 'B', 'C', 'D'], [1, 2, 3, 4])
plt.title('{topic}')
plt.savefig('{filename}', dpi=300, bbox_inches='tight')
plt.close()
""",

            "altair": f"""
# Statistical visualization using Altair
import altair as alt
import pandas as pd

data = pd.DataFrame({{'x': [1, 2, 3, 4], 'y': [1, 4, 2, 3], 'category': ['A', 'B', 'C', 'D']}})
chart = alt.Chart(data).mark_circle(size=100).encode(
    x='x:Q',
    y='y:Q',
    color='category:N'
).properties(title='{topic}')
chart.save('{filename}')
""",

            "pydot": f"""
# DOT language graphs using PyDot
import pydot

graph = pydot.Dot(graph_type='digraph')
node_a = pydot.Node("Node A", style="filled", fillcolor="lightblue")
node_b = pydot.Node("Node B", style="filled", fillcolor="lightgreen")
graph.add_node(node_a)
graph.add_node(node_b)
graph.add_edge(pydot.Edge(node_a, node_b))
graph.write_png('{filename}')
""",

            "bokeh": f"""
# Interactive web visualization using Bokeh
from bokeh.plotting import figure, save, output_file
from bokeh.io import export_png

p = figure(title='{topic}', x_axis_label='x', y_axis_label='y')
p.line([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], legend_label="Line", line_width=2)
p.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], legend_label="Circle", size=10)

output_file('{filename.replace(".png", ".html")}')
save(p)

# Create static version
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], 'o-', linewidth=2, markersize=8)
plt.title('{topic}')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('{filename}', dpi=300, bbox_inches='tight')
plt.close()
""",

            "networkx": f"""
# Graph analysis using NetworkX
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()
G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('A', 'C')])

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', 
        node_size=500, font_size=16, font_weight='bold')
plt.title('{topic}')
plt.savefig('{filename}', dpi=300, bbox_inches='tight')
plt.close()
"""
        }
        
        return templates.get(library, f"# Use {library} to create a diagram for {topic}")
    
    def _ensure_correct_filename(self, code: str, filename: str, library: str) -> str:
        """Ensure the generated code uses the correct filename"""
        
        # More aggressive filename replacement to catch all cases
        
        # 1. Replace any .png filename in quotes
        code = re.sub(r'["\'][^"\']*\.png["\']', f'"{filename}"', code)
        
        # 2. Replace any .html filename for web libraries
        if library in ["pyvis", "pyecharts", "bokeh"]:
            html_filename = filename.replace('.png', '.html')
            code = re.sub(r'["\'][^"\']*\.html["\']', f'"{html_filename}"', code)
            # Ensure PNG output exists
            if '.savefig(' not in code:
                code += f'\n# Ensure PNG output\nimport matplotlib.pyplot as plt\nplt.figure(figsize=(10, 6))\nplt.text(0.5, 0.5, "Interactive {library.title()} Diagram Generated", ha="center", va="center", fontsize=14)\nplt.axis("off")\nplt.savefig("{filename}", dpi=300, bbox_inches="tight")\nplt.close()'
        
        # 3. Library-specific fixes
        if library == "mermaid":
            code = re.sub(r'mermaid\.render\([^)]+\)', 
                         f'mermaid.render(diagram_code, "{filename}")', code)
        elif library == "altair":
            code = re.sub(r'\.save\([^)]+\)', f'.save("{filename}")', code)
        elif library == "pydot":
            code = re.sub(r'\.write_png\([^)]+\)', f'.write_png("{filename}")', code)
        elif library in ["matplotlib", "seaborn"]:
            code = re.sub(r'\.savefig\([^)]+\)', f'.savefig("{filename}", dpi=300, bbox_inches="tight")', code)
        elif library == "plotly":
            code = re.sub(r'\.write_image\([^)]+\)', f'.write_image("{filename}")', code)
            # Add fallback if no write_image found
            if '.write_image(' not in code:
                code += f'\n# Save as image\nfig.write_image("{filename}")'
        
        # 4. Final safety check - if no save command found, add matplotlib fallback
        save_commands = ['.savefig(', '.save(', '.write_png(', '.write_image(', '.render(']
        if not any(cmd in code for cmd in save_commands):
            code += f'''
# Fallback save mechanism
import matplotlib.pyplot as plt
if 'plt' not in globals():
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, "Diagram Generated", ha='center', va='center', fontsize=14)
    plt.axis('off')
plt.savefig("{filename}", dpi=300, bbox_inches="tight")
plt.close()
'''
        
        return code
    
    def _create_fallback_diagram_code(self, filename: str, topic: str) -> str:
        """Create meaningful contextual fallback diagram instead of sine wave"""
        topic_clean = self._sanitize_text(topic)
        
        # Create topic-specific meaningful diagrams instead of generic sine waves
        if any(word in topic_clean.lower() for word in ['neural', 'network', 'ai', 'machine']):
            return f"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Neural Network Diagram for: {topic_clean}
fig, ax = plt.subplots(figsize=(10, 6))

# Draw simple neural network
layers = [3, 4, 2]  # Input, Hidden, Output
layer_names = ['Input', 'Hidden', 'Output']
colors = ['lightblue', 'lightgreen', 'lightcoral']

for layer_idx, (num_nodes, name, color) in enumerate(zip(layers, layer_names, colors)):
    x = layer_idx * 3 + 1
    for node_idx in range(num_nodes):
        y = node_idx * 1.5 - (num_nodes - 1) * 0.75 + 3
        circle = patches.Circle((x, y), 0.2, facecolor=color, edgecolor='black')
        ax.add_patch(circle)
        
        # Connect to next layer
        if layer_idx < len(layers) - 1:
            next_layer_size = layers[layer_idx + 1]
            for next_node in range(next_layer_size):
                next_y = next_node * 1.5 - (next_layer_size - 1) * 0.75 + 3
                ax.plot([x + 0.2, x + 2.8], [y, next_y], 'gray', alpha=0.5)
    
    ax.text(x, 1, name, ha='center', fontweight='bold')

ax.set_xlim(0, 8)
ax.set_ylim(0, 6)
ax.set_title(f'Neural Network: {topic_clean}', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('{filename}', dpi=300, bbox_inches='tight')
plt.close()
"""
        elif any(word in topic_clean.lower() for word in ['algorithm', 'process', 'flow', 'chain']):
            return f"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Algorithm Flowchart for: {topic_clean}
fig, ax = plt.subplots(figsize=(8, 10))

steps = ['Start', 'Input', 'Process', 'Decision', 'Output', 'End']
y_positions = [8, 6.5, 5, 3.5, 2, 0.5]
colors = ['lightgreen', 'lightyellow', 'lightblue', 'orange', 'lightyellow', 'lightcoral']

for i, (step, y, color) in enumerate(zip(steps, y_positions, colors)):
    if step == 'Decision':
        # Diamond shape for decision
        diamond = mpatches.Polygon([(4, y+0.5), (5.5, y), (4, y-0.5), (2.5, y)], 
                                 facecolor=color, edgecolor='black')
        ax.add_patch(diamond)
    else:
        # Rectangle for other steps
        rect = mpatches.Rectangle((2.5, y-0.3), 3, 0.6, facecolor=color, edgecolor='black')
        ax.add_patch(rect)
    
    ax.text(4, y, step, ha='center', va='center', fontweight='bold')
    
    # Add arrows
    if i < len(steps) - 1:
        ax.arrow(4, y-0.4, 0, -1.7, head_width=0.1, head_length=0.1, fc='black', ec='black')

ax.set_xlim(1, 7)
ax.set_ylim(0, 9)
ax.set_title(f'Algorithm Flow: {topic_clean}', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('{filename}', dpi=300, bbox_inches='tight')
plt.close()
"""
        elif any(word in topic_clean.lower() for word in ['system', 'architecture', 'structure']):
            return f"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# System Architecture for: {topic_clean}
fig, ax = plt.subplots(figsize=(10, 8))

# Define system layers
layers = [
    ('User Interface', 4, 'lightblue'),
    ('Application Layer', 3, 'lightgreen'),
    ('Business Logic', 2, 'lightcoral'),
    ('Data Layer', 1, 'lightyellow'),
    ('Database', 0, 'lightgray')
]

for name, y, color in layers:
    rect = mpatches.Rectangle((2, y), 6, 0.8, facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, y+0.4, name, ha='center', va='center', fontweight='bold')
    
    # Add arrows between layers
    if y > 0:
        ax.arrow(5, y-0.1, 0, -0.7, head_width=0.2, head_length=0.1, fc='black', ec='black')

ax.set_xlim(1, 9)
ax.set_ylim(-0.5, 5)
ax.set_title(f'System Architecture: {topic_clean}', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('{filename}', dpi=300, bbox_inches='tight')
plt.close()
"""
        else:
            return f"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Conceptual Diagram for: {topic_clean}
fig, ax = plt.subplots(figsize=(10, 6))

# Create a concept map
concepts = [
    ('Main Concept', 5, 3, 'lightblue'),
    ('Related Idea 1', 2, 4, 'lightgreen'),
    ('Related Idea 2', 8, 4, 'lightcoral'),
    ('Supporting Detail', 5, 1, 'lightyellow')
]

for concept, x, y, color in concepts:
    rect = mpatches.Rectangle((x-1, y-0.3), 2, 0.6, facecolor=color, edgecolor='black')
    ax.add_patch(rect)
    ax.text(x, y, concept, ha='center', va='center', fontweight='bold', fontsize=9)

# Add connections
connections = [((5, 3), (2, 4)), ((5, 3), (8, 4)), ((5, 3), (5, 1))]
for (x1, y1), (x2, y2) in connections:
    ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, alpha=0.7)

ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.set_title(f'Concept Map: {topic_clean}', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('{filename}', dpi=300, bbox_inches='tight')
plt.close()
"""
    
    def _sanitize_text(self, text: str) -> str:
        """Sanitize text to remove problematic Unicode characters"""
        # Replace common problematic Unicode characters
        replacements = {
            'Δ': 'Delta', 'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'θ': 'theta',
            'λ': 'lambda', 'μ': 'mu', 'π': 'pi', 'σ': 'sigma', 'φ': 'phi',
            'ω': 'omega', '∑': 'sum', '∏': 'product', '∫': 'integral',
            '∞': 'infinity', '≤': '<=', '≥': '>=', '≠': '!=', '±': '+/-',
            '×': '*', '÷': '/', '"': '"', '"': '"', ''': "'", ''': "'",
            '–': '-', '—': '-'
        }
        
        for unicode_char, ascii_replacement in replacements.items():
            text = text.replace(unicode_char, ascii_replacement)
        
        # Remove any remaining non-ASCII characters
        text = text.encode('ascii', 'ignore').decode('ascii')
        return text
    
    def should_generate_diagram(self, question: str, topic: str) -> bool:
        """Enhanced check if diagram should be generated"""
        # More comprehensive keyword detection
        always_diagram_topics = [
            'backward chaining', 'forward chaining', 'expert system', 'minimax', 
            'alpha beta', 'search', 'algorithm', 'tree', 'graph', 'network',
            'neural', 'machine learning', 'ai', 'artificial intelligence',
            'process', 'scheduling', 'memory', 'operating system', 'database',
            'data structure', 'protocol', 'architecture', 'system design',
            'flowchart', 'diagram', 'visualization', 'chart', 'plot'
        ]
        
        diagram_keywords = [
            'diagram', 'illustrate', 'draw', 'show', 'visualize', 'graph', 'chart',
            'architecture', 'structure', 'topology', 'layout', 'design', 'model',
            'states', 'process', 'flow', 'network', 'system', 'algorithm',
            'explain', 'describe', 'define', 'demonstrate', 'represent',
            'interactive', 'dashboard', 'flowchart', 'sequence'
        ]
        
        text = f"{question.lower()} {topic.lower()}"
        
        # Check for always-diagram topics first
        if any(topic_keyword in text for topic_keyword in always_diagram_topics):
            return True
            
        # Check for diagram keywords
        if any(keyword in text for keyword in diagram_keywords):
            return True
            
        # For questions longer than 40 characters, likely complex enough for diagrams
        if len(question) > 40:
            return True
            
        return False
    
    def get_library_stats(self) -> Dict[str, int]:
        """Get enhanced library usage statistics"""
        return self.library_counter.copy()
    
    def get_available_libraries(self) -> Dict[str, str]:
        """Get list of available enhanced libraries"""
        return self.enhanced_libraries.copy()

if __name__ == "__main__":
    # Test the enhanced diagram generator
    generator = EnhancedDiagramGenerator()
    
    test_cases = [
        ("Create a flowchart showing the backward chaining process", "Backward Chaining"),
        ("Draw an interactive network topology diagram", "Network Topology"),
        ("Show statistical distribution of algorithm performance", "Algorithm Analysis"),
        ("Create a Gantt chart for project timeline", "Project Management"),
        ("Visualize the expert system architecture", "Expert Systems")
    ]
    
    print("🚀 Testing Enhanced Diagram Generator")
    print("=" * 50)
    
    for question, topic in test_cases:
        if generator.should_generate_diagram(question, topic):
            code = generator.generate_enhanced_diagram_code(question, topic)
            print(f"\n📊 Generated code for '{topic}':")
            print(f"Library usage: {generator.get_library_stats()}")
            print("-" * 50)
    
    print(f"\n📚 Available libraries: {list(generator.get_available_libraries().keys())}") 