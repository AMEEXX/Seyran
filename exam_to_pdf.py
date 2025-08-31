import os
import json
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch

class ExamToPDF:
    def __init__(self, exam_data_path="generated_exams/exam.json", output_path="generated_exams/final_exam.pdf"):
        """Initialize the PDF generator"""
        self.output_path = output_path
        
        # Load exam data
        with open(exam_data_path, 'r') as f:
            self.exam_questions = json.load(f)
        
        # Initialize document
        self.doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Get styles
        self.styles = getSampleStyleSheet()
        
        # Add custom styles
        self.styles.add(
            ParagraphStyle(
                name='SectionTitle',
                parent=self.styles['Heading1'],
                fontSize=14,
                spaceAfter=12
            )
        )
        
        self.styles.add(
            ParagraphStyle(
                name='QuestionText',
                parent=self.styles['Normal'],
                fontSize=12,
                spaceAfter=6
            )
        )
        
        self.styles.add(
            ParagraphStyle(
                name='AnswerText',
                parent=self.styles['Normal'],
                fontSize=10,
                leftIndent=20,
                spaceAfter=12
            )
        )
        
        self.styles.add(
            ParagraphStyle(
                name='HeaderText',
                parent=self.styles['Heading1'],
                fontSize=16,
                alignment=1,  # Center alignment
                spaceAfter=12
            )
        )
        
        self.styles.add(
            ParagraphStyle(
                name='InstructionsText',
                parent=self.styles['Normal'],
                fontSize=12,
                textColor=colors.red,
                spaceAfter=24
            )
        )
    
    def split_text_to_lines(self, text, max_chars=80):
        """Split text into lines of maximum length"""
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + len(current_line) <= max_chars:
                current_line.append(word)
                current_length += len(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines
    
    def create_exam_header(self):
        """Create the exam header with title and instructions"""
        elements = []
        
        # Title
        title = Paragraph("FINAL EXAMINATION", self.styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 0.25 * inch))
        
        # Instructions
        instructions = [
            "Time Allowed: 3 Hours",
            "Total Marks: 100",
            "Answer ALL questions from each section.",
            "Write your answers clearly and show all working where applicable."
        ]
        
        for instruction in instructions:
            p = Paragraph(instruction, self.styles['Normal'])
            elements.append(p)
        
        elements.append(Spacer(1, 0.5 * inch))
        return elements
    
    def create_exam_questions(self):
        """Create the exam questions content"""
        elements = []
        current_section = None
        q_number = 1
        
        for q in self.exam_questions:
            sec = q.get("section", "Unknown Section")
            
            # Add section header if changed
            if sec != current_section:
                current_section = sec
                section_title = Paragraph(sec, self.styles['SectionTitle'])
                elements.append(section_title)
            
            # Question text with marks
            question_text = f"<b>Q{q_number}.</b> ({q.get('marks', 0)} marks) {q.get('question', 'Error: No question text')}"
            elements.append(Paragraph(question_text, self.styles['QuestionText']))
            
            # Add diagram if present
            if q.get("diagram_path"):
                try:
                    img = Image(q["diagram_path"], width=5*inch, height=3*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 0.1 * inch))
                except Exception as e:
                    print(f"Error adding diagram for Q{q_number}: {e}")
            
            elements.append(Spacer(1, 0.25 * inch))
            q_number += 1
        
        return elements
    
    def create_answer_key(self):
        """Create an answer key section (for instructor use)"""
        elements = []
        
        # Answer key header
        answer_key_title = Paragraph("ANSWER KEY (FOR INSTRUCTOR USE ONLY)", self.styles['Title'])
        elements.append(answer_key_title)
        elements.append(Spacer(1, 0.25 * inch))
        
        q_number = 1
        for q in self.exam_questions:
            # Question reference
            ref_text = f"<b>Q{q_number}:</b> {q.get('topic', 'Unknown Topic')}"
            elements.append(Paragraph(ref_text, self.styles['Heading3']))
            
            # Answer or answer outline
            if "answer" in q:
                answer_text = f"<b>Answer:</b> {q['answer']}"
                elements.append(Paragraph(answer_text, self.styles['Normal']))
            elif "answer_outline" in q:
                elements.append(Paragraph("<b>Answer Outline:</b>", self.styles['Normal']))
                for step in q["answer_outline"]:
                    elements.append(Paragraph(f"• {step}", self.styles['Normal']))
            elif "rubric" in q:
                elements.append(Paragraph("<b>Marking Rubric:</b>", self.styles['Normal']))
                for point in q["rubric"]:
                    elements.append(Paragraph(f"• {point}", self.styles['Normal']))
            
            elements.append(Spacer(1, 0.25 * inch))
            q_number += 1
        
        return elements
    
    def generate_student_version(self):
        """Generate student version of the exam (without answers)"""
        # Create content list
        content = []
        
        # Add header and special instructions
        header_question = next((q for q in self.exam_questions if q.get("type") == "header"), None)
        if header_question:
            # Add header with total marks and duration from config
            header_text = header_question.get("content", "")
            if header_text:
                # Remove any date information and use exact format from config
                header_text = header_text.split("(")[0].strip()
                content.append(Paragraph(header_text, self.styles["HeaderText"]))
                content.append(Spacer(1, 12))
            
            # Add special instructions
            if header_question.get("special_instructions"):
                content.append(Paragraph("Special Instructions:", self.styles["Heading2"]))
                content.append(Paragraph(header_question["special_instructions"], self.styles["InstructionsText"]))
                content.append(Spacer(1, 24))
        
        # Group questions by section
        current_section = None
        total_marks = 0
        
        for question in self.exam_questions:
            if question.get("type") == "header":
                continue
                
            section = question.get("section")
            if section != current_section:
                current_section = section
                content.append(Paragraph(section, self.styles["SectionTitle"]))
                content.append(Spacer(1, 12))
            
            # Add question
            question_text = f"{question['question']} [{question['marks']} marks]"
            content.append(Paragraph(question_text, self.styles["QuestionText"]))
            total_marks += question['marks']
            
            # Add space for answer
            content.append(Spacer(1, 24))
        
        # Build PDF
        self.doc.build(content)
        print(f"Student exam PDF generated at: {self.output_path}")

    def generate_instructor_version(self):
        """Generate instructor version of the exam (with answers)"""
        # Create content list
        content = []
        
        # Add header and special instructions
        header_question = next((q for q in self.exam_questions if q.get("type") == "header"), None)
        if header_question:
            # Add header with total marks and duration from config
            header_text = header_question.get("content", "")
            if header_text:
                # Remove any date information and use exact format from config
                header_text = header_text.split("(")[0].strip()
                content.append(Paragraph(header_text, self.styles["HeaderText"]))
                content.append(Spacer(1, 12))
            
            # Add special instructions
            if header_question.get("special_instructions"):
                content.append(Paragraph("Special Instructions:", self.styles["Heading2"]))
                content.append(Paragraph(header_question["special_instructions"], self.styles["InstructionsText"]))
                content.append(Spacer(1, 24))
        
        # Group questions by section
        current_section = None
        total_marks = 0
        
        for question in self.exam_questions:
            if question.get("type") == "header":
                continue
                
            section = question.get("section")
            if section != current_section:
                current_section = section
                content.append(Paragraph(section, self.styles["SectionTitle"]))
                content.append(Spacer(1, 12))
            
            # Add question
            question_text = f"{question['question']} [{question['marks']} marks]"
            content.append(Paragraph(question_text, self.styles["QuestionText"]))
            total_marks += question['marks']
            
            # Add answer
            content.append(Paragraph("Answer:", self.styles["Heading3"]))
            if "answer_outline" in question:
                for step in question["answer_outline"]:
                    content.append(Paragraph(f"• {step}", self.styles["AnswerText"]))
            elif "answer" in question:
                content.append(Paragraph(question["answer"], self.styles["AnswerText"]))
            
            content.append(Spacer(1, 24))
        
        # Build PDF
        self.doc.build(content)
        print(f"Instructor exam PDF generated at: {self.output_path}")

    def generate_student_and_instructor_versions(self):
        """Generate both student and instructor versions of the exam"""
        # Student version
        student_path = "generated_exams/student_exam.pdf"
        self.output_path = student_path
        self.doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        self.generate_student_version()
        
        # Instructor version
        instructor_path = "generated_exams/instructor_exam.pdf"
        self.output_path = instructor_path
        self.doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        self.generate_instructor_version()
        
        return {
            "student_version": student_path,
            "instructor_version": instructor_path
        }

if __name__ == "__main__":
    # Create PDF generator
    pdf_generator = ExamToPDF()
    
    # Generate both versions
    result = pdf_generator.generate_student_and_instructor_versions()
    print("\nExam generation complete!")
    print(f"Student version: {result['student_version']}")
    print(f"Instructor version: {result['instructor_version']}")
