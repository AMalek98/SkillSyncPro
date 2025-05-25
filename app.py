"""
CV to Job Description Matching Application

This module initializes the Flask application and handles the web interface
for the CV to job description matching workflow.
"""

import os
from flask import Flask, request, render_template, jsonify
from cv_workflow import run_workflow

# Create uploads directory if it doesn't exist
os.makedirs('uploads', exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle file upload and display results"""
    if request.method == 'POST':
        # Check if files are uploaded
        if 'cv_file' not in request.files or 'jd_file' not in request.files:
            return render_template('index.html', error="Please upload both CV and job description files")
        
        cv_file = request.files['cv_file']
        jd_file = request.files['jd_file']
        
        # Check if files are valid
        if cv_file.filename == '' or jd_file.filename == '':
            return render_template('index.html', error="Please select both files")
        
        # Check file extensions
        if not cv_file.filename.lower().endswith('.pdf'):
            return render_template('index.html', error="CV must be in PDF format")
        
        if not jd_file.filename.lower().endswith('.txt'):
            return render_template('index.html', error="Job description must be in TXT format")
        
        # Save files
        cv_path = os.path.join('uploads', cv_file.filename)
        jd_path = os.path.join('uploads', jd_file.filename)
        
        cv_file.save(cv_path)
        jd_file.save(jd_path)
        
        # Run workflow
        result = run_workflow(cv_path, jd_path)
        
        # Check for errors
        if 'error' in result:
            return render_template('index.html', error=result['error'])
        
        # Return results
        return render_template(
            'results.html',
            cv_name=result['cv_name'],
            jd_name=result['job_description'],
            hard_skills_cv=result['hard_skills_cv'],
            hard_skills_jd=result['hard_skills_jd'],
            soft_skills_cv=result['soft_skills_cv'],
            soft_skills_jd=result['soft_skills_jd'],
            hard_skills_similarity=f"{result['similarity']['hard_skills'] * 100:.1f}%",
            soft_skills_similarity=f"{result['similarity']['soft_skills'] * 100:.1f}%"
        )
    
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_api():
    """API endpoint for analyzing CV and job description"""
    if 'cv_file' not in request.files or 'jd_file' not in request.files:
        return jsonify({"error": "Please upload both CV and job description files"}), 400
    
    cv_file = request.files['cv_file']
    jd_file = request.files['jd_file']
    
    # Check if files are valid
    if cv_file.filename == '' or jd_file.filename == '':
        return jsonify({"error": "Please select both files"}), 400
    
    # Save files
    cv_path = os.path.join('uploads', cv_file.filename)
    jd_path = os.path.join('uploads', jd_file.filename)
    
    cv_file.save(cv_path)
    jd_file.save(jd_path)
    
    # Run workflow
    result = run_workflow(cv_path, jd_path)
    
    # Return results as JSON
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)