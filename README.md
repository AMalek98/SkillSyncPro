# SkillSyncPro – AI-Powered CV Analyzer for Job Description Matching

SkillSyncPro is an open-source AI-powered CV analyzer that compares your resume against a job description. It extracts and compares hard and soft skills using natural language processing and embeddings to help job seekers and recruiters identify keyword matches.

## Features

- Extracts hard and soft skills from CVs and job descriptions
- Calculates similarity percentages for both skill types
- Displays results in a user-friendly web interface
- Logs results in a structured JSON file

## Technology Stack

- **Workflow Orchestration**: LangGraph
- **Document Loading**: LangChain built-in loaders
- **LLM for Keyword Extraction**: Claude Haiku 3.5
- **Embedding Model**: nomic-embed-text
- **Similarity Calculation**: Cosine Similarity
- **Frontend UI**: Flask
- **Result Logging**: JSON File
- **Containerization**: Docker

## Installation & Setup

### Option 1: Docker (Recommended)

1. Clone this repository
2. Create a `.env` file in the root directory and add your API key:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

3. Make sure you have Docker installed on your system

4. Build and run the application using Docker Compose:

```bash
docker-compose up --build
```

5. Open your browser and navigate to: http://127.0.0.1:5000/

### Option 2: Local Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your API key:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

4. Make sure you have Ollama installed with the nomic-embed-text model:

```bash
ollama pull nomic-embed-text
```

5. Run the Flask application:

```bash
python app.py
```

6. Open your browser and navigate to: http://127.0.0.1:5000/

## Environment Variables

**Important**: You must create a `.env` file in the root directory with your API credentials. The application requires:

- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude Haiku 3.5

Example `.env` file:
```
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-api-key-here
```

## Usage

1. Access the web application at http://127.0.0.1:5000/

2. Upload your CV (PDF format) and job description (TXT format)

3. View the analysis results showing:
   - Extracted hard and soft skills from both documents
   - Similarity percentages for both skill types
   - Overall match score

## Project Structure

- `app.py`: Flask application for web interface
- `cv_workflow.py`: LangGraph workflow implementation
- `docker-compose.yml`: Docker configuration for easy deployment
- `Dockerfile`: Container configuration
- `requirements.txt`: Python dependencies
- `.env`: Environment variables (create this file)
- `templates/`: HTML templates for the web interface
- `uploads/`: Directory where uploaded files are stored
- `results.json`: Output file containing analysis results

## How It Works

1. **File Upload**: User uploads CV and job description through the Flask UI
2. **Text Extraction**: Extracts text from PDF (CV) and TXT (job description)
3. **Keyword Extraction**: Claude Haiku extracts hard and soft skills from both documents
4. **Embedding & Similarity**: Skills are embedded using nomic-embed-text and compared using cosine similarity
5. **Results Display**: Similarity percentages and extracted skills are displayed in the web interface
6. **Logging**: Results are saved to a JSON file for record keeping

## API Endpoint

The application also provides a RESTful API endpoint:

- `POST /api/analyze`: Upload CV and job description files for analysis
  - Returns a JSON object with extracted skills and similarity scores

## Docker Commands

- **Build and run**: `docker-compose up --build`
- **Run in background**: `docker-compose up -d`
- **Stop containers**: `docker-compose down`
- **View logs**: `docker-compose logs`

## Troubleshooting

- **Missing API Key**: Ensure you have created a `.env` file with your `ANTHROPIC_API_KEY`
- **Ollama Connection**: Make sure Ollama is running and the nomic-embed-text model is available
- **Port Conflicts**: If port 5000 is in use, modify the port mapping in `docker-compose.yml`

## License

[MIT License](LICENSE)
