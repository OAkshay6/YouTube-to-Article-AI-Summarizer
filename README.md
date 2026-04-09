# YouTube to Article AI Summarizer

An AI-powered application that converts YouTube videos into structured articles, professional PDFs, and responsive websites using modern LLMs and multi-fallback transcript extraction.

## Features

* Multi-fallback transcript extraction
* Smart summarization (short vs long videos)
* Recursive summarization for long videos
* Gemini as primary model
* Groq (Llama 3) fallback support
* Professional PDF generation
* Responsive website generation
* Streamlit-based UI
* Production-ready architecture

## Tech Stack

* Python
* Streamlit
* LangChain
* Google Gemini
* Groq (Llama 3)
* ReportLab
* yt-dlp
* YouTube Transcript API
* Pytube

## How It Works

1. Extract transcript from YouTube video
2. Clean and process transcript
3. Generate structured article using AI
4. Export article to PDF
5. Generate responsive website

## Project Structure

```
YouTube-to-Article-AI-Summarizer
│
├── app.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/YouTube-to-Article-AI-Summarizer.git
```

Navigate to project folder:

```
cd YouTube-to-Article-AI-Summarizer
```

Create virtual environment:

```
python -m venv venv
```

Activate virtual environment:

### Windows

```
venv\Scripts\activate
```

### Mac / Linux

```
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```
GOOGLE_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
```

## Run Application

```
streamlit run app.py
```

## Example Use Cases

* Convert educational videos into notes
* Convert tutorials into articles
* Generate blog content from YouTube videos
* Create shareable PDF summaries
* Generate article websites automatically

## Future Improvements

* Thumbnail extraction
* Video metadata extraction
* Markdown export
* Notion integration
* Multi-language support

## Author

O Akshaykumar
