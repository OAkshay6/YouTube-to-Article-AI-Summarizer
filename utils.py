# ================================
# Standard Library
# ================================

import os        # Environment variables and system operations
import re        # Regex operations (extract video ID, clean text)
import zipfile   # Create zip files (website export)


# ================================
# Third-Party Libraries
# ================================

import requests  # Fetch subtitles / web content
import yt_dlp    # YouTube metadata & audio extraction
from pytube import YouTube  # Pytube fallback


# ================================
# Environment / Configuration
# ================================

from dotenv import load_dotenv  # Store API keys securely


# ================================
# YouTube Transcript Extraction
# ================================

from youtube_transcript_api import YouTubeTranscriptApi  # Direct transcript extraction


# ================================
# LLM Models
# ================================

from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini model
from langchain_groq import ChatGroq                        # Groq fallback


# ================================
# Text Splitting
# ================================

from langchain_text_splitters import RecursiveCharacterTextSplitter  # Adaptive chunking


# ================================
# YouTube Loader
# ================================

from langchain_community.document_loaders import YoutubeLoader  # LangChain loader


# ================================
# Prompt Templates
# ================================

from langchain_core.prompts import (
    ChatPromptTemplate,          # Structured prompts
    SystemMessagePromptTemplate, # System instructions
    HumanMessagePromptTemplate   # User instructions
)


# ================================
# LangChain Runnables
# ================================

from langchain_core.runnables import (
    RunnablePassthrough,  # Pass input unchanged
    RunnableParallel,     # Parallel execution (future scaling)
    RunnableBranch,       # Smart routing
    RunnableLambda        # Convert functions into pipelines
)


# ================================
# Output Parser
# ================================

from langchain_core.output_parsers import StrOutputParser  # Convert LLM output to string


# ================================
# PDF Generation
# ================================

from reportlab.platypus import (
    SimpleDocTemplate,  # Create PDF
    Paragraph,          # Add formatted text
    Spacer,             # Vertical spacing
    PageBreak           # Page break
)

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # Styles
from reportlab.lib.enums import TA_CENTER  # Text alignment
from reportlab.lib import colors  # Colors

# ================================
# Caching / Performance Utilities
# ================================
from functools import lru_cache

load_dotenv()

# ==============================
# Loading API Key
# ==============================

api_key_gemini = os.getenv('GEMINI_API_KEY')

api_key_groq = os.getenv('GROQ_API_KEY')

# ================================
# Gemini Model (Primary Model)
# ================================

def get_gemini():
    """
    Returns Gemini LLM model.

    Used as the primary model for summarization.
    """

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=8192
    )


# ================================
# Groq Model (Fallback Model)
# ================================

def get_groq():
    """
    Returns Groq LLM model.

    Used when:
    - Gemini quota exhausted
    - Gemini API error
    - Rate limits
    """

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=8000
    )


# ================================
# Adaptive Chunking
# ================================

def get_adaptive_chunks(text):
    """
    Splits long transcript into adaptive chunks
    for recursive summarization.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,      # Size of each chunk
        chunk_overlap=120,   # Overlap for context continuity
        separators=["\n\n", "\n", ".", " ", ""]
    )

    return splitter.split_text(text)


# ================================
# Extract YouTube Video ID
# ================================

def extract_video_id(url):
    """
    Extracts video ID from a YouTube URL.

    Supports:
    - https://www.youtube.com/watch?v=xxxx
    - https://youtu.be/xxxx
    """

    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11})"  # Regex pattern

    match = re.search(pattern, url)

    if match:
        return match.group(1)

    raise ValueError("Could not extract video ID. Check URL.")


# ================================
# Clean Transcript Text
# ================================

def clean_transcript(text):
    """
    Cleans raw transcript text.

    Removes:
    - [Music], [Applause]
    - <think> reasoning blocks
    - "Okay..." reasoning text
    - Extra whitespace
    - Blank lines
    """

    # Remove [Music]
    text = re.sub(r"\[.*?\]", "", text)

    # Clean whitespace
    text = re.sub(r"\n+", "\n", text)     # Remove extra newlines
    text = re.sub(r"\s+", " ", text)      # Remove extra spaces

    return text.strip()


# ================================
# Pytube Transcript Fallback
# ================================

def extract_pytube_transcript(url):
    """
    Extract transcript using Pytube fallback.
    Used when other transcript methods fail.
    """

    try:
        print("Using Pytube fallback")

        yt = YouTube(url)
        caption = yt.captions.get_by_language_code("en")  # English captions

        if caption:
            text = caption.generate_srt_captions()
            return clean_transcript(text)

    except Exception:
        print("Pytube failed")

    return None



# ================================
# Extract YouTube Transcript (Multi-Fallback)
# ================================

@lru_cache(maxsize=20)
def extract_transcript(url):
    """
    Extract transcript using multiple fallback methods.

    Fallback Order:
    1. LangChain Loader
    2. YouTube Transcript API
    3. yt-dlp subtitles
    4. Pytube captions
    """

    video_id = extract_video_id(url)


    # ================================
    # 1. LangChain Loader
    # ================================
    try:
        print("Using LangChain Loader")

        loader = YoutubeLoader.from_youtube_url(url)
        docs = loader.load()

        if docs:
            text = docs[0].page_content
            return clean_transcript(text)

    except Exception:
        print("LangChain failed")


    # ================================
# 2. YouTube Transcript API
# ================================
    try:
        print("[INFO] Using YouTubeTranscriptApi")

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        transcript = transcript_list.find_transcript(
            ["en", "en-IN", "en-GB"]
        )

        transcript = transcript.fetch()

        text = " ".join([t["text"] for t in transcript])

        return clean_transcript(text)

    except Exception:
        print("[WARNING] Transcript API failed")


    # ================================
    # 3. yt-dlp Subtitle Extraction
    # ================================
    try:
        print("Using yt-dlp")

        ydl_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "quiet": True,
            "no_warnings": True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            subtitles = info.get("subtitles") or info.get("automatic_captions")

            if subtitles:

                preferred_langs = ["en", "en-IN", "en-US"]
                link = None

                for lang in preferred_langs:
                    if lang in subtitles:
                        link = subtitles[lang][0]["url"]
                        break
                else:
                    first_lang = list(subtitles.keys())[0]
                    link = subtitles[first_lang][0]["url"]

                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(link, headers=headers, timeout=10)
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")

                if "application/json" in content_type:

                    data = response.json()

                    text = " ".join(
                        seg["utf8"]
                        for event in data.get("events", [])
                        if "segs" in event
                        for seg in event["segs"]
                    )

                else:
                    text = response.text

                return clean_transcript(text)

    except Exception:
        print("yt-dlp failed")


    # ================================
    # 4. Pytube Fallback
    # ================================
    try:
      print("Using Pytube fallback")

      text = extract_pytube_transcript(url)

      if text:
        return clean_transcript(text)

    except Exception:
        print("Pytube fallback failed")


    return None



# ================================
# Extract Audio (Fallback Method)
# ================================

def extract_audio_transcript(url):
    """
    Extract audio when transcript is unavailable.
    Placeholder for future speech-to-text integration.
    """

    try:
        print("Extracting Audio")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": "audio.%(ext)s",
            "quiet": True
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        return None

    except Exception as e:
        print("Audio extraction failed:", e)
        return None


# ================================
# Visual Frame Analysis (Fallback)
# ================================

def analyze_video_frames(url):
    """
    Handles silent videos with no transcript/audio.
    Placeholder for future vision model integration.
    """

    print("Analyzing frames (silent video fallback)")
    return "This appears to be a silent visual video."


# ================================
# Main Video Processing Pipeline
# ================================

def process_video(url):
    """
    Main pipeline:
    1. Transcript extraction
    2. Audio fallback
    3. Visual fallback
    """

    # Transcript
    transcript = extract_transcript(url)

    if transcript:
        print("Transcript found")
        return transcript


    # Audio fallback
    audio = extract_audio_transcript(url)

    if audio:
        print("Audio transcript found")
        return audio


    # Visual fallback
    return analyze_video_frames(url)



def clean_model_output(text):
    """
    Clean model output safely without losing content
    """

    # Remove <think> blocks only
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove leading reasoning only if at beginning
    text = re.sub(
        r"^(Okay|Let me|I need to|First).*?\n",
        "",
        text,
        flags=re.IGNORECASE
    )

    # Remove excessive blank lines (NOT spaces)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()



# ================================
# Base Summarizer Prompt
# ================================

def base_summarizer_prompt(text):
    """
    Creates summarization prompt for short videos.
    Converts transcript into structured article.
    """

    system_prompt = """
You are a professional content writer.

IMPORTANT:
You are NOT summarizing.
You are converting transcript into structured article.

Your goal:
- Preserve ALL important information
- Maintain clarity and structure
- Organize content logically

STRICT RULES:
- DO NOT summarize
- DO NOT remove important details
- DO NOT compress content excessively
- Preserve examples, steps, numbers, and technical details
- Maintain original meaning

DO NOT include:
- reasoning
- thinking
- planning
- explanations
- meta commentary
- "Okay"
- "Let me"
- "I need to"

IGNORE:
- welcome messages
- subscribe requests
- promotions
- branding
- filler phrases

STRICT FORMAT:

Title:
[text]

Introduction:
[text]

Main Points:
- bullet
- bullet

Key Takeaways:
- bullet
- bullet

Summary:
[text]

DO NOT use **bold headings**
Use only the exact labels above

FORMATTING RULES:
- DO NOT use markdown headings like #, ##, ###
- DO NOT use horizontal rules like --- or ***
- Keep formatting consistent


SUPPORT ALL CONTENT TYPES:
- Tutorials
- Podcasts
- Interviews
- Educational videos
- Business content
- Movie reviews
- Technical explanations
- Story-based content
- Any video content


MANDATORY STRUCTURE:

Title:
Introduction:
Main Points:
Key Takeaways:
Summary:

STYLE:
- Professional tone
- Clear structure
- Concise but complete
- Actionable where possible

Return ONLY the article.
"""

    human_prompt = """
Convert this video content into an engaging article.

Structure:
- Title
- Introduction
- Main Points
- Key Takeaways
- Summary

Here is the raw transcript:
{text}
"""

    # Create prompt template
    summarizer_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(human_prompt)
    ])

    return summarizer_prompt



def get_base_summarizer(text):
    """
    Summarizes short videos using base summarizer.

    """
    try:
        print("Using Gemini")
        llm = get_gemini()

        chain = (
            RunnableLambda(lambda text: base_summarizer_prompt(text))
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(text)
        return clean_model_output(response)

    except Exception:

        print("Gemini failed → Using Groq")

        llm = get_groq()

        chain = (
            RunnableLambda(lambda text: base_summarizer_prompt(text))
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(text)
        return clean_model_output(response)
    

# ================================
# Recursive Summarizer Prompt
# ================================

def recursive_summarizer_prompt():
    """
    Creates recursive summarization prompt for long videos.
    """

    system_prompt = """
You are a recursive content structuring engine.

IMPORTANT:
You are NOT summarizing.
You are merging content into a structured article.

Your goal:
- Merge new content into existing article
- Preserve ALL previous information
- Add new information logically
- Avoid duplication only if identical content appears

STRICT RULES:
- DO NOT summarize
- DO NOT remove important details
- DO NOT compress content excessively
- Preserve technical details and examples

FORMATTING RULES:
- Use bold headings only using **Heading**
- DO NOT use markdown headings like #, ##, ###
- DO NOT use horizontal rules like --- or ***
- Keep formatting consistent

DO NOT include:
- reasoning
- thinking
- planning
- meta commentary

SUPPORT ALL CONTENT TYPES:
- Tutorials
- Podcasts
- Interviews
- Educational videos
- Business content
- Movie reviews
- Technical explanations
- Story-based content
- Any video content

Maintain structure:

Title:
Introduction:
Main Points:
Key Takeaways:
Summary:

STYLE:
- Professional
- Clear
- Well organized

Return ONLY updated article.
"""

    human_prompt = """
Current Summary:
{running_summary}

New Content:
{chunk}

Update the summary and maintain structure:
- Title
- Introduction
- Main Points
- Key Takeaways
- Summary
"""

    summarizer_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(human_prompt)
    ])

    return summarizer_prompt




# ================================
# Recursive Summarization Engine
# ================================

def get_recursive_summarizer(text):
    """
    Performs recursive summarization for long transcripts.
    """

    chunks = get_adaptive_chunks(text)   # Split transcript
    running_summary = ""                 # Initialize summary

    prompt = recursive_summarizer_prompt()

    for chunk in chunks:

        try:
            print("Using Gemini")

            chain = (
                prompt
                | get_gemini()
                | StrOutputParser()
            )

            updated_summary = chain.invoke({
                "running_summary": running_summary,
                "chunk": chunk
            })

        except Exception:

            print("Gemini failed → Using Groq")

            chain = (
                prompt
                | get_groq()
                | StrOutputParser()
            )

            updated_summary = chain.invoke({
                "running_summary": running_summary,
                "chunk": chunk
            })

        running_summary = clean_model_output(updated_summary)

    return running_summary




# ================================
# Smart Summarizer
# ================================

@lru_cache(maxsize=10)
def get_smart_summarizer(url):
    """
    Automatically selects summarization strategy
    based on transcript length.
    """

    transcript = process_video(url)

    if transcript is None:
        return "Could not extract transcript"


    # Long video
    if len(transcript.split()) >= 1000:
        print("Using recursive summarizer")
        return get_recursive_summarizer(transcript)


    # Short video
    else:
        print("Using base summarizer")
        return get_base_summarizer(transcript)
    



def extract_title(article):

    lines = article.split("\n")

    for i, line in enumerate(lines):

        if "Title:" in line:

            # If title is on same line
            if len(line.split("Title:")) > 1:
                title = line.split("Title:")[1].strip()

            # If title is next line
            else:
                title = lines[i + 1].strip()

            # Clean filename
            title = re.sub(r'[\\/*?:"<>|]', "", title)

            return title[:50]

    return "AI_YouTube_Summary"

    
# ================================
# Generate PDF
# ================================

def generate_pdf(article, video_title="summary"):

    filename = f"{video_title}.pdf"

    styles = getSampleStyleSheet()

    title_style = styles["Heading1"]
    heading_style = styles["Heading2"]
    body_style = styles["BodyText"]

    story = []

    lines = article.split("\n")

    for line in lines:

        line = line.strip()

        if not line:
            story.append(Spacer(1, 6))
            continue

        if line.startswith("Title:"):
            story.append(Paragraph(line, title_style))

        elif line.startswith("Introduction:"):
            story.append(Spacer(1, 12))
            story.append(Paragraph(line, heading_style))

        elif line.startswith("Main Points:"):
            story.append(Spacer(1, 12))
            story.append(Paragraph(line, heading_style))

        elif line.startswith("Key Takeaways:"):
            story.append(Spacer(1, 12))
            story.append(Paragraph(line, heading_style))

        elif line.startswith("Summary:"):
            story.append(Spacer(1, 12))
            story.append(Paragraph(line, heading_style))

        elif line.startswith("-"):
            bullet = f"• {line[1:].strip()}"
            story.append(Paragraph(bullet, body_style))

        else:
            story.append(Paragraph(line, body_style))

        story.append(Spacer(1, 6))

    doc = SimpleDocTemplate(filename)
    doc.build(story)

    return filename




# ================================
# Website Formatter Prompt
# ================================

def website_formatter_prompt(article):

    system_prompt = """
        You are a Senior Frontend Web Developer.

        Your task:
        Convert article content into a professional blog webpage.

        MANDATORY OUTPUT FORMAT:

        --html--
        [html code]
        --html--

        --css--
        [css code]
        --css--

        --js--
        [js code]
        --js--

        DESIGN REQUIREMENTS:

        - Clean modern blog layout
        - Medium-style article design
        - Centered container (max width 800px)
        - Professional typography
        - Large readable headings
        - Proper spacing between sections
        - Smooth scroll
        - Dark/light theme toggle
        - Responsive mobile-first design

        TYPOGRAPHY:

        - Title: 36px
        - Headings: 24px
        - Body: 18px
        - Line-height: 1.7

        STRUCTURE:

        - Header
        - Article Container
        - Title
        - Introduction
        - Main Points
        - Key Takeaways
        - Summary
        - Footer

        DO NOT:

        - Use markdown
        - Write explanations
        - Write comments outside delimiters

        Return only HTML, CSS and JS.
    """

    human_prompt = '''
        Create a **production-ready article webpages** in the style of **Medium, Dev.to, Hashnode, and Substack**.

        **MANDATORY REQUIREMENTS**:
        - **Mobile-first responsive design** (perfect on all devices)
        - **Clean, modern typography** (system fonts + readability first)
        - **Medium-like article layout** with card-based design
        - **Dark/light theme toggle**
        - **Smooth animations** and **scroll effects**
        - **SEO optimized** with proper meta tags
        - **Accessibility compliant** (ARIA labels, keyboard navigation)

        **CONTENT TO USE**: {article}
    '''

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        HumanMessagePromptTemplate.from_template(human_prompt)
    ])


# ================================
# Generate Website
# ================================

def generate_website(article):
    """
    Generate website files (HTML, CSS, JS) from article.
    """

    try:
        print("Using Gemini")

        chain = (
            RunnableLambda(lambda article: website_formatter_prompt(article))
            | get_gemini()
            | StrOutputParser()
        )

        website = chain.invoke({"article": article})

    except Exception:
        print("Gemini failed → Using Groq")

        chain = (
            RunnableLambda(lambda article: website_formatter_prompt(article))
            | get_groq()
            | StrOutputParser()
        )

        website = chain.invoke({"article": article})

    # Remove thinking blocks if present
    try:
      website = re.sub(r"<think>.*?</think>", "", website, flags=re.DOTALL)
    except Exception:
      pass


    # REMOVE MARKDOWN BLOCKS
    website = website.replace("```html", "")
    website = website.replace("```css", "")
    website = website.replace("```js", "")
    website = website.replace("```", "")

    # Extract HTML, CSS, JS safely
    try:

        html = website.split("--html--")[1].split("--html--")[0]
        css = website.split("--css--")[1].split("--css--")[0]
        js = website.split("--js--")[1].split("--js--")[0]

    except Exception:

        print("Error parsing website output")
        print(website)
        return


    # Save files
    with open("index.html", "w") as f:
        f.write(html.strip())

    with open("style.css", "w") as f:
        f.write(css.strip())

    with open("script.js", "w") as f:
        f.write(js.strip())


    # Zip files
    with zipfile.ZipFile("website.zip", "w") as zipf:
        zipf.write("index.html")
        zipf.write("style.css")
        zipf.write("script.js")

    print("Website Generated")


