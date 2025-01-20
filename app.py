import subprocess
import base64
import io
import streamlit as st
from annotated_text import annotated_text
from streamlit import session_state as ss
import PyPDF2
import spacy
from spacy.util import filter_spans
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
import re
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from rake_nltk import Rake
import json
import plotly.graph_objects as go
import pdfkit
import markdown2
import cohere
import numpy as np
from PIL import Image



# Download the spaCy model
nltk.download('stopwords')
nltk.download('punkt_tab')

# Load API keys
openai_key=st.secrets["openai_api_key"]
cohere_key=st.secrets["cohere_api_key"]

def extract_text_from_pdf(pdf):
    """Extract text from a PDF file using PyPDF2."""
    reader = PyPDF2.PdfReader(pdf)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

def extract_keywords(text):
    """Extract keywords from the input text using RAKE."""
    r = Rake(stopwords=stopwords.words('english'))
    r.extract_keywords_from_text(text)
    return r.get_ranked_phrases()

def extract_skills(text):
    """Extract skills from the input text using a pre-defined list of skills."""
    with open('data/skills.json') as f:
        skills_list = json.load(f)
    
    processed_text = preprocess_text(text)
    processed_text = ' '.join(processed_text)

    nlp = spacy.load("en_core_web_sm")
    skill_patterns = list(nlp.pipe(skills_list))
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("SKILL", skill_patterns)

    # Define the custom spaCy component to match skills
    @Language.component("skill_component")
    def skill_component(doc):
        """ Custom spaCy pipeline component to match skills in the text."""
        matches = matcher(doc)
        spans = [Span(doc, start, end, label="SKILL") for match_id, start, end in matches]
        spans = filter_spans(spans)
        doc.ents = spans
        return doc

    # Add the custom component to the pipeline after the NER component
    nlp.add_pipe("skill_component", after="ner")
    doc = nlp(processed_text)
    unique_skills = set(ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL")
    return list(unique_skills)

@st.cache_data(show_spinner=False)
def calculate_skill_ratio(resume_skills, job_description_skills):
    """Calculate the skill matching ratio between a resume and a job description."""
    matched_skills = set(resume_skills).intersection(set(job_description_skills))
    total_job_skills = len(job_description_skills)
    skill_ratio = (len(matched_skills) / total_job_skills) * 100 if total_job_skills > 0 else 0
    return skill_ratio

@st.cache_data(show_spinner=False)
def get_overall_score(resume_text, job_description_text):
    """Calculate the ATS score between a resume and a job description with weighted components."""
    # Extract keywords from both resume and job description
    resume_keywords = extract_keywords(resume_text)
    job_description_keywords = extract_keywords(job_description_text)

    # Preprocess extracted keywords
    resume_tokens = preprocess_text(' '.join(resume_keywords))
    job_tokens = preprocess_text(' '.join(job_description_keywords))

    # Transform in a single string
    resume_tokens = ' '.join(resume_tokens)
    job_tokens = ' '.join(job_tokens)

    # Extract skills from both resume and job description
    resume_skills = extract_skills(resume_text)
    job_description_skills = extract_skills(job_description_text)

    # Calculate matching score for keywords
    keyword_match_score = get_kw_score(resume_tokens, job_tokens)*100

    # Calculate skill ratio
    skill_ratio = calculate_skill_ratio(resume_skills, job_description_skills)

    # Weighted ATS score (50% keywords, 50% skills)
    ats_score = (0.50 * keyword_match_score) + (0.50 * skill_ratio)

    return ats_score

def get_resume_and_comments(text, delimiter='---'):
    """Extract the resume and comments from the improved response."""
    match = re.match(r'^(.*?)' + re.escape(delimiter) + r'(.*)', text, re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None

@st.cache_data(show_spinner=False)
def get_kw_score(resume_string, job_description_string):
    """
    Calculate the similarity score between a resume and a job description using pre-trained embeddings.
    """
    emb_model = SentenceTransformer("all-mpnet-base-v2")
    resume_embedding = emb_model.encode(resume_string)
    jd_embedding = emb_model.encode(job_description_string)

    resume_embedding = np.array(resume_embedding)
    jd_embedding = np.array(jd_embedding)

    cosine_similarity = np.dot(resume_embedding, jd_embedding) / (np.linalg.norm(resume_embedding) * np.linalg.norm(jd_embedding))

    return cosine_similarity

@st.cache_data(show_spinner=False)
def get_gpt_response(messages, model="gpt-4o-mini", temperature=0.2, top_p=0.1):
    """ Get a response from the OpenAI GPT-4 model."""

    client = OpenAI(
        api_key=openai_key,
    )

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )

    return chat_completion.choices[0].message.content

@st.cache_data(show_spinner=False)
def get_cohere_response(messages, model="command-r-plus-08-2024", temperature=0.3, top_p=0.3):
    """ Get a response from the Cohere model."""

    co = cohere.ClientV2(cohere_key)

    response = co.chat(
        model=model,
        messages=messages,
        temperature=temperature,
        p=top_p,)

    return response.message.content[0].text

@st.cache_data(show_spinner=False)
def get_messages(resume_text, job_description, keywords, skills_keep, skills_add):
    """Prepare the messages for the LLM model."""

    messages = [
    {
        "role": "system",
        "content": """
You are a resume analysis expert tasked with optimizing a candidate’s resume to align with a given job description.
The goal is to ensure the resume mirrors the job description's wording and relevant terms, which increases its chances of passing through Applicant Tracking Systems (ATS) without inventing or adding false details.

Your responsibilities include:

1. **Keyword Matching**:
   - Create a new version of the resume, using keywords and extracts from the job description, ensuring that the resume mirrors the language of the job description.
   - You are going to receive a list of skills, some are to keep, and others to add to the resume.
   - Do not add the new skills to the skills section, ensure that these words are naturally integrated into the resume, at least in one of these sections: summary, work experience, or projects.
   - Modify the wording to match the job description as closely as possible, considering the candidate's experience and qualifications. Example:
      - If the resume mentions "dashboard" and the job description uses "data visualization", you should replace "dashboard" with "data visualization." or at least add "data visualization" to the resume.
   - You can modify any section of the resume to better align with the job description.

2. **Formatting**:
   - Follow a structured Markdown format for the revised resume using headers (`#`, `##`, `###`) for each section.
   - The skills section should only include hard skills.

The resume must be returned in the following Markdown structure:
    
# John Doe
Toronto, ON | +1234567890 | **[john.doe@example.com](john.doe@example.com)** | **[linkedin.com/in/johndoe](https://linkedin.com/in/johndoe)**

### SUMMARY
Experienced software engineer with 5+ years in backend development...

### WORK EXPERIENCE
#### **Senior Software Engineer**, ABC Corp (Toronto, ON) | **Jan 2020 - Present**
- Led a team of 5 engineers...
- Improved system performance by 25%...

#### **Software Developer**, XYZ Inc. (Vancouver, BC) | **Mar 2017 - Dec 2019**
- Developed a high-traffic e-commerce platform...

### SKILLS [one single bullet point with comma-separated skills]
- Python, SQL, Django, Flask, AWS, GCP, MySQL, PostgreSQL, Data Storytelling, Data Visualization

### EDUCATION
**B.Sc. in Computer Science**, University of Technology (Vancouver, BC) | **2016**

### PROJECTS
#### E-commerce Platform (2020)
- **Tech**: Python, Django, AWS
- **Description**: Built a scalable e-commerce platform...

### CERTIFICATIONS
- AWS Certified Solutions Architect - Associate (2019)   

3. **Resume Feedback**:
   - At the end of the resume, add a delimiter (---) and provide feedback. Consider the revised resume and the job description for the analysis.
     - ### Strong points: where the new resume aligns well with the job description.
     - ### Weak points: what are the weaknesses of the new resume that the candidate needs to address to close gaps in qualifications or skills.
     - ### Suggestions for improvement: providing constructive feedback on how the candidate can enhance their resume to better match the job requirements.
   
"""
    },
    {
        "role": "user",
        "content": f"""
Please extract the relevant information from the following resume text and adjust the wording based on the provided job description, keyword list, and skills list:

**Resume:**
{resume_text}

**Job Description:**
{job_description}

**Keywords extracted from Job Description:**
{keywords}

**Skills to keep in the resume:**
{skills_keep}

**Skills to include in the resume:**
{skills_add}
"""
    }
]

    return messages

def load_css(file_name):
    """ Load external CSS file."""
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Streamlit app
# Create session state manager
if 'stage' not in ss:
    ss.stage = 0
def set_stage(stage):
    """ Set the stage of the application."""
    ss.stage = stage

# Set page configuration
st.set_page_config(
    page_title="resumego. increase your matches",
    page_icon="assets/images/favicon.ico",
    layout="wide",
)

# Load custom CSS
load_css('css/styles.css')

st.image("assets/images/resumego_logo_app.png", width=300)

st.markdown("# Does your resume show your true potential to recruiters?")
st.write("Provide us your resume and job description and watch as **resumego.** closes the gap for the perfect fit.")
st.markdown("## 1. Upload your resume")

uploaded_file = st.file_uploader("", type="pdf")
if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)

st.markdown("## 2. Paste the job description")
job_description = st.text_area("", height=200)

st.button("**Fine-tune my resume**", on_click=set_stage, args = (1,))
if ss.stage > 0:
    if resume_text and job_description:
        keywords_jd = extract_keywords(job_description)
        job_skills = extract_skills(job_description)
        resume_skills = extract_skills(resume_text)
        missing_skills = [skill for skill in job_skills if skill not in resume_skills]
        st.markdown("## 3. Select the skills to add to your resume")
        skills_include = st.multiselect("", missing_skills, missing_skills)
        st.button("Continue", on_click=set_stage, args = (2,))
        if ss.stage > 1:
            messages = get_messages(resume_text, job_description, keywords_jd, resume_skills, skills_include)
            improved_response = get_cohere_response(messages)
            improved_resume, comments_resume = get_resume_and_comments(improved_response)


            st.markdown(
                """
                <style>
                .resume-container {
                    background-color: #f7f7f7;
                    padding: 15px;
                    border-radius: 10px;
                    font-family: Arial, sans-serif;
                }
                </style>
                """, 
                unsafe_allow_html=True
    )

            st.markdown("## 4. Your improved resume")

            st.markdown(
        '''
        <style>
        .streamlit-expanderHeader {
            background-color: rgb(250, 250, 250);
            color: black; # Adjust this for expander header color
        }
        .streamlit-expanderContent {
            background-color: rgb(250, 250, 250);
            color: black; # Expander content color
        }
        </style>
        ''',
        unsafe_allow_html=True
    )
            with st.expander("Your fine-tuned resume is ready! Hit the **Download** button below to get it.",expanded=True):
                st.markdown(improved_resume)
            # Convert the Markdown to HTML
            html_body = markdown2.markdown(improved_resume)
            wkhtmltopdf_path = subprocess.check_output(['which', 'wkhtmltopdf'], universal_newlines=True).strip()
            # Configure pdfkit to use the binary
            config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)
            options = {
            'margin-top': '15mm',
            'margin-bottom': '15mm',
            'margin-left': '15mm',
            'margin-right': '15mm',
            'encoding': 'UTF-8'
            }
            pdf_file_path = f"{uploaded_file.name}_improved.pdf"
            # Generate PDF from HTML
            pdfkit.from_string(html_body, pdf_file_path, options=options)
            # Provide the PDF as a download
            with open(pdf_file_path, 'rb') as pdf_file:
                st.download_button(label="Download", data=pdf_file, file_name=pdf_file_path, mime="application/pdf")
            st.markdown("## 5. A few comments about your resume")
            st.write(comments_resume)
            # Get score between resume and job description using vector embeddings and cosine similarity
            score = get_overall_score(improved_resume, job_description)
            st.markdown("## 6. Performance analysis")
            
            col1, col2 = st.columns([0.4, 0.6])

            # Set the color of the gauge bar based on the score
            if score < 70:
                bar_color = "#e4002b"
            elif 70 <= score < 85:
                bar_color = "#ffbb00"
            else:
                bar_color = "#006400"

            # Plot gauge in the left column
            with col1:
                st.markdown("### Resume score")
                # Create a Plotly gauge figure
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={'suffix': "%", 'font': {'size': 36}},
                    gauge={
                        'axis': {
                            'range': [0, 100],
                            'tickmode': "array",
                            'tickvals': [0, 25, 50, 75, 100],
                            'ticktext': ["0%", "25%", "50%", "75%", "100%"] 
                        },
                        'bar': {'color': bar_color}, 
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "black"
                    }
                ))

                # Define figure size in the layout
                fig.update_layout(
                    width=400,
                    height=180,
                    margin=dict(l=30, r=40, t=20, b=5),
                )

                # Display the gauge in Streamlit
                st.plotly_chart(fig)

            # Skills Comparison in the right column
            with col2:
                # Skills Comparison
                st.markdown("### Skills matching")

                # Extract skills from the resume and job description
                resume_skills = extract_skills(improved_resume)

                # Use annotated_text to highlight matches in green
                resume_annotations = []
                for skill in job_skills:
                    if skill in resume_skills:
                        resume_annotations.appenƒ√d((skill, "match", "#4CAF50"))
                    else:
                        resume_annotations.append((skill, "not matched", "#FF6347"))

                # Display the annotated resume skills
                annotated_text(*resume_annotations)
    else:
        st.error("Please upload a resume and paste a job description.")

bg_logo = Image.open("assets/images/bganal_bw.png")

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

st.markdown(
    f'<br><div style="text-align: center;"><img src="data:image/png;base64,{image_to_base64(bg_logo)}" width="130"></div>',
    unsafe_allow_html=True,
)
