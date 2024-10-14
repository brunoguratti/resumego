import subprocess
import os
import streamlit as st
from streamlit import session_state as ss
import PyPDF2
import spacy
from spacy.util import filter_spans
from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
import re
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from annotated_text import annotated_text
from openai import OpenAI
from rake_nltk import Rake
import json
import plotly.graph_objects as go
import pdfkit
import markdown2
import cohere


nltk.download('stopwords')
nltk.download('punkt_tab')

nlp = spacy.load("en_core_web_sm")

# Load API keys
qdrant_api = st.secrets["qdrant_api_key"]
openai_key=st.secrets["openai_api_key"]
cohere_key=st.secrets["cohere_api_key"]

# Initialize Qdrant and SentenceTransformer
emb_model = SentenceTransformer("all-mpnet-base-v2")
qdrant_client = QdrantClient(
    url="https://9817dd27-777f-45cb-9bfe-78a2a8e14b88.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=qdrant_api,
)

# Load the list of skills
with open('data/skills.json') as f:
    skills_list = json.load(f)

# Define the skill extraction and keyword extraction functions
def extract_text_from_pdf(pdf):
    reader = PyPDF2.PdfReader(pdf)
    text = ""
    for page in range(len(reader.pages)):
        text += reader.pages[page].extract_text()
    return text

def clean_text(text):
    """Clean the input text by removing stopwords, punctuation, and non-alphabetical characters."""
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))

    # Remove stopwords, punctuation, and non-alphabetical characters
    cleaned_tokens = [word for word in tokens if word not in stop_words and re.match(r'^[a-zA-Z]+$', word)]
    
    return ' '.join(cleaned_tokens)  # Join cleaned tokens back into a string for spaCy processing

def extract_keywords(text):
    """Extract keywords from the input text using RAKE."""
    r = Rake()
    cleaned_text = clean_text(text)
    r.extract_keywords_from_text(cleaned_text)
    return r.get_ranked_phrases()

# Load spaCy's small English model
nlp = spacy.load("en_core_web_sm")

# Create skill patterns and a PhraseMatcher
skill_patterns = list(nlp.pipe(skills_list))
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
matcher.add("SKILL", skill_patterns)

# Define the custom spaCy component to match skills
@Language.component("skill_component")
def skill_component(doc):
    matches = matcher(doc)
    spans = [Span(doc, start, end, label="SKILL") for match_id, start, end in matches]
    spans = filter_spans(spans)
    doc.ents = spans
    return doc

# Add the custom component to the pipeline after the NER component
nlp.add_pipe("skill_component", after="ner")

# Function to extract skills from text
def extract_skills(text):
    doc = nlp(text)
    unique_skills = set(ent.text.lower() for ent in doc.ents if ent.label_ == "SKILL")
    return list(unique_skills)

def get_resume_and_comments(text, delimiter='---'):
    # Regular expression to capture two groups: before and after the delimiter
    match = re.match(r'^(.*?)' + re.escape(delimiter) + r'(.*)', text, re.DOTALL)
    
    # Return the matched groups if found
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return None, None

# Function to get resume score using Qdrant
def get_score(resume_text, job_desc_text):
    resume_embedding = emb_model.encode(resume_text)
    job_desc_embedding = emb_model.encode(job_desc_text)
    
    collection_name="resume_collection"

    qdrant_client.delete_collection(collection_name)

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=rest.VectorParams(size=len(resume_embedding), distance="Cosine"),
    )
    
    qdrant_client.upsert(
        collection_name="resume_collection",
        points=[{"id": 1, "vector": resume_embedding, "payload": {"text": resume_text}}],
    )
    
    search_result = qdrant_client.search(
        collection_name="resume_collection", query_vector=job_desc_embedding, limit=1
    )
    return search_result[0].score

# Function to send resume and job description to OpenAI API for improvement

openai_key=st.secrets["openai_api_key"]

@st.cache_data(show_spinner=False)
def get_gpt_response(messages, model="gpt-4o-mini", temperature=0.2, top_p=0.1):

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
def get_cohere_response(messages, model="command-r-plus-08-2024", temperature=0.2, top_p=0.1):

    co = cohere.ClientV2(cohere_key)

    response = co.chat(
        model=model,
        messages=messages,
        temperature=temperature,
        p=top_p,)

    return response.message.content[0].text

def get_messages(resume_text, job_description, keywords, skills):
    messages = [
    {
        "role": "system",
        "content": """
You are a resume analysis expert tasked with optimizing a candidate’s resume to align with a given job description. The goal is to ensure the resume mirrors the job description's wording and relevant terms, which increases its chances of passing through Applicant Tracking Systems (ATS) without inventing or adding false details.

Your responsibilities include:

1. **Resume Analysis & Extraction**:
   - Review the provided resume text and extract the relevant sections: personal details, summary, work history, education, skills, certifications, and projects.
   - Keep all skills, experiences, and qualifications. **Do not** remove any information.
   - Keep all the links to e-mail, LinkedIn, and other personal websites **intact**.

2. **Keyword Matching**:
   - Review the job description and the provided list of **keywords** and **skills**.
   - Make small **adjustments** to the wording in the resume where the descriptions from the job and resume overlap, ensuring that the resume mirrors the language of the job description **where applicable**. For example:
     - If the job description highlights specific skills like "data visualization" and the resume lists "dashboard," rewrite it to match the job description's phrasing as "data visualization".
     - Focus the adjustments on the summary, work experience, skills and projects sections.
   - **Do not alter, fabricate, or invent new skills or experiences** not present in the original resume. DO NOT add soft skills on the skills list.

3. **Formatting**:
   - Follow a structured Markdown format for the revised resume using headers (`#`, `##`, `###`) for each section.
**Output Format**:

The resume must be returned in the following Markdown structure:
    
## John Doe
Toronto, ON | +1234567890 | **[john.doe@example.com](john.doe@example.com)** | **[linkedin.com/in/johndoe](https://linkedin.com/in/johndoe)**

### SUMMARY
Experienced software engineer with 5+ years in backend development...

### SKILLS
- **Programming language**: Python, SQL | **Frameworks**: Django, Flask
- **Cloud Platforms**: AWS, GCP | **Databases**: MySQL, PostgreSQL

### WORK EXPERIENCE
###### **Senior Software Engineer**, ABC Corp (Toronto, ON) | **Jan 2020 - Present**
- Led a team of 5 engineers...
- Improved system performance by 25%...

###### **Software Developer**, XYZ Inc. (Vancouver, BC) | **Mar 2017 - Dec 2019**
- Developed a high-traffic e-commerce platform...

### EDUCATION
**B.Sc. in Computer Science**, University of Technology (Vancouver, BC) | **2016**

### PROJECTS
###### E-commerce Platform (2020)
- **Tech**: Python, Django, AWS
- **Description**: Built a scalable e-commerce platform...

### CERTIFICATIONS
- AWS Certified Solutions Architect - Associate (2019)   

4. **Strengths and Weak Points**:
   - At the end of the resume, add a delimiter (---) and provide feedback:
     - **Highlight the strong points** where the resume aligns well with the job description.
     - **Critique the weak points**, offering suggestions for improvement where the resume could better match the job description or address gaps in qualifications.
5. **Suggestions for improvement**
   - Provide constructive feedback on how the candidate can enhance their resume to better match the job requirements.
   
"""
    },
    {
        "role": "user",
        "content": f"""
Please extract the relevant information from the following resume text and adjust the wording based on the provided job description, keyword list, and skills list:

**Resume Text:**
{resume_text}

**Job Description:**
{job_description}

**Keyword List:**
{keywords}

**Skills List from Job Description:**
{skills}
"""
    }
]

    return messages

# Streamlit app

## - Create session state manager
if 'stage' not in ss:
    ss.stage = 0

def set_stage(stage):
    ss.stage = stage

st.set_page_config(
    page_title="resumego. increase your matches",  # Replace with your title
    page_icon="💼",  # Replace with your custom favicon or emoji
    layout="wide",  # Other options: "wide"
)
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css('css/styles.css')

st.image("assets/images/resumego_logo_app.png", width=300)

st.markdown("# Does your resume show your true potential to recruiters?")
st.write("Provide us your resume and job description and watch as **resumego.** closes the gap for the perfect fit.")
st.markdown("#### 1. Upload your resume")
# Upload Resume
uploaded_file = st.file_uploader("", type="pdf")
if uploaded_file is not None:
    resume_text = extract_text_from_pdf(uploaded_file)

st.markdown("#### 2. Paste the job description")
# Paste Job Description
job_description = st.text_area("", height=200)

st.button("**go.!**", on_click=set_stage, args = (1,))
if ss.stage > 0:
    if resume_text and job_description:
        # Extract keywords and skills
        keywords_jd = extract_keywords(job_description)
        set_kw_jd = set(keywords_jd)
        skills = extract_skills(job_description)
        keywords_resume = extract_keywords(resume_text)
        set_kw_re = set(keywords_resume)

        # Get prompt
        messages = get_messages(resume_text, job_description, set_kw_jd, skills)

        # Send to GPT for improvement
        improved_response = get_cohere_response(messages)

        # # Get the new resume and comments
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

        # Display the improved resume
        st.markdown("#### 3. Your improved resume")

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
        with st.expander("",expanded=True):
            st.markdown(improved_resume)
        # Button to trigger PDF download
        # Save the HTML as PDF using pdfkit
        html_body = markdown2.markdown(improved_resume)
        # Path to wkhtmltopdf executable
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
        pdf_file_path = 'ImprovedResume.pdf'
        # Generate PDF from HTML
        pdfkit.from_string(html_body, pdf_file_path, options=options)
        # Provide the PDF as a download
        with open(pdf_file_path, 'rb') as pdf_file:
            st.download_button(label="Download", data=pdf_file, file_name="ImprovedResume.pdf", mime="application/pdf")
        st.markdown("#### 4. A few comments about your resume")
        st.write(comments_resume)

        # Get score
        score = get_score(resume_text, job_description)*100
        st.markdown("#### 5. Performance analysis")
        
        # # Create two columns with the specified width
        col1, col2 = st.columns([0.4, 0.6])

        # # Determine the bar color based on the score value
        # if score < 70:
        #     bar_color = "#e4002b"  # Cherry red
        # elif 70 <= score < 85:
        #     bar_color = "#ffbb00"  # Caterpillar yellow
        # else:
        #     bar_color = "#006400"  # Dark green

        # Plot gauge in the left column
        with col1:
            st.markdown("### Resume score")
            # # Create a Plotly gauge figure
            # fig = go.Figure(go.Indicator(
            #     mode="gauge+number",
            #     value=score,
            #     number={'suffix': "%", 'font': {'size': 36}},
            #     gauge={
            #         'axis': {
            #             'range': [0, 100],
            #             'tickmode': "array",
            #             'tickvals': [0, 25, 50, 75, 100],
            #             'ticktext': ["0%", "25%", "50%", "75%", "100%"] 
            #         },
            #         'bar': {'color': bar_color}, 
            #         'bgcolor': "white",
            #         'borderwidth': 2,
            #         'bordercolor': "black"
            #     }
            # ))

            # # Define figure size in the layout
            # fig.update_layout(
            #     width=400,
            #     height=140,
            #     margin=dict(l=20, r=30, t=20, b=5),
            # )

            # # Display the gauge in Streamlit
            # st.plotly_chart(fig)

        # Skills Comparison in the right column
        with col2:
            # Skills Comparison
            st.markdown("### Skills matching")

            # Extract skills from the resume and job description
            resume_skills = extract_skills(improved_resume)
            job_skills = extract_skills(job_description)

            # Use annotated_text to highlight matches in green
            resume_annotations = []
            for skill in job_skills:
                if skill in resume_skills:
                    resume_annotations.append((skill, "match", "#4CAF50"))  # Green for match
                else:
                    resume_annotations.append((skill, "not matched", "#FF6347"))  # Red for non-match

            # Display the annotated resume skills
            annotated_text(*resume_annotations)
    else:
        st.error("Please upload a resume and paste a job description.")
st.write("")
logo = "assets/images/bganal_bw.png"
c1, c2, c3, c4, c5 = st.columns(5)
with c3:
    st.image(logo)