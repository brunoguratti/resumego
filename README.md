# ResumeGo

**ResumeGo** is a powerful resume enhancement and matching tool designed to optimize resumes for job applications by aligning them with job descriptions. It leverages natural language processing (NLP) and AI models to provide valuable insights and adjustments to maximize your resume's relevance to job postings.

## Features

- **Resume Matching**: Adjusts resume content to better fit the language of job descriptions without altering factual information.
- **Keyword Extraction**: Identifies important keywords, monograms, and bigrams from both resumes and job descriptions to improve relevance.
- **Scoring System**: Provides a quality score for how well the resume aligns with the job description and offers explanations on strengths and areas for improvement.
- **Customizable Resume Analysis**: Users can upload job descriptions to tailor resume adjustments based on specific roles.

## Installation

### Install Dependencies

To install the necessary dependencies, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/brunoguratti/resumego.git
   cd resumego
   ```

2. Set up the conda environment:

   ```bash
   conda create --name resumego python=3.8
   conda activate resumego
   ```

3. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

### Configure API Key

**ResumeGo** requires an API key for Cohere to perform NLP tasks such as text generation and keyword extraction. You need to configure the API key properly:

1. Create a `.streamlit/secrets.toml` file in the root directory of your project with the following content:

   ```toml
   [secrets]
   cohere_api_key = "your-cohere-api-key"
   ```

   Replace `"your-cohere-api-key"` with your actual Cohere API key.

2. Alternatively, you can set the `cohere_api_key` in your environment variables or directly in the code by assigning the key to the `cohere_key` variable:

   ```python
   cohere_key = "your-cohere-api-key"
   ```

### Model Setup

Make sure you download the necessary NLP models:

```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Upload your resume (PDF format) and paste the job description for analysis.
3. View the resume suggestions and quality score based on job description matching.
4. Download the adjusted resume.

## Project Structure

```bash
├── .streamlit/              # Configuration files for Streamlit  
├── assets/images/           # Image assets for the app  
├── css/                     # Custom CSS for styling  
├── data/                    # Folder for storing input and output files  
├── app.py                   # Streamlit application file
├── LICENSE.md               # Project LICENSE file
├── packages.txt             # List of system-level packages for deployment  
├── requirements.txt         # Python dependencies list  
└── README.md                # Project README file
```

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please fork the repository and submit a pull request. Feel free to open issues for bug fixes or feature requests.

## License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. See the [LICENSE](https://github.com/brunoguratti/resumego/LICENSE.ms) file for more details.

## Contact

Feel free to connect with or contact me via [LinkedIn](https://www.linkedin.com/in/brunoguratti/).
