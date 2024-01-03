# Call Listening Automation 
This project involves a speech-to-text pipeline where audio data is transcribed and aligned with a script. Ask questions to confirm facts and figures. Generate all price related conversation snippets.

## Description
This project is an audio analysis and processing system built in Python, utilizing machine learning and natural language processing techniques. It is designed to transcribe, diarize, and extract meaningful information from audio files. Key features include audio transcription using the Whisper model, speaker diarization, alignment, extraction of specific information from transcribed text, and a question-answering functionality based on transformer models.

## Key Features
- **Audio Transcription:** Converts spoken words in audio files to text using the Whisper model.
- **Speaker Diarization:** Identifies different speakers in the audio.
- **Text Alignment:** Aligns the transcribed text with respective speakers.
- **Information Extraction:** Extracts specific information like price-related conversations from transcribed text.
- **Question and Answering System:** Answers questions based on the transcribed context using transformer models.

## Key Components
- **AudioProcessing.py:** Handles audio operations like noise reduction.
- **ClaimAnalyzer.py:** Analyzes claims made in conversations.
- **Data.py:** Manages data-related functionalities.
- **Formatter.py:** Formats and aligns the transcript with speaker data.
- **InformationExtractor.py:** Extracts specific information from the transcript.
- **Main.py:** The main script integrating all components.
- **QuesAndAnswer.py:** Implements a question-answering system based on the transcript.
- **Requirements.txt:** Lists all Python package dependencies.

## An overview
- ques_and_answer.py
		Implements QuestionAnsweringHandler.
		Utilizes transformers library for question-answering tasks.

- main.py
		Main entry point for the application.
		Integrates components like AudioProcessingHandler, TranscriptAlignmentFormatter, QuestionAnsweringHandler, ClaimAnalyzer, and PriceInfoExtractor.
		Uses FastAPI for web application functionalities.

- information_extractor.py
		Contains PriceInfoExtractor class.
		Extracts price-related information using spaCy.

- formatter.py
		Includes TranscriptAlignmentFormatter class.
		Focuses on aligning and formatting conversation transcripts.

- data.py
		Manages MongoDB interactions for data handling.
		Functions for storing and retrieving data are included.

- claim_analyzer.py
		Used for analyzing claims or facts in transcripts.
		Integrates TensorFlow, spaCy, and sentence transformers.

- audio_processing.py
		Defines AudioProcessingHandler class.
		Handles audio processing tasks like transcription and speaker diarization.

## Installation
To set up the project, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/rahulanand2/call_listen_automate.git
   
2. Install Python 3.8+.
3. **Install dependencies**
   '''bash
   `pip install -r requirements.txt`
4. Run main.py, respective API's will load on port 8000.

## API Endpoints
1.  **/process_audio/**    
    -   **Functionality:** Processes uploaded audio files.
    -   **Input Parameters:** Audio file.
    -   **Details:** Transcribes audio, performs speaker diarization, aligns the transcript, and formats the conversation. The processed data is stored in MongoDB.
    
2.  **/get_answer/**    
    -   **Functionality:** Provides answers to questions based on the context of a specified audio file.
    -   **Input Parameters:** Audio file ID, question text.
    -   **Details:** Retrieves the transcribed text from MongoDB and uses the QuestionAnsweringHandler to find answers.
3.  **/extract_price_info/**
    
    -   **Functionality:** Extracts price-related information from a specified audio file's transcript.
    -   **Input Parameters:** Audio file ID.
    -   **Details:** Fetches the transcript from MongoDB and uses the PriceInfoExtractor for price information extraction.
4.  **/check_claim**
    
    -   **Functionality:** Analyzes claims in a specified segment of an audio file's transcript.
    -   **Input Parameters:** Audio file ID, segment of text for claim analysis.
    -   **Details:** Retrieves the relevant transcript segment from MongoDB and uses the ClaimAnalyzer for claim assessment.