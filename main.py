__author__ = 'Rahul Anand'

from audio_processing import AudioProcessingHandler
from formatter import TranscriptAlignmentFormatter
from ques_and_answer import QuestionAnsweringHandler
from data import store_response, get_text_from_mongodb
from information_extractor import PriceInfoExtractor
from fastapi import FastAPI, UploadFile, File, Form
from audio_processing import AudioProcessingHandler
from formatter import TranscriptAlignmentFormatter
from ques_and_answer import QuestionAnsweringHandler
from typing import List

app = FastAPI()

audio_handler = AudioProcessingHandler("base", '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')
alignment_formatter = TranscriptAlignmentFormatter(audio_handler)
qa_handler = QuestionAnsweringHandler()
price_extractor = PriceInfoExtractor()

# Process an audio file
audio_path = r"/InboundSampleRecording.wav"
# script = audio_handler.transcribe(audio_path)
# diarized = audio_handler.diarize(audio_path)
# script_aligned = alignment_formatter.align_transcript(script, audio_path, diarized)
# formatted_transcript = alignment_formatter.format_conversation(script_aligned)

# print(formatted_transcript)

# question = "How much discount was offered to Carolyn?"
# bert_answer = qa_handler.get_answer(question, formatted_transcript)
# print(bert_answer)
#
# price_information = price_extractor.extract_price_related_info(formatted_transcript)
# print(price_information)


@app.post("/process_audio/")
async def process_audio(files: List[UploadFile] = File(...)):
    responses = []
    for file in files:
        contents = await file.read()
        # Assuming you're saving the file temporarily for processing
        with open(file.filename, "wb") as f:
            f.write(contents)

        # Process the file
        script = audio_handler.transcribe(file.filename)
        diarized = audio_handler.diarize(file.filename)
        script_aligned = alignment_formatter.align_transcript(script, file.filename, diarized)
        formatted_transcript = alignment_formatter.format_conversation(script_aligned)

        # Append the result to the responses list
        responses.append(formatted_transcript)
        store_response(file.filename, formatted_transcript, "mongodb://localhost:27017/")
    return {"responses": responses}

@app.post("/get_answer/")
async def get_answer(question: str = Form(...), file_name: str = Form(...)):
    # Get an answer to the question based on the provided context
    context = get_text_from_mongodb(file_name=file_name, mongo_uri= "mongodb://localhost:27017/")
    answer = qa_handler.get_answer(question, context)
    return {"question": question, "answer": answer}

@app.post("/extract_price_info/")
async def extract_price_info(file_name: str = Form(...)):
    formatted_transcript = get_text_from_mongodb(file_name=file_name, mongo_uri= "mongodb://localhost:27017/")
    extracted_info = price_extractor.extract_price_related_info(formatted_transcript)
    return {"extracted_information": extracted_info}