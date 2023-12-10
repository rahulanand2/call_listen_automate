__author__ = 'Rahul Anand'

from audio_processing import AudioProcessingHandler
from formatter import TranscriptAlignmentFormatter
from ques_and_answer import QuestionAnsweringHandler
from information_extractor import PriceInfoExtractor


audio_handler = AudioProcessingHandler("base", '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')
alignment_formatter = TranscriptAlignmentFormatter(audio_handler)
qa_handler = QuestionAnsweringHandler()
price_extractor = PriceInfoExtractor()

# Process an audio file
audio_path = r"C:\Users\rishu\Documents\Call_listening\InboundSampleRecording.wav"
script = audio_handler.transcribe(audio_path)
diarized = audio_handler.diarize(audio_path)
script_aligned = alignment_formatter.align_transcript(script, audio_path, diarized)
formatted_transcript = alignment_formatter.format_conversation(script_aligned)

print(formatted_transcript)

question = "How much discount was offered to Carolyn?"
bert_answer = qa_handler.get_answer(question, formatted_transcript)
print(bert_answer)

price_information = price_extractor.extract_price_related_info(formatted_transcript)
print(price_information)