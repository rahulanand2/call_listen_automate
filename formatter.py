__author__ = 'Rahul Anand'
from whisperx import load_align_model, align
import torch
from whisperx.diarize import assign_word_speakers
from audio_processing import AudioProcessingHandler


class TranscriptAlignmentFormatter:
    """
    A class for aligning transcripts with speakers and formatting the conversation transcript.

    Attributes:
        device (torch.device): Device on which the models will run.
    """
    def __init__(self, audio_processor: AudioProcessingHandler, device=None):
        """
        Initializes the TranscriptAlignmentFormatter with the specified device.

        Args:
            device (torch.device): Device to run the models on. Defaults to GPU if available, else CPU.
        """
        self.audio_processor = audio_processor
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def align_transcript(self, script, audio_path, diarized_output):
        """
        Aligns transcript segments with speakers.

        Args:
            script (dict): Transcription result from the Whisper model.
            audio_path (str): Path to the audio file being processed.
            diarized_output (df) : DataFrame of diarized output of each segment

        Returns:
            list: A list of aligned transcript segments.
        """
        model_a, metadata = load_align_model(language_code=script["language"], device=self.device)
        temp_alignment = align(script["segments"], model_a, metadata, audio_path, self.device)

        result_segments, word_seg = list(assign_word_speakers(
            diarized_output, temp_alignment
        ).values())
        transcribed = []

        for result_segment in result_segments:
            # Directly use the 'speaker' attribute from the segment
            speaker = result_segment.get('speaker', 'Unknown Speaker')
            transcribed.append(
                {
                    "start": result_segment["start"],
                    "end": result_segment["end"],
                    "text": result_segment["text"],
                    "speaker": speaker,
                }
            )
        return transcribed

    @staticmethod
    def format_conversation(transcript_list):
        """
        Formats the aligned conversation transcript for readability.

        Args:
            transcript_list (list): A list of transcript segments with speaker information.

        Returns:
            str: A formatted string representing the conversation.
        """
        formatted_conversation = ""
        current_speaker = None
        current_text = ""
        for entry in transcript_list:
            speaker = entry.get('speaker', 'Unknown Speaker')
            text = entry.get('text', '').strip()
            if speaker != current_speaker:
                if current_speaker is not None:
                    formatted_conversation += f"{current_speaker}: '{current_text.strip()}'\n"
                current_text = text
                current_speaker = speaker
            else:
                current_text += ' ' + text
        if current_speaker is not None:
            formatted_conversation += f"{current_speaker}: '{current_text.strip()}'"
        return formatted_conversation
