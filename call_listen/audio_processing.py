__author__ = 'Rahul Anand'
from whisperx.diarize import DiarizationPipeline
import whisper
import torch

class AudioProcessingHandler:
    """
    A class to handle audio processing tasks including transcription and diarization.

    Attributes:
        device (torch.device): Device on which the models will run.
        whisper_model (whisper.Model): Whisper model for audio transcription.
        diarization_pipeline (DiarizationPipeline): Pipeline for speaker diarization.
    """
    def __init__(self, model_name="base", token=None, device=None):
        """
        Initializes the AudioProcessingHandler with the specified Whisper model and diarization pipeline.

        Args:
            model_name (str): Name of the Whisper model to be used for transcription.
            token (str): Authentication token for Hugging Face models (used for diarization).
            device (torch.device): Device to run the models on. Defaults to GPU if available, else CPU.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.whisper_model = whisper.load_model(model_name, self.device)
        self.diarization_pipeline = DiarizationPipeline(use_auth_token=token) if token else None
        self.diarized_output = None

    def transcribe(self, audio_path):
        """
        Transcribes the given audio file using the Whisper model.

        Args:
            audio_path (str): Path to the audio file to be transcribed.

        Returns:
            dict: A dictionary containing transcription results.
        """
        return self.whisper_model.transcribe(audio_path)

    def diarize(self, audio_path):
        """
        Performs speaker diarization on the given audio file.

        Args:
            audio_path (str): Path to the audio file for diarization.

        Returns:
            dict: A dictionary containing diarization results.

        Raises:
            ValueError: If the diarization token is not provided.
        """
        if self.diarization_pipeline:
            return self.diarization_pipeline(audio_path)
        else:
            raise ValueError("Diarization token not provided")