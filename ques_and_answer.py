__author__ = 'Rahul Anand'
from transformers import pipeline


class QuestionAnsweringHandler:
    """
    A class for handling question answering tasks using transformer models.

    Attributes:
        pipeline (transformers.pipeline): The pipeline for the question-answering task.
    """

    def __init__(self, model_name="bert-large-uncased-whole-word-masking-finetuned-squad"):
        """
        Initializes the QuestionAnsweringHandler with a specified transformer model.

        Args:
            model_name (str): The name of the transformer model to be used for question answering.
        """
        self.pipeline = pipeline("question-answering", model=model_name)

    def get_answer(self, question, context):
        """
        Retrieves an answer to a question based on the provided context using the transformer model.

        Args:
            question (str): The question for which an answer is sought.
            context (str): The context or passage where the answer may be found.

        Returns:
            str: The answer to the question derived from the context.
        """
        return self.pipeline(question=question, context=context)['answer']