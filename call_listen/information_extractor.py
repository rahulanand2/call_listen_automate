__author__ = 'Rahul Anand'

import spacy
from spacy.matcher import Matcher


class PriceInfoExtractor:
    """
    A class for extracting price-related information from text using spaCy's NLP capabilities.

    Attributes:
        nlp (spacy.language.Language): spaCy language model for processing text.
        matcher (spacy.matcher.Matcher): Matcher for finding price-related terms in text.
    """

    def __init__(self):
        """
        Initializes the PriceInfoExtractor with a spaCy language model and a matcher for price-related terms.
        """
        self.nlp = spacy.load("en_core_web_sm")
        self.matcher = Matcher(self.nlp.vocab)
        patterns = [[{"LOWER": "price"}], [{"LOWER": "cost"}], [{"LOWER": "discount"}], [{"LOWER": "charge"}],
                    [{"LOWER": "pay"}]]
        self.matcher.add("PriceDiscountPatterns", patterns)

    def extract_price_related_info(self, text):
        """
        Extracts price-related conversations from the given text.

        Args:
            text (str): The text from which price-related information is to be extracted.

        Returns:
            list: A list of strings containing price-related conversations.
        """
        price_related_conversations = []
        current_speaker = None
        current_text = ""

        # Iterate over sentences and extract relevant ones
        for line in text.split('\n'):
            if line.startswith('SPEAKER'):
                if current_text and self.matcher(self.nlp(current_text)):
                    price_related_conversations.append(f"{current_speaker}: '{current_text.strip()}'")
                    current_text = ""
                current_speaker = line.split(':')[0]
                current_text = line.split(': ')[1]
            else:
                current_text += " " + line

        # Add the last accumulated text if it's relevant
        if current_text and self.matcher(self.nlp(current_text)):
            price_related_conversations.append(f"{current_speaker}: '{current_text.strip()}'")

        return price_related_conversations
