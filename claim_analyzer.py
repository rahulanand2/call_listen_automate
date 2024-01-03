__author__ = 'Rahul Anand'

import re
import tensorflow_hub as hub
import tensorflow as tf
import spacy
from sentence_transformers import SentenceTransformer, util
import torch
from ques_and_answer import QuestionAnsweringHandler


class ClaimAnalyzer(QuestionAnsweringHandler):
    def __init__(self):
        super().__init__()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        # self.model = SentenceTransformer('bert-large-uncased')
        # self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        self.nlp = spacy.load("en_core_web_sm")  # Load Spacy model for dependency parsing
        self.price_segments = None
        self.discount_segments = None
        self.product_details = None

    def analyze_claim(self, claim, transcript):

        try:
            claim_embedding = self.model.encode(claim, convert_to_tensor=True)
            transcript_segments = transcript.split('. ')
            transcript_embeddings = self.model.encode(transcript_segments, convert_to_tensor=True)
            cos_scores = util.pytorch_cos_sim(claim_embedding, transcript_embeddings)[0]
            top_result_idx = torch.argmax(cos_scores)

            extracted_info = self.extract_important_info(transcript)

        except:
            if not isinstance(claim, list):
                claim = [claim]

            # Similarly, ensure transcript is split into a list of sentences
            transcript_segments = transcript.split('. ')
            transcript_segments = [seg for seg in transcript_segments if seg.strip()]
            claim_embedding = self.model(claim)
            transcript_embeddings = self.model(transcript_segments)
            claim_embedding = tf.nn.l2_normalize(claim_embedding, axis=1)
            transcript_embeddings = tf.nn.l2_normalize(transcript_embeddings, axis=1)
            cos_scores = tf.matmul(claim_embedding, transcript_embeddings, transpose_b=True)

            # Find the segment with the highest similarity score
            top_result_idx = tf.argmax(cos_scores, axis=1).numpy()[0]

            # Check if top_result_idx is within the bounds of transcript_segments
            if top_result_idx >= len(transcript_segments):
                return None, None, None

            # # Enhanced analysis with custom rules
            # extracted_info = self.extract_important_info(transcript)

        return transcript_segments[top_result_idx], cos_scores[top_result_idx].item()

    def extract_important_info(self, transcript):
        """
        Extracts important information from the transcript.

        Args:
            transcript (str): The conversation transcript.

        Returns:
            dict: Extracted information including price, discount, contract terms, and product details.
        """
        info = {
            "prices": [],
            "price_segments": [],
            "discounts": [],
            "discount_segments": [],
            "contract_terms": [],
            "product_details": []
        }
        keywords_price = ('price', 'money', 'amount', 'cost', 'pay')

        for segment in transcript.split('. '):
            doc = self.nlp(segment)

            # Extract prices
            for ent in doc.ents:
                if ent.label_ == "MONEY":
                    info["prices"].append(ent.text)

            if any(keyword in segment.lower() for keyword in keywords_price):
                prices = re.findall(r'\$\d+(?:\.\d{2})?', segment)
                info["price_segments"].append(segment)

            # Extract discounts
            if 'discount' in segment.lower():
                discount_values = re.findall(r'\b\d+%|\b\d+\.\d+%', segment)
                info["discounts"].extend(discount_values)
                info["discount_segments"].append(segment)

            # Extract contract terms and product details using custom logic
            if 'contract' in segment.lower() or 'subscription' in segment.lower():
                info["contract_terms"].append(segment)
            if 'product' in segment.lower() or 'package' in segment.lower() or 'trial' in segment.lower():
                info["product_details"].append(segment)

        # Aggregating and analyzing the discount information
        final_discount = self.aggregate_discounts(info["discounts"])
        info["aggregated_discount"] = final_discount

        self.price_segments = info['price_segments']
        self.discount_segments = info['discount_segments']
        self.product_details = info['product_details']
        return info

    def aggregate_discounts(self, discounts):
        """
        Aggregates discount values to calculate the final discount.

        Args:
            discounts (list): A list of discount strings.

        Returns:
            str: Aggregated discount value.
        """
        total_discount = 0
        for discount in discounts:
            discount_value = float(discount.strip('%'))
            total_discount += discount_value

        if total_discount > 100:
            total_discount = 100  # Cap the discount at 100%

        return f"{total_discount}%"

    def analyse_claim_segments(self, question, transcript, segment='price'):
        '''

        :return:
        '''

        self.extract_important_info(transcript=transcript)

        # Ensure context is a single string
        context = ' '.join(self.price_segments) if isinstance(self.price_segments, list) else self.price_segments

        if 'price' in segment:
            answer = self.get_answer(question=question, context=context)
        elif 'discount' in segment:
            answer = self.get_answer(question=question, context=context)

        return answer

    def price_discount_conversation(self):
        '''

        :return:
        '''

        return self.price_segments, self.discount_segments, self.product_details
