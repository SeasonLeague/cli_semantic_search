import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import time
import sys
import threading
import queue
import random

class TextStreamer:
    """A simple text streaming class to mimic the behavior of TextStreamer in transformers."""
    
    def __init__(self, output=sys.stdout, stream_interval=0.01):
        self.output = output
        self.stream_interval = stream_interval
        self.text_queue = queue.Queue()
        self.streaming = False
        self.stream_thread = None
    
    def _stream_text(self):
        while self.streaming:
            try:
                text_chunk = self.text_queue.get(block=False)
                for char in text_chunk:
                    self.output.write(char)
                    self.output.flush()
                    # Randomize slightly for more natural feel
                    delay = self.stream_interval * (0.5 + random.random())
                    time.sleep(delay)
                self.text_queue.task_done()
            except queue.Empty:
                time.sleep(0.01)
    
    def start(self):
        self.streaming = True
        self.stream_thread = threading.Thread(target=self._stream_text)
        self.stream_thread.daemon = True
        self.stream_thread.start()
    
    def stop(self):
        self.streaming = False
        if self.stream_thread:
            self.stream_thread.join(timeout=1.0)
    
    def put(self, text):
        self.text_queue.put(text)
    
    def wait_until_done(self):
        self.text_queue.join()


class SimpleSemanticSearch:
    def __init__(self):
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('punkt')
        
        self.vectorizer = TfidfVectorizer(
            stop_words=stopwords.words('english'),
            lowercase=True,
            norm='l2'
        )
        
        self.documents = []
        self.document_vectors = None
    
    def add_documents(self, documents, ids=None):
        if ids is None:
            start_idx = len(self.documents)
            ids = list(range(start_idx, start_idx + len(documents)))
        
        self.documents.extend(list(zip(ids, documents)))
        
        self._update_vectors()
    
    def _update_vectors(self):
        texts = [doc[1] for doc in self.documents]
        self.document_vectors = self.vectorizer.fit_transform(texts)
    
    def search(self, query, top_k=5):
        query_vector = self.vectorizer.transform([query])
        
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            doc_id, doc_text = self.documents[idx]
            score = similarities[idx]
            results.append((doc_id, doc_text, score))
        
        return results


class StreamingFAQChatbot:
    
    def __init__(self, faq_data, confidence_threshold=0.3, stream_speed=0.03):
        self.faq_data = faq_data
        self.confidence_threshold = confidence_threshold

        self.questions = [item["question"] for item in faq_data]
        self.answers = [item["answer"] for item in faq_data]
        
        self.search_engine = SimpleSemanticSearch()
        self.search_engine.add_documents(self.questions)
        
        self.streamer = TextStreamer(stream_interval=stream_speed)
        self.streamer.start()
    
    def respond(self, query):
        results = self.search_engine.search(query, top_k=1)
        
        if not results or results[0][2] < self.confidence_threshold:
            response = "I'm sorry, I don't understand your question. Could you rephrase it?"
            self.streamer.put(response)
            return None
        
        doc_id, matched_question, score = results[0]
        
        answer = self.answers[doc_id]
        self.streamer.put(answer)
        
        return {
            "matched_question": matched_question,
            "confidence": score,
            "answer": answer
        }
    
    def close(self):
        self.streamer.stop()


if __name__ == "__main__":
    # Sample customer support FAQ data
    faq_data = [
        {
            "question": "How do I reset my password?",
            "answer": "You can reset your password by clicking the 'Forgot Password' link on the login page. You will receive an email with instructions to create a new password."
        },
        {
            "question": "What payment methods do you accept?",
            "answer": "We accept Visa, Mastercard, American Express, PayPal, and bank transfers. All payments are processed securely through our payment gateway."
        },
        {
            "question": "How long does shipping take?",
            "answer": "Standard shipping takes 3-5 business days within the continental US. International shipping typically takes 7-14 business days depending on the destination country."
        },
        {
            "question": "Can I return a product if I'm not satisfied?",
            "answer": "Yes, we offer a 30-day satisfaction guarantee. If you're not happy with your purchase, you can return it within 30 days for a full refund or exchange."
        },
        {
            "question": "How do I track my order?",
            "answer": "You can track your order by logging into your account and viewing your order history. Alternatively, you can use the tracking number provided in your shipping confirmation email."
        },
        {
            "question": "Do you offer discounts for bulk orders?",
            "answer": "Yes, we offer volume discounts for orders over $500. Please contact our sales team for a custom quote based on your requirements."
        },
        {
            "question": "How can I change my delivery address?",
            "answer": "You can update your delivery address before your order ships by contacting our customer service team. Once an order has shipped, the delivery address cannot be changed."
        },
        {
            "question": "What is your refund policy?",
            "answer": "Refunds are processed within 5-7 business days after we receive the returned item. The refund will be issued to the original payment method used for the purchase."
        }
    ]
    
    print("Streaming FAQ Chatbot Demo")
    print("-------------------------")
    print("Type 'quit' to exit\n")
    
    chatbot = StreamingFAQChatbot(faq_data, stream_speed=0.02)
    
    try:
        while True:
          
            query = input("You: ")
            
            if query.lower() in ["quit", "exit", "bye"]:
                print("\nThank you for using our support chatbot!")
                break
            
            print("\nBot: ", end="")
            
            
            match_info = chatbot.respond(query)
            
           
            chatbot.streamer.wait_until_done()
            
           
            if match_info:
                print(f"\n\n(Matched: '{match_info['matched_question']}' with confidence {match_info['confidence']:.2f})")
            print()
    
    finally:
        chatbot.close()