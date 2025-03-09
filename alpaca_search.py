import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from semantic_search import TextStreamer
import json
import os
import sys
import time
import random

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

class AlpacaStreamingKnowledgeBase:
    """A knowledge base assistant that streams responses with thinking steps using Alpaca dataset."""
    
    def __init__(self, alpaca_json_path, stream_speed=0.02, thinking_speed=0.003, max_entries=50000):
        self.alpaca_data = self._load_alpaca_data(alpaca_json_path, max_entries)
        
        self.instructions = [entry.get("instruction", "") for entry in self.alpaca_data]
        self.inputs = [entry.get("input", "") for entry in self.alpaca_data]
        self.outputs = [entry["output"] for entry in self.alpaca_data]
        
        self.search_texts = []
        for instruction, input_text in zip(self.instructions, self.inputs):
            search_text = ""
            if instruction:
                search_text += instruction + " " + instruction
            if input_text:
                search_text += " " + input_text
            
            if not search_text.strip():
                search_text = "Empty entry"
                
            self.search_texts.append(search_text)
        
        self.search_engine = SimpleSemanticSearch()
        self.search_engine.add_documents(self.search_texts)
        
        self.answer_streamer = TextStreamer(stream_interval=stream_speed)
        self.thinking_streamer = TextStreamer(stream_interval=thinking_speed)
        
        self.answer_streamer.start()
        self.thinking_streamer.start()
    
    def _load_alpaca_data(self, json_path, max_entries):
        print(f"Loading Alpaca dataset from {json_path}...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        valid_data = []
        skipped = 0
        
        for entry in data:
            if "output" in entry and entry["output"].strip():
                if "instruction" not in entry:
                    entry["instruction"] = ""
                if "input" not in entry:
                    entry["input"] = ""
                valid_data.append(entry)
            else:
                skipped += 1
        
        if max_entries and max_entries < len(valid_data):
            valid_data = valid_data[:max_entries]
            
        print(f"Loaded {len(valid_data)} valid entries from Alpaca dataset (skipped {skipped} invalid entries)")
        
        print("\nSample entries:")
        for i in range(min(3, len(valid_data))):
            print(f"Entry {i+1}:")
            print(f"  Instruction: {valid_data[i]['instruction'][:50]}...")
            print(f"  Input: {valid_data[i]['input'][:50] if valid_data[i]['input'] else 'None'}")
            print(f"  Output length: {len(valid_data[i]['output'])} chars")
            print()
            
        return valid_data
    
    def _generate_thinking_steps(self, query, results):
        thinking_steps = [
            f"Analyzing query: '{query}'",
            "Searching knowledge base...",
            f"Found {len(results)} relevant examples",
            "Ranking results by relevance...",
            "Extracting key information...",
            "Formulating response..."
        ]
        
        for i, (doc_id, _, score) in enumerate(results[:3]):
            entry_desc = self._get_entry_description(doc_id)
            thinking_steps.insert(3 + i, f"Found relevant entry: '{entry_desc}' (Relevance: {score:.2f})")
        
        return thinking_steps
    
    def _get_entry_description(self, doc_id):
        if self.instructions[doc_id]:
            desc = self.instructions[doc_id][:50]
            if len(self.instructions[doc_id]) > 50:
                desc += "..."
            return desc
        elif self.inputs[doc_id]:
            desc = self.inputs[doc_id][:50]
            if len(self.inputs[doc_id]) > 50:
                desc += "..."
            return f"Input: {desc}"
        else:
            return "Content entry"
    
    def respond(self, query, show_thinking=True):
        processed_query = query
        
        if query.endswith('?'):
            processed_query = query[:-1]  
        
        results = self.search_engine.search(processed_query, top_k=5)
        
        min_score = 0.2 if len(query.split()) < 5 else 0.3
        
        if not results or results[0][2] < min_score:
            response = "I don't have enough information to answer that question accurately. Could you try rephrasing or asking something else?"
            self.answer_streamer.put(response)
            return None
        
        if show_thinking:
            thinking_steps = self._generate_thinking_steps(query, results)
            print("\nThinking: ", end="")
            
            for step in thinking_steps:
                self.thinking_streamer.put(step + "... ")
                self.thinking_streamer.wait_until_done()
                time.sleep(0.2 + random.random() * 0.3)  
            
            print("\n\nAnswer: ", end="")
        
        top_results = []
        for doc_id, _, score in results[:3]:
            if score >= min_score:
                top_results.append({
                    "instruction": self.instructions[doc_id],
                    "input": self.inputs[doc_id],
                    "output": self.outputs[doc_id],
                    "score": score,
                    "entry_type": self._determine_entry_type(doc_id)
                })
        
        answer = self._format_answer(query, top_results)
        self.answer_streamer.put(answer)
        
        related_topics = []
        for result in top_results[1:3]:
            if result["instruction"]:
                topic = result["instruction"][:40] + "..." if len(result["instruction"]) > 40 else result["instruction"]
                related_topics.append(topic)
            elif result["input"]:
                input_snippet = result["input"][:40] + "..." if len(result["input"]) > 40 else result["input"]
                related_topics.append(input_snippet)
        
        return {
            "query": query,
            "answer": answer,
            "related_topics": related_topics,
            "top_score": top_results[0]["score"] if top_results else 0
        }
    
    def _determine_entry_type(self, doc_id):
        has_instruction = bool(self.instructions[doc_id].strip())
        has_input = bool(self.inputs[doc_id].strip())
        
        if has_instruction and not has_input:
            return "question_answer"  
        elif has_instruction and has_input:
            return "instruction_with_input"  
        elif not has_instruction and has_input:
            return "input_only"  
        else:
            return "output_only"  
    
    def _format_answer(self, query, results):
       
        if not results:
            return "I don't have enough relevant information to answer that or go and use chatGPT :)."
        
        top_result = results[0]
        answer = top_result["output"]
        score = top_result["score"]
        
        is_question = any(query.lower().startswith(q) for q in ["what", "how", "why", "when", "where", "who", "which", "is", "are", "can", "do", "does"]) or query.endswith("?")
        
        if top_result["entry_type"] == "question_answer" and score > 0.6:
            return answer
            
        elif top_result["entry_type"] == "instruction_with_input":
            if score > 0.7:
                return answer
            else:
                instruction = top_result["instruction"]
                if query.lower() in instruction.lower():
                    return answer
                else:
                    context = f"Based on a similar request: '{instruction}'"
                    return f"{context}\n\n{answer}"
                
        elif is_question and score < 0.5:
            return f"While I don't have exact information on this question, here's a related response that might help:\n\n{answer}"
            
        else:
            return answer
    
    def close(self):
        self.answer_streamer.stop()
        self.thinking_streamer.stop()


if __name__ == "__main__":
    alpaca_json_path = "alpaca_data_cleaned.json"  
    
    print("Streaming Knowledge Base Assistant")
    print("----------------------------------------")
    print("Type 'quit' to exit")
    print("Type 'fast' to toggle thinking steps")
    print()
    
    if not os.path.exists(alpaca_json_path):
        print(f"Error: Alpaca dataset not found at {alpaca_json_path}")
        print("Please download the dataset and update the path.")
        sys.exit(1)
    
    assistant = AlpacaStreamingKnowledgeBase(alpaca_json_path)
    
    show_thinking = True
    
    try:
        while True:
            query = input("You: ")
            
            if query.lower() in ["quit", "exit", "bye", "see you"]:
                print("\nThank you for using the knowledge assistant!")
                break
                
            if query.lower() in ["fast", "toggle"]:
                show_thinking = not show_thinking
                print(f"\nThinking steps {'enabled' if show_thinking else 'disabled'}")
                continue
            
            response_info = assistant.respond(query, show_thinking=show_thinking)
            
            assistant.answer_streamer.wait_until_done()
            
            if response_info and 'related_topics' in response_info and response_info['related_topics']:
                print("\n")
                print(f"Related topics: {', '.join(response_info['related_topics'])}")
                
                if 'top_score' in response_info:
                    print(f"Match confidence: {response_info['top_score']:.2f}")
            print()
    
    finally:
        assistant.close()