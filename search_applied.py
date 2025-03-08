import pandas as pd
from semantic_search import TextStreamer, SimpleSemanticSearch
import json
import os
import sys
import time
import random

class StreamingKnowledgeBase:
    """A knowledge base assistant that streams responses with thinking steps."""
    
    def __init__(self, knowledge_data, stream_speed=0.02, thinking_speed=0.003):

        self.knowledge_data = knowledge_data
        
        self.titles = [entry["title"] for entry in knowledge_data]
        self.contents = [entry["content"] for entry in knowledge_data]
        self.categories = [entry.get("category", "General") for entry in knowledge_data]
        
        self.search_texts = [
            f"{title}. {content}" for title, content in zip(self.titles, self.contents)
        ]
        
        self.search_engine = SimpleSemanticSearch()
        self.search_engine.add_documents(self.search_texts)
        
        self.answer_streamer = TextStreamer(stream_interval=stream_speed)
        self.thinking_streamer = TextStreamer(stream_interval=thinking_speed)
        
        self.answer_streamer.start()
        self.thinking_streamer.start()
    
    def _generate_thinking_steps(self, query, results):
        thinking_steps = [
            f"Analyzing query: '{query}'",
            "Searching knowledge base...",
            f"Found {len(results)} relevant documents",
            "Ranking results by relevance...",
            "Extracting key information...",
            "Formulating response..."
        ]
        
        for i, (doc_id, _, score) in enumerate(results[:3]):
            title = self.titles[doc_id]
            category = self.categories[doc_id]
            thinking_steps.insert(3 + i, f"Found relevant article: '{title}' (Category: {category}, Relevance: {score:.2f})")
        
        return thinking_steps
    
    def respond(self, query, show_thinking=True):
        results = self.search_engine.search(query, top_k=3)
        
        if not results or results[0][2] < 0.3:
            response = "I don't have enough information to answer that question accurately. Could you try rephrasing or asking something else?"
            self.answer_streamer.put(response)
            return None
        
        if show_thinking:
            thinking_steps = self._generate_thinking_steps(query, results)
            print("\nThinking: ", end="")
            
            for step in thinking_steps:
                self.thinking_streamer.put(step + "... ")
                self.thinking_streamer.wait_until_done()
                time.sleep(0.2 + random.random() * 0.3)  # Random pause between steps
            
            print("\n\nAnswer: ", end="")
        
        doc_id, _, score = results[0]
        title = self.titles[doc_id]
        content = self.contents[doc_id]
        category = self.categories[doc_id]
        
        answer = self._format_answer(query, content, title)
        self.answer_streamer.put(answer)
        
        return {
            "title": title,
            "category": category,
            "confidence": score,
            "answer": answer,
            "related_titles": [self.titles[r[0]] for r in results[1:3]]
        }
    
    def _format_answer(self, query, content, title):
        if query.lower().startswith("what is") or query.lower().startswith("what are"):
            return content
            
        elif "how to" in query.lower() or "how do i" in query.lower():
            return f"To understand {title}, here's what you need to know: {content}"
            
        else:
            return f"Regarding {title}: {content}"
    
    def close(self):
        self.answer_streamer.stop()
        self.thinking_streamer.stop()


if __name__ == "__main__":
    knowledge_data = [
        {
            "title": "Python Basics",
            "category": "Programming",
            "content": "Python is a high-level, interpreted programming language known for its readability and simplicity. It supports multiple paradigms including procedural, object-oriented, and functional programming. Python features a dynamic type system and automatic memory management, making it beginner-friendly while still powerful for advanced applications."
        },
        {
            "title": "Machine Learning Introduction",
            "category": "Data Science",
            "content": "Machine learning is a subfield of artificial intelligence that focuses on developing systems that can learn from and make predictions based on data. The primary aim is to enable computers to learn automatically without human intervention. Major approaches include supervised learning, unsupervised learning, and reinforcement learning."
        },
        {
            "title": "Web Development Fundamentals",
            "category": "Programming",
            "content": "Web development involves creating websites and web applications. Frontend development focuses on what users see and interact with, using technologies like HTML, CSS, and JavaScript. Backend development handles server-side logic and database interactions using languages like Python, Ruby, or Node.js. Full-stack developers work on both frontend and backend components."
        },
        {
            "title": "Database Management Systems",
            "category": "Computing",
            "content": "Database management systems (DBMS) are software applications that interact with users, other applications, and the database itself to capture and analyze data. A DBMS allows for data definition, creation, querying, update, and administration. Common types include relational databases (SQL) and non-relational databases (NoSQL)."
        },
        {
            "title": "Natural Language Processing",
            "category": "Data Science",
            "content": "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human language. Applications include chatbots, translation services, sentiment analysis, and information extraction."
        }
    ]
    
    print("Streaming Knowledge Base Assistant")
    print("--------------------------------")
    print("Type 'quit' to exit")
    print("Type 'fast' to toggle thinking steps")
    print()
    
    assistant = StreamingKnowledgeBase(knowledge_data)
    
    show_thinking = True
    
    try:
        while True:
            query = input("You: ")
            
            if query.lower() in ["quit", "exit", "bye"]:
                print("\nThank you for using the knowledge assistant!")
                break
                
            if query.lower() in ["fast", "toggle"]:
                show_thinking = not show_thinking
                print(f"\nThinking steps {'enabled' if show_thinking else 'disabled'}")
                continue
            
            response_info = assistant.respond(query, show_thinking=show_thinking)
            
            assistant.answer_streamer.wait_until_done()
            
            if response_info:
                print("\n")
                print(f"From: {response_info['title']} (Category: {response_info['category']})")
                if response_info['related_titles']:
                    print(f"Related topics: {', '.join(response_info['related_titles'])}")
            print()
    
    finally:
        assistant.close()