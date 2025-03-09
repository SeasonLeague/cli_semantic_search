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
            "title": "Database Management Systems",
            "category": "Computing",
            "content": "Database management systems (DBMS) are software applications that interact with users, other applications, and the database itself to capture and analyze data. A DBMS allows for data definition, creation, querying, update, and administration. Common types include relational databases (SQL) and non-relational databases (NoSQL)."
        },
        {
            "title": "Natural Language Processing",
            "category": "Data Science",
            "content": "Natural Language Processing (NLP) is a field of artificial intelligence that focuses on the interaction between computers and humans through natural language. The ultimate objective of NLP is to read, decipher, understand, and make sense of human language. Applications include chatbots, translation services, sentiment analysis, and information extraction."
        },
        {
            "title": "Intro to JavaScript",
            "category": "Programming",
            "content": "JavaScript is a dynamic, high-level programming language primarily used for web development. It enables interactive web pages and is supported by all modern web browsers. With JavaScript, developers can manipulate HTML and CSS to create responsive, dynamic websites."
        },
        {
            "title": "The Theory of Relativity",
            "category": "Physics",
            "content": "Albert Einstein's theory of relativity revolutionized the way we understand space, time, and gravity. It includes both the Special and General Theory of Relativity, explaining how objects move through space and how massive objects warp the fabric of spacetime."
        },
        {
            "title": "Deep Learning Basics",
            "category": "AI & Machine Learning",
            "content": "Deep learning is a subset of machine learning that uses neural networks with many layers (hence 'deep') to model complex patterns in large amounts of data. Deep learning has powered advancements in image recognition, natural language processing, and more."
        },
        {
            "title": "The History of the Internet",
            "category": "Technology",
            "content": "The Internet has its roots in the 1960s with the development of ARPANET, a U.S. government project aimed at creating a communication system resistant to nuclear attack. By the 1990s, the World Wide Web emerged, transforming the internet into a global network for communication, commerce, and information."
        },
        {
            "title": "Basic Concepts in Cryptography",
            "category": "Security",
            "content": "Cryptography is the practice of securing communication and data from third-party access. Key concepts include encryption, decryption, hashing, and digital signatures, which protect data confidentiality, integrity, and authenticity."
        },
        {
            "title": "Basic HTML Structure",
            "category": "Web Development",
            "content": "HTML (HyperText Markup Language) is the standard markup language for creating web pages. HTML documents consist of elements such as headings, paragraphs, links, and images. The basic structure includes the 'html', 'head', and 'body' tags, with the 'head' section containing metadata, and the 'body' section containing the content."
        },
        {
            "title": "Newton's Laws of Motion",
            "category": "Physics",
            "content": "Sir Isaac Newton's Three Laws of Motion describe the relationship between a body and the forces acting upon it. They form the foundation of classical mechanics: 1) An object in motion stays in motion unless acted upon by an external force. 2) Force equals mass times acceleration. 3) Every action has an equal and opposite reaction."
        },
        {
            "title": "HTML5 & CSS3 Features",
            "category": "Web Development",
            "content": "HTML5 introduced semantic elements, improved audio/video capabilities, and APIs for offline web apps. CSS3 brought in new features like transitions, animations, and flexbox for responsive design, making it easier to create dynamic, interactive websites."
        },
        {
            "title": "Overview of the Solar System",
            "category": "Astronomy",
            "content": "The Solar System consists of the Sun and the objects that are gravitationally bound to it, including eight planets, their moons, asteroids, comets, and the Kuiper Belt. Earth is the third planet from the Sun and supports life, a rarity in the known universe."
        },
        {
            "title": "Introduction to Cybersecurity",
            "category": "Security",
            "content": "Cybersecurity involves protecting computer systems and networks from cyberattacks, unauthorized access, and data breaches. This includes both hardware and software protections and covers practices like encryption, firewalls, and antivirus programs."
        },
        {
            "title": "Principles of Quantum Mechanics",
            "category": "Physics",
            "content": "Quantum mechanics describes the behavior of matter and energy at atomic and subatomic scales. Unlike classical physics, quantum mechanics introduces concepts like wave-particle duality, uncertainty principle, and superposition, which challenge our everyday understanding of the world."
        },
        {
            "title": "What is Blockchain?",
            "category": "Technology",
            "content": "Blockchain is a decentralized, distributed ledger technology that securely records transactions across multiple computers. It is most known for its use in cryptocurrencies like Bitcoin, but its applications also extend to fields such as supply chain management, voting systems, and more."
        },
        {
            "title": "The Human Genome Project",
            "category": "Biology",
            "content": "The Human Genome Project was an international research initiative aimed at mapping all the genes in the human genome. Completed in 2003, it provided a complete sequence of human DNA, advancing our understanding of genetics and leading to breakthroughs in medicine and disease prevention."
        },
        {
            "title": "Understanding Artificial Intelligence",
            "category": "AI & Machine Learning",
            "content": "Artificial Intelligence (AI) involves creating machines that can perform tasks that would normally require human intelligence, such as visual perception, decision-making, and language translation. Key subfields include machine learning, robotics, and natural language processing."
        },
        {
            "title": "Basic Concepts in Data Science",
            "category": "Data Science",
            "content": "Data science involves extracting knowledge and insights from large datasets using various techniques including statistics, machine learning, and data visualization. It helps organizations make informed decisions based on data-driven analysis."
        },
        {
            "title": "Introduction to the Theory of Evolution",
            "category": "Biology",
            "content": "The theory of evolution, proposed by Charles Darwin, suggests that species evolve over time through natural selection. Organisms with traits that improve their chances of survival are more likely to reproduce, passing on these advantageous traits to future generations."
        },
        {
            "title": "The Big Bang Theory",
            "category": "Cosmology",
            "content": "The Big Bang Theory is the leading explanation for the origin of the universe, proposing that the universe began as a singularity around 13.8 billion years ago and has been expanding ever since. Evidence includes cosmic background radiation and the observed redshift of distant galaxies."
        },
        {
            "title": "Principles of Data Visualization",
            "category": "Data Science",
            "content": "Data visualization is the graphical representation of data and information. Common types of visualizations include bar charts, line graphs, scatter plots, and heat maps. Effective visualizations help people understand patterns, trends, and outliers in complex data."
        },
        {
            "title": "The Role of Nanotechnology in Medicine",
            "category": "Technology",
            "content": "Nanotechnology involves manipulating matter on an atomic or molecular scale. In medicine, it is used for targeted drug delivery, medical imaging, and the development of new materials for implants and prosthetics."
        },
        {
            "title": "Introduction to Web APIs",
            "category": "Web Development",
            "content": "Web APIs (Application Programming Interfaces) allow different software systems to communicate with each other over the web. They enable developers to access the functionality of external services, like social media platforms or payment systems, and integrate them into applications."
        },
        {
            "title": "Artificial Intelligence in Healthcare",
            "category": "AI & Healthcare",
            "content": "AI is transforming healthcare by providing tools for disease diagnosis, personalized treatment plans, drug discovery, and more. Machine learning models can analyze medical images, predict patient outcomes, and assist healthcare professionals in making data-driven decisions."
        },
        {
            "title": "The Impact of Social Media on Society",
            "category": "Sociology",
            "content": "Social media platforms have significantly changed how people communicate, interact, and perceive the world. While they offer opportunities for connection, they also raise concerns about privacy, misinformation, and the mental health impact of constant connectivity."
        },
        {
            "title": "Introduction to Theoretical Physics",
            "category": "Physics",
            "content": "Theoretical physics involves the use of mathematical models and abstractions to explain and predict natural phenomena. It includes fields such as quantum mechanics, relativity, and string theory, and often bridges the gap between experimental physics and pure mathematics."
        },
        {
            "title": "What is Machine Learning?",
            "category": "Data Science",
            "content": "Machine learning is a type of artificial intelligence that enables systems to learn from data and improve performance over time without being explicitly programmed. It is used in various applications, such as recommendation systems, speech recognition, and image classification."
        },
        {
            "title": "The Internet of Things (IoT) in Smart Cities",
            "category": "Technology",
            "content": "The Internet of Things (IoT) is the backbone of smart cities, where connected devices, such as sensors and smart meters, are used to collect data that can improve urban living. IoT technologies help in traffic management, energy efficiency, and waste management."
        },
        {
            "title": "The Role of Artificial Intelligence in Education",
            "category": "AI & Education",
            "content": "Artificial intelligence in education is reshaping traditional learning by offering personalized learning experiences, intelligent tutoring systems, and automation of administrative tasks. AI tools can adapt to individual students' needs, making education more accessible and efficient."
        },
        {
            "title": "The Impact of Climate Change on Agriculture",
            "category": "Environment",
            "content": "Climate change has serious implications for global agriculture, including changes in precipitation patterns, increased frequency of extreme weather events, and shifting growing seasons. These disruptions affect crop yields and food security, requiring new approaches to farming and resource management."
        },
        {
            "title": "What is Genetic Engineering?",
            "category": "Biology",
            "content": "Genetic engineering involves the modification of an organism's DNA to achieve desired traits. It is used in agriculture to create crops resistant to pests and diseases, in medicine to produce insulin and vaccines, and in research to study genes and disease mechanisms."
        },
        {
            "title": "The Evolution of the Smartphone",
            "category": "Technology",
            "content": "The smartphone has evolved from a basic mobile phone to a powerful computing device. With advancements in mobile processors, cameras, and operating systems, smartphones now offer functionalities such as social media, GPS navigation, and mobile gaming."
        },
        {
            "title": "Renewable Energy Sources",
            "category": "Energy",
            "content": "Renewable energy sources, such as solar, wind, and hydropower, provide an alternative to fossil fuels. They are crucial in addressing climate change and ensuring sustainable energy for the future. Technological advancements are making renewable energy more efficient and accessible."
        },
        {
            "title": "The Process of Photosynthesis",
            "category": "Biology",
            "content": "Photosynthesis is the process by which green plants and some other organisms convert light energy, usually from the sun, into chemical energy stored in glucose. It is vital for life on Earth, as it provides oxygen and food for nearly all living organisms."
        },
        {
            "title": "The Role of Artificial Intelligence in Autonomous Vehicles",
            "category": "AI & Technology",
            "content": "Artificial intelligence plays a key role in the development of autonomous vehicles. By using sensors, cameras, and machine learning algorithms, self-driving cars can navigate roads, recognize obstacles, and make decisions in real time without human intervention."
        },
        {
            "title": "The Industrial Revolution",
            "category": "History",
            "content": "The Industrial Revolution was a period of significant technological and socioeconomic change that began in the late 18th century. It marked the transition from agrarian economies to industrialized ones, leading to innovations in manufacturing, transportation, and communication."
        },
        {
            "title": "The Role of Microorganisms in the Ecosystem",
            "category": "Biology",
            "content": "Microorganisms, including bacteria, fungi, and viruses, play a crucial role in the Earth's ecosystems. They are involved in processes such as decomposition, nutrient cycling, and symbiosis, and many are essential for the health of plants, animals, and humans."
        },
        {
            "title": "Electricity and Magnetism",
            "category": "Physics",
            "content": "Electricity and magnetism are fundamental forces of nature. Electric charges produce electric fields, and moving electric charges create magnetic fields. These forces are interconnected in electromagnetism, which is the basis for technologies like motors, generators, and electromagnets."
        },
        {
            "title": "The Solar Energy Industry",
            "category": "Energy",
            "content": "Solar energy is harnessed through photovoltaic cells that convert sunlight into electricity. The solar energy industry has grown rapidly in recent years, driven by advancements in solar panel technology, falling costs, and increased demand for renewable energy solutions."
        },
        {
            "title": "The Role of Genetics in Disease",
            "category": "Biology",
            "content": "Genetics plays a major role in many diseases, with some being inherited through family lines. Understanding genetic disorders and their causes can lead to better treatment options, gene therapies, and even preventive measures."
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