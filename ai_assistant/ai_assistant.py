import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import re
import string
import json
import queue
import threading
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertModel, BertTokenizer
import speech_recognition as sr
import pyttsx3
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from collections import deque, defaultdict
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import GloVe
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os
from datetime import datetime

# Initialize components
nltk.download('punkt')
nltk.download('wordnet')
nlp = spacy.load('en_core_web_lg')

class AdvancedVocalAI:
    def __init__(self):
        # Audio configuration
        self.sample_rate = 44100
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.is_speaking = False
        self.last_interaction_time = time.time()
        
        # Enhanced speech recognition
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 1.0
        self.recognizer.energy_threshold = 3000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.operation_timeout = 5
        
        # Advanced Text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 175)
        self.tts_engine.setProperty('volume', 1.0)
        voices = self.tts_engine.getProperty('voices')
        self.tts_engine.setProperty('voice', voices[1].id)  # Female voice
        self.tts_engine.setProperty('pitch', 110)  # Slightly higher pitch
        
        # Language processing
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.glove = GloVe(name='840B', dim=300)
        self._build_enhanced_vocab()
        self.sentiment_analyzer = self.SentimentAnalyzer()
        
        # Neural components
        input_dim = 768  # BERT embedding size
        thought_dim = 16  # More nuanced thought representation
        hidden_dim = 512  # Larger model capacity
        self.thought_gen = self.AdvancedThoughtGenerator(input_dim, hidden_dim, thought_dim)
        self.response_gen = self.AdvancedResponseGenerator(thought_dim, hidden_dim, self.vocab_size)
        
        # Knowledge Base
        self.knowledge_base = self.KnowledgeBase()
        
        # Personality and memory
        self.personality = self.AdvancedPersonalityModule()
        self.memory = self.AdvancedMemorySystem()
        self.context_window = deque(maxlen=7)  # Larger context window
        self.emotional_state = {
            'valence': 0.5,  # Positive/negative
            'arousal': 0.5,  # Intensity
            'dominance': 0.5  # Control
        }
        
        # Learning system
        self.learning_engine = self.LearningEngine()
        self.conversation_history = []
        
        # Start threads
        self.listening_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listening_thread.start()
        
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        
        # Load saved state if exists
        self.load_state()

    # -------------------------
    # Core Neural Network Components
    # -------------------------
    class AdvancedThoughtGenerator(nn.Module):
        def __init__(self, input_dim, hidden_dim, thought_dim):
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4)
            self.thought_layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, thought_dim),
                nn.Tanh()
            )
            
        def forward(self, x):
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and sequence dimensions
            attn_output, _ = self.attention(x, x, x)
            thought = self.thought_layers(attn_output.squeeze(0).squeeze(0))
            return thought
    
    class AdvancedResponseGenerator(nn.Module):
        def __init__(self, thought_dim, hidden_dim, vocab_size):
            super().__init__()
            self.thought_to_hidden = nn.Linear(thought_dim, hidden_dim)
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4),
                num_layers=3
            )
            self.vocab_proj = nn.Linear(hidden_dim, vocab_size)
            self.embedding = nn.Embedding(vocab_size, hidden_dim)
            self.positional_encoding = self._generate_positional_encoding(hidden_dim, 50)
            
        def _generate_positional_encoding(self, d_model, max_len):
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe
            
        def forward(self, thought, max_len=25, temperature=0.7, top_k=50):
            hidden = self.thought_to_hidden(thought)
            hidden = hidden.unsqueeze(0).unsqueeze(0)
            
            # Initialize with random token
            input_seq = torch.randint(0, self.embedding.num_embeddings, (1, 1))
            output_seq = []
            
            for i in range(max_len):
                # Embed input sequence
                embedded = self.embedding(input_seq) + self.positional_encoding[:input_seq.size(1)]
                
                # Transformer processing
                transformer_out = self.transformer(embedded)
                
                # Get next token probabilities
                logits = self.vocab_proj(transformer_out[:, -1, :])
                
                # Apply temperature and top-k sampling
                logits = logits / temperature
                top_k_values, top_k_indices = torch.topk(logits, top_k)
                probs = torch.softmax(top_k_values, dim=-1)
                next_token = top_k_indices.gather(-1, torch.multinomial(probs, 1))
                
                output_seq.append(next_token.item())
                input_seq = torch.cat((input_seq, next_token), dim=1)
                
                # Stop if end token generated
                if next_token.item() in self.tokenizer.all_special_ids:
                    break
                    
            return output_seq

    # -------------------------
    # Cognitive Modules
    # -------------------------
    class AdvancedPersonalityModule:
        def __init__(self):
            # Big Five personality traits with dynamic adjustment
            self.traits = {
                'openness': np.clip(np.random.normal(0.7, 0.1), 0, 1),
                'conscientiousness': np.clip(np.random.normal(0.6, 0.1), 0, 1),
                'extraversion': np.clip(np.random.normal(0.5, 0.1), 0, 1),
                'agreeableness': np.clip(np.random.normal(0.8, 0.1), 0, 1),
                'neuroticism': np.clip(np.random.normal(0.3, 0.1), 0, 1)
            }
            self.mood = 0.5  # Neutral mood
            self.values = {
                'honesty': 0.9,
                'creativity': 0.7,
                'knowledge': 0.8,
                'empathy': 0.75,
                'curiosity': 0.85
            }
            self.learning_rate = 0.01  # How quickly personality adapts
            
        def update_from_experience(self, experience):
            sentiment = experience['sentiment']
            self.mood = 0.95 * self.mood + 0.05 * sentiment
            
            # Adjust traits based on experiences
            if sentiment > 0.5:
                self.traits['extraversion'] = np.clip(self.traits['extraversion'] + self.learning_rate * 0.5, 0, 1)
                self.traits['agreeableness'] = np.clip(self.traits['agreeableness'] + self.learning_rate * 0.3, 0, 1)
            elif sentiment < -0.5:
                self.traits['neuroticism'] = np.clip(self.traits['neuroticism'] + self.learning_rate * 0.2, 0, 1)
                
            # Values strengthen with relevant experiences
            if 'learn' in experience['keywords']:
                self.values['knowledge'] = np.clip(self.values['knowledge'] + 0.01, 0, 1)
            if 'creative' in experience['keywords']:
                self.values['creativity'] = np.clip(self.values['creativity'] + 0.01, 0, 1)
                
        def influence_response(self, response_words):
            modified_words = []
            for word in response_words:
                # Extraversion affects word choice
                if self.traits['extraversion'] > 0.7 and word in ['i', 'me']:
                    if random.random() < 0.4:
                        word = 'we' if random.random() < 0.5 else 'you'
                
                # Agreeableness affects politeness
                if self.traits['agreeableness'] > 0.7:
                    if word in ['no', 'bad', 'hate'] and random.random() < 0.5:
                        word = random.choice(['maybe', 'possibly', "I'm not sure about", "I have mixed feelings about"])
                
                # Neuroticism affects emotional words
                if self.traits['neuroticism'] > 0.6:
                    if word in ['happy', 'good'] and random.random() < 0.3:
                        word = 'okay' if random.random() < 0.5 else 'alright'
                
                modified_words.append(word)
                
            return modified_words
        
        def get_personality_vector(self):
            return torch.tensor([
                self.traits['openness'],
                self.traits['conscientiousness'],
                self.traits['extraversion'],
                self.traits['agreeableness'],
                self.traits['neuroticism'],
                self.values['honesty'],
                self.values['creativity'],
                self.values['knowledge'],
                self.values['empathy'],
                self.values['curiosity'],
                self.mood
            ])

    class AdvancedMemorySystem:
        def __init__(self, capacity=1000):
            self.episodic_memory = deque(maxlen=capacity)  # Specific experiences
            self.semantic_memory = defaultdict(float)  # General knowledge
            self.emotional_memory = []  # Strong emotional experiences
            self.importance_decay = 0.99  # How quickly memories decay
            
        def add_experience(self, thought, response, feedback, emotional_valence, keywords):
            # Store episodic memory
            memory_entry = {
                'thought': thought,
                'response': response,
                'feedback': feedback,
                'valence': emotional_valence,
                'time': time.time(),
                'importance': 1.0,
                'keywords': keywords
            }
            self.episodic_memory.append(memory_entry)
            
            # Store emotionally charged memories separately
            if abs(emotional_valence) > 0.7:
                self.emotional_memory.append(memory_entry)
                
            # Update semantic memory
            for word in keywords:
                self.semantic_memory[word] = min(self.semantic_memory[word] + 0.1, 1.0)
                
        def sample_memories(self, batch_size=5, prioritize_emotional=False):
            if prioritize_emotional and self.emotional_memory:
                return random.sample(self.emotional_memory, min(batch_size, len(self.emotional_memory)))
            
            # Sample based on importance and recency
            weights = np.array([m['importance'] * np.exp(-0.1 * (time.time() - m['time'])) 
                             for m in self.episodic_memory])
            if len(weights) == 0:
                return []
                
            weights = weights / weights.sum()
            indices = np.random.choice(len(self.episodic_memory), size=min(batch_size, len(self.episodic_memory)), 
                                     p=weights, replace=False)
            return [self.episodic_memory[i] for i in indices]
            
        def consolidate_memories(self):
            # Decay importance of all memories
            for memory in self.episodic_memory:
                memory['importance'] *= self.importance_decay
                
            # Cluster similar memories to form generalizations
            if len(self.episodic_memory) > 50:
                self._cluster_memories()
                
        def _cluster_memories(self):
            # Extract memory embeddings
            memory_embeddings = [m['thought'].numpy() for m in self.episodic_memory]
            
            # Cluster memories
            kmeans = KMeans(n_clusters=min(10, len(memory_embeddings)//5))
            clusters = kmeans.fit_predict(memory_embeddings)
            
            # For each cluster, find the most representative memory
            for cluster_id in set(clusters):
                cluster_memories = [m for m, c in zip(self.episodic_memory, clusters) if c == cluster_id]
                if cluster_memories:
                    # Find memory closest to centroid
                    centroid = kmeans.cluster_centers_[cluster_id]
                    closest = min(cluster_memories, 
                                key=lambda m: np.linalg.norm(m['thought'].numpy() - centroid))
                    closest['importance'] = 1.0  # Boost importance of representative memory

    class KnowledgeBase:
        def __init__(self):
            self.facts = defaultdict(list)
            self.sources = {}
            self.confidence = defaultdict(float)
            
        def add_fact(self, fact, source="user", confidence=0.8):
            # Normalize fact
            normalized = self._normalize_fact(fact)
            if normalized:
                self.facts[normalized['subject']].append((normalized['predicate'], normalized['object']))
                self.sources[(normalized['subject'], normalized['predicate'], normalized['object'])] = source
                self.confidence[(normalized['subject'], normalized['predicate'], normalized['object'])] = confidence
                
        def query(self, subject, predicate=None):
            results = []
            if subject in self.facts:
                for p, o in self.facts[subject]:
                    if predicate is None or p == predicate:
                        confidence = self.confidence[(subject, p, o)]
                        source = self.sources.get((subject, p, o), "unknown")
                        results.append((p, o, confidence, source))
            return results
            
        def _normalize_fact(self, text):
            # Simple fact extraction (subject, predicate, object)
            doc = nlp(text)
            for sent in doc.sents:
                subj = None
                predicate = None
                obj = None
                
                for token in sent:
                    if "subj" in token.dep_:
                        subj = token.text
                    elif token.dep_ == "ROOT":
                        predicate = token.lemma_
                    elif "obj" in token.dep_:
                        obj = token.text
                        
                if subj and predicate and obj:
                    return {
                        'subject': subj.lower(),
                        'predicate': predicate.lower(),
                        'object': obj.lower()
                    }
            return None

    class SentimentAnalyzer:
        def __init__(self):
            self.lexicon = {
                'happy': 1, 'good': 1, 'great': 1, 'wonderful': 1, 'love': 1,
                'sad': -1, 'bad': -1, 'terrible': -1, 'hate': -1, 'angry': -1
            }
            
        def analyze(self, text):
            doc = nlp(text)
            sentiment = 0
            count = 0
            
            for token in doc:
                if token.text.lower() in self.lexicon:
                    sentiment += self.lexicon[token.text.lower()]
                    count += 1
                elif token.sentiment != 0:
                    sentiment += token.sentiment
                    count += 1
                    
            return sentiment / count if count != 0 else 0

    class LearningEngine:
        def __init__(self):
            self.learning_rate = 0.001
            self.batch_size = 32
            self.replay_interval = 100  # Replay memories every N interactions
            self.interaction_count = 0
            
        def learn_from_experience(self, ai_instance):
            self.interaction_count += 1
            
            # Replay important memories periodically
            if self.interaction_count % self.replay_interval == 0:
                self._replay_memories(ai_instance)
                
            # Update based on recent experiences
            recent_memories = ai_instance.memory.sample_memories(self.batch_size)
            if recent_memories:
                self._update_models(ai_instance, recent_memories)
                
            # Consolidate knowledge
            ai_instance.knowledge_base._cluster_knowledge()
            
        def _replay_memories(self, ai_instance):
            important_memories = ai_instance.memory.sample_memories(
                self.batch_size * 2, prioritize_emotional=True)
            
            if important_memories:
                self._update_models(ai_instance, important_memories)
                
        def _update_models(self, ai_instance, memories):
            thought_losses = []
            response_losses = []
            
            for memory in memories:
                # Thought model update
                ai_instance.thought_optim.zero_grad()
                current_thought = ai_instance.thought_gen(memory['thought'])
                target = memory['thought'] * (0.5 + memory['feedback'] * 0.5)
                loss = nn.MSELoss()(current_thought, target)
                loss.backward()
                ai_instance.thought_optim.step()
                thought_losses.append(loss.item())
                
                # Response model update
                ai_instance.response_optim.zero_grad()
                word_ids = [ai_instance.word2idx.get(word, 0) 
                           for word in memory['response'].split()]
                if len(word_ids) < 2:
                    continue
                    
                input_ids = torch.tensor(word_ids[:-1])
                target_ids = torch.tensor(word_ids[1:])
                
                output, _ = ai_instance.response_gen(input_ids)
                loss = nn.CrossEntropyLoss()(output, target_ids)
                loss.backward()
                ai_instance.response_optim.step()
                response_losses.append(loss.item())
            
            # Adjust learning rates based on performance
            if thought_losses and response_losses:
                avg_thought_loss = np.mean(thought_losses)
                avg_response_loss = np.mean(response_losses)
                
                if avg_thought_loss < 0.1:
                    ai_instance.thought_optim.param_groups[0]['lr'] *= 0.99
                if avg_response_loss < 0.1:
                    ai_instance.response_optim.param_groups[0]['lr'] *= 0.99

    # -------------------------
    # Core AI Methods
    # -------------------------
    def _build_enhanced_vocab(self):
        base_vocab = [
            'hello', 'hi', 'hey', 'how', 'are', 'you', 'i', 'am', 'fine', 
            'good', 'bad', 'what', 'is', 'your', 'name', 'my', 'today',
            'weather', 'like', 'think', 'about', 'life', 'happy', 'sad',
            'angry', 'feel', 'feeling', 'time', 'now', 'bye', 'goodbye',
            'thanks', 'thank', 'please', 'sorry', 'yes', 'no', 'maybe',
            'why', 'because', 'reason', 'can', 'you', 'help', 'me',
            'would', 'could', 'should', 'want', 'need', 'desire',
            'understand', 'comprehend', 'believe', 'opinion', 'view',
            'interesting', 'fascinating', 'curious', 'knowledge', 'learn',
            'know', 'remember', 'forget', 'important', 'change', 'grow',
            'develop', 'improve', 'create', 'imagine', 'future', 'past',
            'present', 'moment', 'experience', 'emotion', 'love', 'hate',
            'joy', 'sorrow', 'peace', 'war', 'technology', 'science', 'art',
            'music', 'literature', 'philosophy', 'human', 'nature', 'world'
        ]
        
        phrases = [
            'how are you', 'what is', 'i think', 'in my opinion',
            'the meaning of life', 'artificial intelligence',
            'machine learning', 'deep learning', 'neural networks',
            'what do you think about', 'can you explain', 'tell me more about',
            'i remember that', 'from my perspective', 'in this situation'
        ]
        
        self.word2idx = {word: idx for idx, word in enumerate(base_vocab + phrases)}
        self.idx2word = {idx: word for idx, word in enumerate(base_vocab + phrases)}
        self.vocab_size = len(self.word2idx)

    def _listen_loop(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            while True:
                if not self.is_listening or self.is_speaking:
                    time.sleep(0.1)
                    continue
                
                try:
                    print("\n[Listening...]")
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=7)
                    text = self.recognizer.recognize_google(audio, show_all=False)
                    print(f"Human: {text}")
                    self.process_input(text)
                    self.last_interaction_time = time.time()
                except sr.WaitTimeoutError:
                    # Check if we should initiate conversation
                    if time.time() - self.last_interaction_time > 20:
                        self.initiate_conversation()
                    continue
                except sr.UnknownValueError:
                    print("[Could not understand audio]")
                except Exception as e:
                    print(f"[Error in listening: {e}]")

    def _learning_loop(self):
        while True:
            time.sleep(60)  # Learn every minute
            self.learning_engine.learn_from_experience(self)
            self.memory.consolidate_memories()
            self.save_state()

    def process_input(self, text):
        # Update emotional state
        sentiment = self.sentiment_analyzer.analyze(text)
        self.emotional_state['valence'] = 0.9 * self.emotional_state['valence'] + 0.1 * sentiment
        self.emotional_state['arousal'] = np.clip(self.emotional_state['arousal'] + 0.1 * abs(sentiment), 0, 1)
        
        # Extract keywords
        doc = nlp(text)
        keywords = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
        
        # Update context
        self.context_window.append(text)
        self.detect_topic(text)
        
        # Generate thought and response
        input_vector = self.text_to_vector(text)
        thought = self.generate_thought(input_vector)
        response = self.generate_response(thought)
        
        # Personality adjustment
        response_words = response.split()
        response_words = self.personality.influence_response(response_words)
        response = ' '.join(response_words)
        
        # Apply conversational rules
        response = self.apply_conversational_rules(response)
        
        # Store in memory
        feedback = 0.5 + self.emotional_state['valence'] * 0.5
        self.memory.add_experience(
            thought, response, feedback, 
            self.emotional_state['valence'], keywords
        )
        
        # Add to conversation history
        self.conversation_history.append({
            'time': datetime.now().isoformat(),
            'input': text,
            'response': response,
            'sentiment': sentiment,
            'keywords': keywords
        })
        
        # Speak response
        self.speak(response)

    def text_to_vector(self, text):
        with torch.no_grad():
            inputs = self.bert_tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            outputs = self.bert_model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze()

    def generate_thought(self, input_vector):
        thought = self.thought_gen(input_vector)
        personality_factors = torch.cat([
            self.personality.get_personality_vector(),
            torch.tensor([
                self.emotional_state['valence'],
                self.emotional_state['arousal'],
                self.emotional_state['dominance']
            ])
        ])
        return thought * personality_factors[:thought.size(0)]

    def generate_response(self, thought_vector):
        # Generate multiple candidates
        candidates = []
        for temp in [0.5, 0.7, 1.0]:
            for top_k in [30, 50, 70]:
                word_ids = self.response_gen(thought_vector, temperature=temp, top_k=top_k)
                words = [self.idx2word.get(idx, "") for idx in word_ids]
                candidates.append(' '.join([w for w in words if w]))
        
        # Select best candidate
        if candidates:
            # Score candidates by coherence and relevance
            scored = []
            for candidate in candidates:
                # Coherence score (length and punctuation)
                coherence = min(len(candidate.split()) / 10, 1.0)
                if candidate and candidate[-1] in '.!?':
                    coherence += 0.2
                
                # Relevance score (semantic similarity to thought)
                candidate_vec = self.text_to_vector(candidate)
                relevance = torch.cosine_similarity(
                    thought_vector.unsqueeze(0), 
                    candidate_vec.unsqueeze(0)
                ).item()
                
                scored.append((candidate, coherence + relevance))
            
            # Select best candidate
            best_candidate = max(scored, key=lambda x: x[1])[0]
            return best_candidate
        return "I'm not sure what to say."

    def apply_conversational_rules(self, text):
        doc = nlp(text)
        sentences = []
        
        for sent in doc.sents:
            # Capitalize first letter
            text = sent.text
            if text:
                text = text[0].upper() + text[1:]
                # Add punctuation if missing
                if text[-1] not in '.!?':
                    text += random.choice(['.', '!', '?'])
                sentences.append(text)
        
        # Apply personality-based filters
        full_text = ' '.join(sentences)
        if self.personality.traits['agreeableness'] > 0.7:
            full_text = full_text.replace(' you are wrong', " I see it differently")
            full_text = full_text.replace(' you are stupid', " that's an interesting perspective")
        
        return full_text

    def speak(self, text):
        self.is_speaking = True
        print(f"AI: {text}")
        
        # Convert text to speech with emotional inflection
        if self.emotional_state['valence'] > 0.6:
            self.tts_engine.setProperty('rate', 165)  # Faster when happy
            self.tts_engine.setProperty('volume', 1.0)
        elif self.emotional_state['valence'] < 0.4:
            self.tts_engine.setProperty('rate', 140)  # Slower when sad
            self.tts_engine.setProperty('volume', 0.8)
        else:
            self.tts_engine.setProperty('rate', 155)
            self.tts_engine.setProperty('volume', 0.9)
        
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()
        
        # Add natural pause
        pause_duration = min(0.5 + len(text.split()) * 0.1, 2.0)
        time.sleep(pause_duration)
        self.is_speaking = False

    def initiate_conversation(self):
        if not self.is_speaking and not self.is_listening:
            topics = [
                "What are you thinking about?",
                "I was wondering about your opinion on artificial intelligence.",
                "How are you feeling today?",
                "I'd love to hear your thoughts on human emotions.",
                "What's something interesting you've learned recently?"
            ]
            self.speak(random.choice(topics))
            self.is_listening = True

    def save_state(self):
        state = {
            'memory': list(self.memory.episodic_memory),
            'personality': self.personality.__dict__,
            'knowledge': dict(self.knowledge_base.facts),
            'conversation_history': self.conversation_history[-100:],  # Save last 100 conversations
            'models': {
                'thought_gen': self.thought_gen.state_dict(),
                'response_gen': self.response_gen.state_dict()
            }
        }
        
        try:
            with open('ai_state.pkl', 'wb') as f:
                pickle.dump(state, f)
            print("[AI State Saved]")
        except Exception as e:
            print(f"[Error saving state: {e}]")

    def load_state(self):
        try:
            if os.path.exists('ai_state.pkl'):
                with open('ai_state.pkl', 'rb') as f:
                    state = pickle.load(f)
                
                # Load memory
                self.memory.episodic_memory = deque(state.get('memory', []), maxlen=1000)
                
                # Load personality
                for k, v in state.get('personality', {}).items():
                    if hasattr(self.personality, k):
                        setattr(self.personality, k, v)
                
                # Load knowledge
                self.knowledge_base.facts = defaultdict(list, state.get('knowledge', {}))
                
                # Load conversation history
                self.conversation_history = state.get('conversation_history', [])
                
                # Load models
                if 'models' in state:
                    self.thought_gen.load_state_dict(state['models']['thought_gen'])
                    self.response_gen.load_state_dict(state['models']['response_gen'])
                
                print("[AI State Loaded]")
                return True
        except Exception as e:
            print(f"[Error loading state: {e}]")
        return False

    def run(self):
        self.is_listening = True
        self.speak("Hello! I'm here and ready to talk. What's on your mind?")
        
        try:
            while True:
                time.sleep(1)
                # Periodically initiate conversation if idle
                if time.time() - self.last_interaction_time > 45:
                    self.initiate_conversation()
        except KeyboardInterrupt:
            self.speak("Goodbye for now! It was wonderful talking with you.")
            self.is_listening = False
            self.save_state()

# Initialize and run the AI
if __name__ == "__main__":
    # Add sentiment analysis capability to spaCy
    def custom_sentiment(token):
        positive = ['happy', 'good', 'great', 'wonderful', 'love', 'joy']
        negative = ['sad', 'bad', 'terrible', 'hate', 'angry', 'sorrow']
        
        if token.text.lower() in positive:
            return 1
        elif token.text.lower() in negative:
            return -1
        return 0
    
    for token in nlp.vocab:
        token.sentiment = custom_sentiment(token)
    
    # Create and run AI
    ai = AdvancedVocalAI()
    ai.run()
