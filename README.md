<h1 align="center">🤖 AI Assistant - Transformer Based Chatbot</h1>
<p align="center">
  A smart AI assistant that understands your emotions, processes your thoughts, and replies like a human – built using <strong>Python</strong>, <strong>spaCy</strong>, and <strong>PyTorch</strong>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python-blue.svg">
  <img src="https://img.shields.io/badge/Library-spaCy%20%7C%20PyTorch-red">
  <img src="https://img.shields.io/badge/Status-Development-orange">
  <img src="https://img.shields.io/badge/Made%20By-Haris%20Sheikh-green">
</p>

---

## 📌 Overview

`AI Assistant` is a lightweight yet powerful **NLP chatbot** that:
- Detects your **emotions**
- Understands your **thoughts**
- Responds using a custom-built **Transformer model**

All this is done with **modular Python classes** for easy integration, testing, and scaling.

---

## 🌟 Features

### 🔹 1. Sentiment Analyzer
- Classifies user input into:
  - 😊 Positive
  - 😐 Neutral
  - 😔 Negative
- Works using a simple keyword-based analysis.
- Prepares the assistant for emotionally intelligent responses.

### 🔹 2. Thought Vectorizer
- Uses spaCy to convert text into **thought vectors**
- Averages word embeddings to represent the full sentence meaning
- Example: `"I'm feeling great"` → `vector([0.25, 0.83, ...])`

### 🔹 3. Transformer Response Generator
- Mini version of a transformer built with PyTorch
- Components included:
  - ✅ Positional Encoding
  - ✅ Multi-Head Attention
  - ✅ Feedforward Layers
- Learns conversational patterns from provided data

### 🔹 4. Smart AIResponder
- Main interface for users
- Combines vectorized input + sentiment + transformer to generate replies
- Intelligent and customizable response logic

---

## 🧱 Architecture

