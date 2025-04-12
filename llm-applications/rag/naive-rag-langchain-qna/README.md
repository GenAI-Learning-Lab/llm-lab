# Naive RAG LangChain Q&A System 📚

This repository contains a Retrieval-Augmented Generation (RAG) system built using LangChain, designed for question-answering from PDF documents. The application leverages a naive approach to demonstrate the core functionalities of RAG, making it an excellent starting point for understanding and experimenting with this architecture.

## Overview

The Naive RAG LangChain Q&A System allows users to:

- Process PDF documents
- Generate embeddings for text chunks
- Perform similarity searches to retrieve relevant information
- Generate context-aware responses to user queries

## Embeddings

The system utilizes **Sentence Transformers** for generating embeddings from text chunks. This library provides state-of-the-art models for transforming sentences into dense vector representations, enabling effective similarity searches and context retrieval.

## Groq Cloud API for LLM Inference

The system utilizes the **Groq** Cloud API for generating responses to user queries. Groq is designed for efficient and context-aware language generation, making it well-suited for applications that require high-quality answers based on retrieved information. By leveraging Groq, the application can provide accurate and relevant responses to questions about the content of the PDF documents.

## Prerequisites

Before you begin, ensure you have the following:

- A Groq API key (obtain one from [Groq Console](https://console.groq.com))
- Python 3.9 or higher
- A Jupyter environment (Google Colab or local Jupyter Notebook)

## Setup Instructions

1. Set your Groq API key in the environment.
2. Install the required packages.
3. Use the provided sample PDF document URL for testing.

## Usage

Once set up, you can interact with the system by providing questions related to the content of the PDF documents. The application will process the documents, create a vector store, and utilize the Groq model to generate answers based on the retrieved context.

## Understanding the Flow

1. **Document Processing**: Load and split PDF documents into manageable chunks.
2. **Vector Store**: Store embeddings in FAISS for efficient similarity searches.
3. **Question Answering**: Process user questions and generate answers based on the context retrieved from the vector store.

## Next Steps

- Experiment with different chunk sizes and retrieval settings.
- Fine-tune the prompt templates and model parameters for improved responses.
- Explore additional features and enhancements to the system.

## Common Issues and Solutions

1. **API Key Errors**: Ensure your Groq API key is set correctly.
2. **Memory Issues**: Adjust chunk sizes or the number of retrieved documents if errors occur.
3. **Poor Responses**: Fine-tune the prompt or increase the retrieved context for better answers.

Happy exploring!