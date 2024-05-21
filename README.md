# Voice-chat-enabled-RAG-chatbot-with-real-time-internet-access 🤖💬🎤🤳
An open-source project that uses cutting-edge NLP models and real-time web search to provide dynamic voice query responses. Features include speech-to-text with Nemo, text generation with Mistral-7B, DuckDuckGo search integration, and text-to-speech with edge-tts, all in a user-friendly Gradio interface.

## Features 
- **Speech-to-Text Transcription**: Utilizes Nemo for converting voice input to text.
- **Text Generation**: Employs Mistral-7B for generating coherent and contextually relevant responses.
- **Web Search Integration**: Integrates DuckDuckGo Search to provide real-time, updated information.
- **Text-to-Speech Conversion**: Uses edge-tts to convert text responses back into audio for seamless interaction.
- **User-Friendly Interface**: Built with Gradio for an interactive and intuitive web application experience.
- **Cross-Platform Compatibility**: Designed to work smoothly on both desktop and mobile devices.

## Project Overview 
This project aims to create a personal assistant that can handle voice queries and provide instant, accurate responses. It combines various advanced technologies to deliver a robust user experience.

## Key Technologies 
- **Nemo**: For speech-to-text conversion.
- **Mistral-7B**: A powerful large language model for generating text responses.
- **DuckDuckGo API**: For fetching real-time search results.
- **edge-tts**: For converting text responses to speech.
- **Gradio**: For creating the web-based user interface.
- **Asyncio**: For handling asynchronous operations in Python.

## Demo Output 
<p align="center">
<img src="Screenshot (7).png" />
</p>

## 📽️ Demo Video

![Demo](Demo2 (1).gif)

## Implementation Details:

### Requirements:

- huggingface_hub
- gradio
- langchain-community
- duckduckgo-search==5.3.1b1
- transformers
- torch
- inflect
- edge-tts
- asyncio
- streaming-stt-nemo==0.2.0

### Installation:
- Clone this repository to your local machine.

- Install Dependencies and Setup :
  
  Install using requirements file provided -
  ```
  git clone https://github.com/yourusername/voice-activated-assistant.git
  cd voice-activated-assistant
  pip install -r requirements.txt
  ```
- Access the web interface at http://localhost:7860 and start interacting with your personal assistant!

## Algorithmic Breakdown
- **Voice Input**: User provides a voice input via the microphone.
- **Transcription**: Nemo converts the audio input into text.
- **Text Generation**: Mistral-7B processes the text and generates a relevant response.
- **Web Search**: If necessary, DuckDuckGo Search API is used to fetch real-time information.
- **Response Generation**: The final text response is converted back into audio using edge-tts.
- **Playback**: The audio response is played back to the user, completing the interaction loop.

## Contribution 
- Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes that can make this app better and contribute to OPEN SOURCE FREEDOM! 💫
