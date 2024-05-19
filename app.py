from huggingface_hub import InferenceClient
import gradio as gr
import random
import tempfile
import asyncio
from streaming_stt_nemo import Model
import edge_tts
from langchain_community.tools import DuckDuckGoSearchRun

API_URL = "https://api-inference.huggingface.co/models/"
client = InferenceClient("mistralai/Mistral-7B-Instruct-v0.1")

# Initialize DuckDuckGo search tool
duckduckgo_search = DuckDuckGoSearchRun()

# Initialize ASR model
default_lang = "en"
engines = { default_lang: Model(default_lang) }

def transcribe(audio):
    """Transcribes the audio file to text."""
    lang = "en"
    model = engines[lang]
    text = model.stt_file(audio)[0]
    return text

def format_prompt(message, history):
    """Formats the prompt for the language model."""
    prompt = "<s>"
    for user_prompt, bot_response in history:
        prompt += f"[INST] {user_prompt} [/INST]"
        prompt += f" {bot_response}</s> "
    prompt += f"[INST] {message} [/INST]"
    return prompt

def generate(prompt, history, temperature=0.9, max_new_tokens=512, top_p=0.95, repetition_penalty=1.0):
    """Generates a response from the language model."""
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
    top_p = float(top_p)

    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=random.randint(0, 10**7),
    )

    formatted_prompt = format_prompt(prompt, history)

    stream = client.text_generation(formatted_prompt, **generate_kwargs, stream=True, details=True, return_full_text=False)
    output = ""

    for response in stream:
        output += response.token.text
        yield output

    search_result = duckduckgo_search.run(prompt)
    if search_result:
        yield search_result
    else:
        yield "Sorry, I couldn't find any relevant information."

async def respond(audio):
    """Handles the full pipeline: transcribe, generate response, and TTS."""
    try:
        # Transcribe audio to text
        user_text = transcribe(audio)
        
        # Generate response using the language model
        history = []
        response_generator = generate(user_text, history)
        response_text = ""
        for response in response_generator:
            response_text = response

        # Convert the text response to speech
        communicate = edge_tts.Communicate(response_text)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_path = tmp_file.name
            await communicate.save(tmp_path)
        return response_text, tmp_path
    except Exception as e:
        return str(e), None

additional_inputs = [
    gr.Slider(
        label="Temperature",
        value=0.9,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        interactive=True,
        info="Higher values produce more diverse outputs",
    ),
    gr.Slider(
        label="Max new tokens",
        value=512,
        minimum=64,
        maximum=1024,
        step=64,
        interactive=True,
        info="The maximum numbers of new tokens",
    ),
    gr.Slider(
        label="Top-p (nucleus sampling)",
        value=0.90,
        minimum=0.0,
        maximum=1,
        step=0.05,
        interactive=True,
        info="Higher values sample more low-probability tokens",
    ),
    gr.Slider(
        label="Repetition penalty",
        value=1.2,
        minimum=1.0,
        maximum=2.0,
        step=0.05,
        interactive=True,
        info="Penalize repeated tokens",
    )
]

customCSS = """
#component-7 { # this is the default element ID of the chat component
  height: 800px; # adjust the height as needed
  flex-grow: 1;
}
"""

with gr.Blocks(css=customCSS) as demo:
    gr.Markdown("# RAG_FRIDAY_4.0🤖 WELCOME TO OPEN-SOURCE FREEDOM🤗(like never before)")
    gr.Markdown("Getting real-time updated results for prompts is still proprietary in the face of GPT-4, Co-Pilot etc. This app serves as an open-source alternative for this! UPDATE: Previous version of this app i.e. RAG_FRIDAY_mark_3 is also available, this is just a upgrade providing voice-based search comfort for users")

    with gr.Row():
        input_audio = gr.Audio(label="Voice Chat (BETA)", sources="microphone", type="filepath", waveform_options=False)
        output_text = gr.Textbox(label="Text Response")
        output_audio = gr.Audio(label="JARVIS", type="filepath", interactive=False, autoplay=True, elem_classes="audio")
        gr.Interface(fn=respond, inputs=[input_audio], outputs=[output_text, output_audio], live=True)

    gr.Markdown("## Additional Parameters")
    for slider in additional_inputs:
        slider.render()

demo.queue().launch(debug=True)
