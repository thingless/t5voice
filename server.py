import os
import io
import glob
import numpy
import argparse
import torchaudio
import torchaudio.transforms
from speechbrain.pretrained import EncoderClassifier
import torch
from tqdm import tqdm
import torch.nn.functional as F
import tempfile
import shutil
import sys
import torch
import time
import soundfile as sf
from contextlib import contextmanager
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from transformers import SpeechT5HifiGan
from datasets import load_dataset
from langchain.text_splitter import CharacterTextSplitter
import numpy as np
import logging
from tqdm import tqdm

from quart import Quart, render_template, request, make_response

logger = logging.getLogger(__name__)

# define the webapp
app = Quart(__name__)

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"  # hard-coded for now

@app.route("/")
async def index():
    return await render_template("index.html")

@app.route("/gen", methods=["POST"])
async def generate():
    # Get the wavfile and text
    wavfile = (await request.files)['file']
    text = (await request.form)['text']

    # Convert it to embeddings
    with context_timer("generate_embeddings"):
        emb = make_embed(wavfile.stream)

    # Generate the output
    outdata = gen_speech(embeddings=emb, text=text)

    # Write the response
    response = await make_response(outdata)
    response.headers['Content-Type'] = 'audio/wav'
    #response.headers['Content-Disposition'] = 'attachment; filename=out.wav'
    return response

@contextmanager
def context_timer(name='timer'):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        tim = end - start
        if tim < 1:
            logger.info("Timer %s took %3.0f ms", name, tim*1000)
        else:
            logger.info("Timer %s took %6.2f sec", name, tim)

def make_embed(fobj):
    spk_model = "speechbrain/spkrec-xvect-voxceleb"
    size_embed = 512

    tmpdir = tempfile.mkdtemp()
    try:
        classifier = EncoderClassifier.from_hparams(source=spk_model, run_opts={"device": device}, savedir=tmpdir)

        signal, fs = torchaudio.load(fobj)
        if len(signal.shape) > 1 and signal.shape[0] > 1:
            signal = signal[0]  # left channel only
        if fs != 16000:
            # resample
            resampler = torchaudio.transforms.Resample(fs, 16000, dtype=signal.dtype)
            signal = resampler(signal)
        #assert fs == 16000, fs
        with torch.no_grad():
            embeddings = classifier.encode_batch(signal)
            embeddings = F.normalize(embeddings, dim=2)
            embeddings = embeddings.squeeze().cpu().numpy()
        assert embeddings.shape[0] == size_embed, embeddings.shape[0]
    finally:
        assert len(tmpdir) > 3
        shutil.rmtree(tmpdir)

    return embeddings

def gen_speech(text, embeddings=None, voice_index=None):
#    voice_index = None  # use embedding
#    #voice_index = 6799
#    #voice_index = 6800
#    #voice_index = 7306

    assert embeddings is not None or voice_index is not None
    assert embeddings is None or voice_index is None

    token_limit = 500

    with context_timer('load_models'):
        processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
        model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
        model.to(device)
        vocoder.to(device)

    if voice_index:
        with context_timer('load_dataset'):
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            speaker_embeddings = torch.tensor(embeddings_dataset[voice_index]["xvector"]).unsqueeze(0)

    else:
        with context_timer('load_my_embedding'):
            speaker_embeddings = torch.tensor(embeddings).unsqueeze(0).to(device)

    text_splitter = CharacterTextSplitter(chunk_size=token_limit, chunk_overlap=0)
    batches = text_splitter.split_text(text)

    with context_timer('processor'):
        inputses = [
            processor(text=txt, return_tensors="pt")
            for txt in batches
        ]
        for d in inputses:
            d['input_ids'] = d['input_ids'].to(device)

    with context_timer('generate'):
        # slow step
        with torch.no_grad():
            speeches = [
                model.generate_speech(
                    input_ids=inputs["input_ids"],
                    speaker_embeddings=speaker_embeddings,
                    vocoder=vocoder,
                )
                for inputs in tqdm(inputses)
            ]

    with context_timer('write'):
        ofile = io.BytesIO()
        out = torch.concat(speeches).cpu().numpy()
        sf.write(ofile, out, samplerate=16000, format='WAV', subtype='PCM_24')
        ofile.seek(0)

    return ofile.read()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run()
    #main()

