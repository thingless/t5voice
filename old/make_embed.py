import os
import glob
import numpy
import argparse
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch
from tqdm import tqdm
import torch.nn.functional as F
import tempfile
import shutil
import sys

#wavfile = 'arctic_a0508.wav'
wavfile = sys.argv[1]
spk_model = "speechbrain/spkrec-xvect-voxceleb"
size_embed = 512

def f2embed(wav_file, classifier, size_embed):
    signal, fs = torchaudio.load(wav_file)
    assert fs == 16000, fs
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        embeddings = F.normalize(embeddings, dim=2)
        embeddings = embeddings.squeeze().cpu().numpy()
    assert embeddings.shape[0] == size_embed, embeddings.shape[0]
    return embeddings

def main():
    tmpdir = tempfile.mkdtemp()
    print("Using tmpdir", tmpdir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    classifier = EncoderClassifier.from_hparams(source=spk_model, run_opts={"device": device}, savedir=tmpdir)

    emb = f2embed(
        wavfile,
        classifier,
        size_embed,
    )

    numpy.save('embed.npy', emb)

    assert len(tmpdir) > 3
    shutil.rmtree(tmpdir)

if __name__ == "__main__":
    main()
