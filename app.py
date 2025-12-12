from flask import Flask, render_template, request, send_from_directory
import numpy as np
import librosa
import soundfile as sf
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/output"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def int16_to_nibbles(x):
    u = np.uint16(x)
    return [
        int((u >> 12) & 0xF),
        int((u >> 8) & 0xF),
        int((u >> 4) & 0xF),
        int(u & 0xF),
    ]


def nibbles_to_int16(nibs):
    u = (nibs[0] << 12) | (nibs[1] << 8) | (nibs[2] << 4) | nibs[3]
    if u >= 32768:
        u -= 65536
    return np.int16(u)


def hamming74_encode_nibble(nib):
    d1 = (nib >> 3) & 1
    d2 = (nib >> 2) & 1
    d3 = (nib >> 1) & 1
    d4 = nib & 1

    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4

    return np.array([p1, p2, d1, p3, d2, d3, d4], dtype=np.uint8)


def hamming74_decode_nibble(code):
    c = code.copy()
    b1, b2, b3, b4, b5, b6, b7 = c

    s1 = b1 ^ b3 ^ b5 ^ b7
    s2 = b2 ^ b3 ^ b6 ^ b7
    s3 = b4 ^ b5 ^ b6 ^ b7

    pos = s1 * 1 + s2 * 2 + s3 * 4
    if pos != 0:
        c[pos - 1] ^= 1

    d1, d2, d3, d4 = c[2], c[4], c[5], c[6]
    return (d1 << 3) | (d2 << 2) | (d3 << 1) | d4


def proses_hamming(path):
    y_float, sr = librosa.load(path, sr=None)
    y_int16 = np.clip(y_float * 32767, -32768, 32767).astype(np.int16)
    N = len(y_int16)

    codes = np.zeros((N, 4, 7), dtype=np.uint8)
    for i in range(N):
        nibs = int16_to_nibbles(y_int16[i])
        for j in range(4):
            codes[i, j] = hamming74_encode_nibble(nibs[j])

    codes_err = codes.copy()
    for i in range(0, N, 100):
        codes_err[i, 0, 6] ^= 1

    err_audio = np.zeros_like(y_int16)
    fix_audio = np.zeros_like(y_int16)

    for i in range(N):
        nibs_err = []
        nibs_fix = []
        for j in range(4):
            cw = codes_err[i, j]

            d1, d2, d3, d4 = cw[2], cw[4], cw[5], cw[6]
            nibs_err.append((d1 << 3) | (d2 << 2) | (d3 << 1) | d4)

            nibs_fix.append(hamming74_decode_nibble(cw))

        err_audio[i] = nibbles_to_int16(nibs_err)
        fix_audio[i] = nibbles_to_int16(nibs_fix)

    a = err_audio.astype(np.float32) / 32767.0
    f = fix_audio.astype(np.float32) / 32767.0

    return y_float, a, f, sr


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return render_template("index.html", error="Tidak ada file dipilih!")

        filename = secure_filename(file.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(path)

        y, a, f, sr = proses_hamming(path)

        sf.write(os.path.join(app.config["UPLOAD_FOLDER"], "original.wav"), y, sr)
        sf.write(os.path.join(app.config["UPLOAD_FOLDER"], "error.wav"), a, sr)
        sf.write(os.path.join(app.config["UPLOAD_FOLDER"], "corrected.wav"), f, sr)

        return render_template(
            "index.html",
            orig="original.wav",
            err="error.wav",
            fix="corrected.wav",
        )

    return render_template("index.html")


@app.route("/output/<filename>")
def output_file(filename):
    return send_from_directory("static/output", filename)


if __name__ == "__main__":
    app.run(debug=True)
