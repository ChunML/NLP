import requests
from zipfile import ZipFile
import unicodedata
import os
import re

from process_cornell import create_training_files


def maybe_download_and_read_file(url, filename):
    """ Download and unzip training data

    Args:
        url: data url
        filename: zip filename

    Returns:
        Training data: an array containing text lines from the data
    """
    if not os.path.exists(filename):
        session = requests.Session()
        response = session.get(url, stream=True)

        CHUNK_SIZE = 32768
        with open(filename, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    zipf = ZipFile(filename)
    zipf.extractall()

    input_text_file, target_text_file, _ = create_training_files()

    with open(input_text_file, 'r') as f:
        input_lines = f.read()

    input_lines = input_lines.split('\n')[:-1]

    with open(target_text_file, 'r') as f:
        target_lines = f.read()

    target_lines = target_lines.split('\n')[:-1]

    return list(zip(input_lines, target_lines))


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s
