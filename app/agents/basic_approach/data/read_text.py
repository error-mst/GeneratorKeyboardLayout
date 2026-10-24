import itertools

def read_text_in_batch(file_path: str, batch_size: int = 1000):
    with open(file_path, 'r', encoding="utf8") as f:
        while lines := list(itertools.islice(f, batch_size)):
            yield "".join(lines)