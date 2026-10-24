from collections import Counter

from data.read_text import read_text_in_batch

def get_ngrams(text: str, n: int=2) -> list[str]:
    return [text[i: i + n] for i in range(len(text) - n + 1)]

def text_profile(file_path: str):
    char_counter = Counter()
    bigram_counter = Counter()
    trigram_counter = Counter()

    batched_text = read_text_in_batch(file_path)
    for batch in batched_text:
        char_counter.update(batch)

        # обработка н-грамм
        bigrams = Counter(get_ngrams(batch, n=2))
        trigrams = Counter(get_ngrams(batch, n=3))

        bigram_counter.update(bigrams)
        trigram_counter.update(trigrams)

    return {
        "char_cnts": char_counter,
        "bigram_cnts": bigram_counter,
        "trigram_cnts": trigram_counter
    }