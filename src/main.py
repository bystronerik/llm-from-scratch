import logging

from tokenizer import CharacterTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

with open("datasets/basic.txt", encoding="utf-8") as f:
    text = f.read()

logging.info("Loaded %d characters", len(text))

tokenizer = CharacterTokenizer(text)

encoded = tokenizer.encode(text)

logging.info("Encoded: %s", encoded)

decoded = tokenizer.decode(encoded)

logging.info("Decoded: %s", decoded)
