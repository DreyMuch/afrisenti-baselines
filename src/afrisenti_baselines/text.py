"""Light, language-agnostic tweet normalisation.

Nothing here is language-specific on purpose: the same cleaning has to work for
Igbo diacritics, Twi's ɛ and ɔ, Pidgin spellings and Swahili.
"""

import re
import unicodedata

_URL = re.compile(r"https?://\S+|www\.\S+")
_USER = re.compile(r"@\w+")
_REPEAT = re.compile(r"(.)\1{2,}")
_SPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    # NFC so that "ụ" typed as u + combining dot matches the precomposed form
    text = unicodedata.normalize("NFC", text).lower()
    text = _URL.sub(" url ", text)
    text = _USER.sub(" @user ", text)
    text = text.replace("#", " ")
    # "soooo" -> "soo": keeps the emphasis signal without exploding the vocabulary
    text = _REPEAT.sub(r"\1\1", text)
    return _SPACE.sub(" ", text).strip()
