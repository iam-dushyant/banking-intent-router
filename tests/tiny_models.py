from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
from tokenizers.trainers import WordPieceTrainer
from transformers import (
    BertConfig,
    BertModel,
    BertTokenizerFast,
    DistilBertConfig,
    DistilBertModel,
)

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

def tokenizer(texts: list[str]) -> BertTokenizerFast:
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(
        texts, WordPieceTrainer(vocab_size=500, special_tokens=SPECIAL)
    )
    cls, sep = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", cls), ("[SEP]", sep)]
    )
    return BertTokenizerFast(tokenizer_object=tok)

def bert(texts: list[str], out: Path) -> Path:
    tok = tokenizer(texts)
    cfg = BertConfig(
        vocab_size=tok.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
    )
    BertModel(cfg).save_pretrained(out)
    tok.save_pretrained(out)
    return out

def distilbert(texts: list[str], out: Path) -> Path:
    tok = tokenizer(texts)
    cfg = DistilBertConfig(
        vocab_size=tok.vocab_size,
        dim=32,
        n_layers=2,
        n_heads=2,
        hidden_dim=64,
    )
    DistilBertConfig(cfg).save_pretrained(out)
    tok.save_pretrained(out)
    return out