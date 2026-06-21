from tinychat.generate import generate
from tinychat.model import build_model
from tinychat.tokenizer import load_tokenizer, train_tokenizer

CORPUS = ["Once upon a time there was a little cat who liked to play."] * 100


def _tok(tmp_path):
    p = str(tmp_path / "tok.json")
    train_tokenizer(CORPUS, save_path=p, vocab_size=512)
    return load_tokenizer(p)


def test_generate_returns_decodable_string(small_cfg, tmp_path):
    tok = _tok(tmp_path)
    model = build_model(small_cfg, "fp16")
    out = generate(model, tok, "Once upon a time", max_new_tokens=10, seed=0)
    assert isinstance(out, str)
    # continuation is at most max_new_tokens tokens
    assert len(tok.encode(out).ids) <= 10


def test_generate_deterministic_under_seed(small_cfg, tmp_path):
    tok = _tok(tmp_path)
    model = build_model(small_cfg, "fp16")
    a = generate(model, tok, "Once upon a time", max_new_tokens=15, seed=42)
    b = generate(model, tok, "Once upon a time", max_new_tokens=15, seed=42)
    assert a == b
