import torch
from router_ml import lstm

def test_tokenize_keeps_contractions_and_punctuation():
    assert lstm.tokenize("Card hasn't arrived?") == ["card", "hasn't", "arrived", "?"]

def test_vocab_maps_unknown_words():
    vocab = lstm.Vocab.build(["my card", "my card"], min_freq=2)
    assert vocab.encode("my zebra") == [vocab.stoi["my"], lstm.UNK]

def test_padding_does_not_change_the_output():
    torch.manual_seed(0)
    model = lstm.BiLSTMClassifier(vocab_size=10, n_classes=3).eval()
    short = torch.tensor([[2, 3, 4]])
    padded = torch.tensor([[2, 3, 4, lstm.PAD, lstm.PAD]])
    assert torch.allclose(model(short), model(padded), atol=1e-5)

def test_training_beats_chance(mini):
    _, _, val_acc = lstm.train(mini, epochs=20, device="cpu", batch_size=4)
    assert val_acc > 0.5