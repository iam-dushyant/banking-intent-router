from router_ml import data

def test_labels_are_cleaned():
    assert data.clean_label("Refund_not_showing_up") == "refund_not_showing_up"
    assert data.clean_label("reverted_card_payment?") == "reverted_card_payment"

def test_build_splits_and_labels(mini):
    assert set(mini) == {"train", "validation", "test"}
    assert data.label_names(mini) == ["card_arrival", "exchange_rate", "top_up_failed"]
    
    assert set(mini["validation"]["label"]) == {0, 1, 2}
    assert mini["train"].num_rows + mini["validation"].num_rows == 18

def test_sample_per_class(mini):
    few = data.sample_per_class(mini["train"], 2)
    assert few.num_rows == 6
    assert sorted(few["label"]) == [0, 0, 1, 1, 2, 2]