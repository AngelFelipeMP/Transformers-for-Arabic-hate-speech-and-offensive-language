pre_trained_model = "bert-base-uncased"
transformer = AutoModel.from_pretrained(pre_trained_model)
tokenizer = AutoTokenizer.from_pretrained(pre_trained_model)

max_len = 15
Example1 = "Angel table home car"
Example2 = "bhabha char roofing house get"
Example3 = "I wan to go to the beach for surfing"

pt_batch = tokenizer(
    [Example1, Example2, Example3],
    padding=True,
    truncation=True,
    add_special_tokens=True,
    max_length=max_len,
    return_tensors="pt")

print(pt_batch)
