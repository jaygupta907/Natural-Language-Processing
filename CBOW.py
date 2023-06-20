import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from tqdm import tqdm


class CBOW(nn.Module):
    def __init__(self, embedding_dim=100, vocab_size=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeddings = self.embedding(inputs).mean(1).squeeze(1)
        return self.linear(embeddings)


def create_dataset():

    with open("text.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    nlp = spacy.load("en_core_web_sm")
    tokenized_text = [token.text for token in nlp(raw_text)]
    vocab = set(tokenized_text)

    word_to_idx = {word: i for i, word in enumerate(vocab)}
    idx_to_word = {i: word for i, word in enumerate(vocab)}

    data = []
    for i in range(2, len(tokenized_text)-2):
        context = [
            tokenized_text[i-2],
            tokenized_text[i-1],
            tokenized_text[i+1],
            tokenized_text[i+2]
        ]
        target = tokenized_text[i]

        context_idxs = [word_to_idx[w] for w in context]
        target_idx = word_to_idx[target]
        data.append((context_idxs, target_idx))

    return data, word_to_idx, idx_to_word


def main():
    EMBEDDING_SIZE = 100
    data, word_to_idx, idx_to_word = create_dataset()
    loss_func = nn.CrossEntropyLoss()
    net = CBOW(EMBEDDING_SIZE, vocab_size=len(word_to_idx))
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    context_data = torch.tensor([ex[0] for ex in data])
    labels = torch.tensor([ex[1] for ex in data])
    print(context_data.shape)
    print(labels.shape)

    dataset = torch.utils.data.TensorDataset(context_data, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=True)
    loop = tqdm(dataloader, leave=True)
    for epoch in range(1000):
        for idx, (context, label) in enumerate(loop):
            print(context.shape)
            print(label.shape)
            output = net(context)
            print(output.shape)
            loss = loss_func(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
    predict("absorbing and carbon dioxide", net, word_to_idx, idx_to_word)


def predict(text, net, word_to_idx, idx_to_word):
    nlp = spacy.load("en_core_web_sm")
    tokenized_text = [token.text for token in nlp(text)]
    idxs = [word_to_idx[word] for word in tokenized_text]
    context = torch.tensor(idxs).unsqueeze(dim=0)
    prediction = torch.argmax(net(context)).item()
    print(idx_to_word[prediction])


if __name__ == "__main__":
    main()
