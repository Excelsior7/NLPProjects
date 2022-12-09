from Transformer_implementation import Vocab
from Transformer_implementation import loadModel

model, source_vocab, target_vocab = loadModel(False);

# model.eval();

# with torch.no_grad():
#     ...

print("PARAMETERS ", next(iter(model.parameters())))

print("VOCAB SOURCE ", source_vocab.idx_to_token)
print("VOCAB TARGET ", target_vocab.idx_to_token)
