from sentence_transformers import SentenceTransformer
from preprocess import process_text, save_compressed, load_compressed

model = SentenceTransformer('infgrad/stella-base-zh-v2')
total_params = sum(p.numel() for p in model.parameters())
print(f'Num of parameters {total_params}')

data = load_compressed()

train_text = data['train']['content'].to_list()
valid_text = data['valid']['content'].to_list()
test_text = data['test']['content'].to_list()
train_text = [process_text(t) for t in train_text]
valid_text = [process_text(t) for t in valid_text]
test_text = [process_text(t) for t in test_text]

results = {}
results['train'] = model.encode(train_text, batch_size=4, show_progress_bar=True)
results['valid'] = model.encode(valid_text, batch_size=4, show_progress_bar=True)
results['test'] = model.encode(test_text, batch_size=4, show_progress_bar=True)

embedding_path = './data/embeddings.pkl.gz'
save_compressed(results, embedding_path)
