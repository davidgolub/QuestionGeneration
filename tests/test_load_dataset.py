from data_loaders.card_loader import CardLoader 

card_loader = CardLoader(base_path='card2code/third_party/magic')
print(card_loader.train_dataset['inputs'][0])
print(card_loader.train_dataset['outputs'][0])