import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k, multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List, Tuple, Generator


SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# initialize the tokenizer for the source and target languages
token_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3

special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

vocab_transform = {}

# extract the tokens from the training set of each sentence
# data_iter is the pair of source and target sentences
# 0 is the index of the source language and 1 is the index of the target language
def yield_tokens(data_iter: Iterable, language: str) -> Generator[str, None, None]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# Build the vocabulary from the training set
for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    sorted_dataset = sorted(train_iter, key=lambda x: len(x[0].split( )))
    vocab_transform[language] = build_vocab_from_iterator(yield_tokens(sorted_dataset, language),
                                                          min_freq=1,
                                                          specials=special_symbols,
                                                          special_first=True)
# set unknown or out of vocabulary tokens to the index 0
for language in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[language].set_default_index(UNK_IDX)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tensor_transform_s(token_ids: List[int]) -> torch.Tensor:
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.flip(torch.tensor(token_ids), dims=(0,)),
                      torch.tensor([EOS_IDX])))

def tensor_transform_t(token_ids: List[int]) -> torch.Tensor:
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))


# transformatin pipeline where output of one transform is input of the next transform
# transforms is a list of functions
def sequential_transforms(*transforms):
    def func(text_input):
        for transform in transforms:
            text_input = transform(text_input)
        return text_input
    return func

text_transform = {}

# calling collate_fn after transform text (tokenization, vocabulary, tensor transformation (adds <bos>, <eos> tokens))
# collate_fn : transform the batch of data into a batch of tensors and pad the sequences to the same length
def collate_fn(batch: List[Tuple[str, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
       src_sequences = text_transform[SRC_LANGUAGE](src_sample.rstrip("\n"))
       src_sequences = torch.tensor(src_sequences, dtype=torch.int64)
      
       tgt_sequences = text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n"))
       tgt_sequences = torch.tensor(tgt_sequences, dtype=torch.int64)
       
       src_batch.append(src_sequences)
       tgt_batch.append(tgt_sequences)
    
    # add pad tokens to the sequences to make them of the same length
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    src_batch = src_batch.t() #.t() is transpose, original is (max_sequence_length, batch_size), after transpose is (batch_size, max_sequence_length)
    tgt_batch = tgt_batch.t()

    return src_batch.to(device), tgt_batch.to(device)

def get_translation_dataloader(batch_size: int = 4, flip_batch: bool = False):
    
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    sorted_train_iter = sorted(train_iter, key=lambda x: len(x[0].split( ))) # sort the training data by the length of the source sentence
    
    if flip_batch:
        # transform the source sentence by tokenizing, vocabulary, and tensor transformation (adds <bos>, <eos> tokens)
        text_transform[SRC_LANGUAGE] = sequential_transforms(token_transform[SRC_LANGUAGE], vocab_transform[SRC_LANGUAGE], tensor_transform_s)
    else:
        text_transform[SRC_LANGUAGE] = sequential_transforms(token_transform[SRC_LANGUAGE], vocab_transform[SRC_LANGUAGE], tensor_transform_t)
    
    text_transform[TGT_LANGUAGE] = sequential_transforms(token_transform[TGT_LANGUAGE], vocab_transform[TGT_LANGUAGE], tensor_transform_t)

    train_dataloader = DataLoader(sorted_train_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    
    valid_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    sorted_valid_iter = sorted(valid_iter, key=lambda x: len(x[0].split( )))
    
    valid_dataloader = DataLoader(sorted_valid_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    
    return train_dataloader, valid_dataloader

def index_to_eng(seq_eng):
    # Handle batch of sequences
    if len(seq_eng.shape) > 1:
        return [" ".join([vocab_transform['en'].get_itos()[index.item()] for index in seq]) for seq in seq_eng]
    return " ".join([vocab_transform['en'].get_itos()[index.item()] for index in seq_eng])

def index_to_de(seq_de):
    # Handle batch of sequences
    if len(seq_de.shape) > 1:
        return [" ".join([vocab_transform['de'].get_itos()[index.item()] for index in seq]) for seq in seq_de]
    return " ".join([vocab_transform['de'].get_itos()[index.item()] for index in seq_de])

def get_vocab_size(language: str) -> int:
    return len(vocab_transform[language])