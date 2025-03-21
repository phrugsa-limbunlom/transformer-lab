import torch
import torch.nn as nn
from Seq2SeqTransformer import Seq2SeqTransformer
from translation import DEVICE, generate_square_subsequent_mask
from data_loader import BOS_IDX, EOS_IDX, get_translation_dataloader, get_vocab_size, index_to_de, index_to_eng

# Set random seed for reproducibility
torch.manual_seed(0)

# Model configuration
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
SRC_VOCAB_SIZE = get_vocab_size(SRC_LANGUAGE)
TGT_VOCAB_SIZE = get_vocab_size(TGT_LANGUAGE)
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 2048
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

# Initialize device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize transformer model
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                               NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

def generate_square_subsequent_mask(size, device=DEVICE):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    # src_mask = src_mask.to(DEVICE)

    # Encode the source sequence
    memory = model.encode(src, src_mask=src_mask)  # [seq_len, batch_size, d_model]
    memory = memory.transpose(0, 1)  # [batch_size, seq_len, d_model]
    
    # Initialize decoder input
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    
    for _ in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

if __name__ == "__main__":
    # Initialize model parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)
    transformer.load_state_dict(torch.load('transformer_de_to_en_model.pt', map_location=DEVICE))

    # Load data
    train_dataloader, _ = get_translation_dataloader(batch_size=128, flip_batch=True)
    print(f"Number of batches in dataloader: {len(train_dataloader)}")

    # Process first sample
    print("\nLooking at the first sample:")
    for n, (src, tgt) in enumerate(train_dataloader):
        print(f"\nsample {n}")
        print("german input")
        german_texts = index_to_de(src)
        for i, text in enumerate(german_texts):
            print(f"Sequence {i}: {text}")
            break
        print("\nenglish target")
        english_texts = index_to_eng(tgt)
        for i, text in enumerate(english_texts):
            print(f"Sequence {i}: {text}")
            break
        print("_________")

        break

    # Prepare for inference
    print("Source shape:", src.shape)
    num_tokens = src.shape[1]  # This is the sequence length
    print(f"Number of tokens in source sentence: {num_tokens}")

    # First step of inference - process first sequence in the batch
    src_ = src[0].unsqueeze(0)  # Keep batch dimension
    print("Source shape after processing:", src_.shape)

    # Encode without source mask (encoder can attend to all positions)
    memory = transformer.encode(src_, None)
    print("Memory shape before transpose:", memory.shape)
    
    # Transpose memory to [seq_len, batch_size, feature_dim]
    memory = memory.transpose(0, 1)
    print("Memory shape after transpose:", memory.shape)

    print("--------------------------------")

    # Initialize decoder input
    ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(DEVICE)
    print("Initial decoder input:", ys)

    # Generate target mask
    tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
    print("Target mask:", tgt_mask)

    # First decoding step
    out = transformer.decode(ys, memory, tgt_mask)
    print("Decoder output shape:", out.shape)

    out = out.transpose(0, 1)
    print("Transposed output shape:", out.shape)

    print("Last token shape:", out[:, -1].shape) # [batch_size, feature_dim (EMB_SIZE)]

    # Generate first word
    logits = transformer.generator(out[:, -1])
    print("Logits shape:", logits.shape)

    _, next_word_index = torch.max(logits, dim=1)
    print("English output:", index_to_eng(next_word_index))

    print("--------------------------------")

    # Second step of inference
    next_word_index = next_word_index.item()
    print("Next word index:", next_word_index)

    ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word_index)], dim=0)
    print("Updated decoder input:", ys)

    # Update target mask
    tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
    print("Updated target mask:", tgt_mask)

    # Second decoding step
    out = transformer.decode(ys, memory, tgt_mask) # [seq_len, batch_size, feature_dim (EMB_SIZE)]
    print("Decoder output shape:", out.shape)

    out = out.transpose(0, 1) # [batch_size, seq_len, feature_dim (EMB_SIZE)]
    print("Transposed output shape:", out.shape) 

    print("Last token shape:", out[:, -1].shape) # [batch_size, feature_dim (EMB_SIZE)]

    # Generate second word
    prob = transformer.generator(out[:, -1]) # [batch_size, vocab_size]
    print("Probability (Logits) shape:", prob.shape)
    _, next_word_index = torch.max(prob, dim=1)
    print("English output:", index_to_eng(next_word_index))
    next_word_index = next_word_index.item()

    print("--------------------------------")

    print("Greedy decoding")

    # Greedy decoding
    print("src:", index_to_de(src_))
    src_mask = (torch.zeros(1, 1)).type(torch.bool).to(DEVICE)
    max_len = src.shape[1] + 5  # Use sequence length from source
    output = greedy_decode(transformer, src_, src_mask, max_len=max_len, start_symbol=BOS_IDX)
    print("Output shape:", output.shape)
    print("Output:", index_to_eng(output))
    