from tqdm import tqdm
from data_loader import get_translation_dataloader, get_vocab_size, index_to_eng, index_to_de
from Seq2SeqTransformer import Seq2SeqTransformer
import torch
import torch.nn as nn
from timeit import default_timer as timer
from matplotlib import pyplot as plt

# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# Make sure the tokens are in order of their indices to properly insert them in vocab
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Reset the dataloader for training
train_dataloader, val_dataloader = get_translation_dataloader(batch_size=128, flip_batch=True)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_square_subsequent_mask(size, device=DEVICE):
    mask = (torch.triu(torch.ones((size, size), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, device=DEVICE):
    # Get sequence lengths from the correct dimension
    src_seq_len = src.shape[1]  # [batch_size, seq_len]
    tgt_seq_len = tgt.shape[1]  # [batch_size, seq_len]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)

    # Adjust padding masks to match the sequence lengths
    src_padding_mask = (src == PAD_IDX)  # [batch_size, seq_len]
    tgt_padding_mask = (tgt == PAD_IDX)  # [batch_size, seq_len]

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

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

def train_model(model, optimizer, train_dataloader):
    model.train()
    losses = 0

    # Wrap train_dataloader with tqdm for progress logging
    train_iterator = tqdm(train_dataloader, desc="Training", leave=False)

    for src, tgt in train_iterator:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # Remove last token from target sequence for input (batch size, seq_len)
        tgt_input = tgt[:, :-1]
        
        # Create masks with the correct sequence lengths
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        src_mask = src_mask.to(DEVICE)
        tgt_mask = tgt_mask.to(DEVICE)
        src_padding_mask = src_padding_mask.to(DEVICE)
        tgt_padding_mask = tgt_padding_mask.to(DEVICE)

        # Forward pass
        logits = model(src=src, 
                      tgt=tgt_input, 
                      src_mask=src_mask, 
                      tgt_mask=tgt_mask, 
                      src_padding_mask=src_padding_mask, 
                      tgt_padding_mask=tgt_padding_mask, 
                      memory_key_padding_mask=src_padding_mask)
        logits = logits.to(DEVICE)

        optimizer.zero_grad()

        # Remove first token from target sequence for loss calculation (batch size, seq_len)
        tgt_out = tgt[:, 1:]

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        # Update tqdm progress bar with the current loss
        train_iterator.set_postfix(loss=loss.item())

    return losses / len(list(train_dataloader))
 
def evaluate(model):
    model.eval()
    losses = 0

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        
        # Remove last token from target sequence for input
        tgt_input = tgt[:, :-1]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        
        logits = model(src=src, 
                      tgt=tgt_input, 
                      src_mask=src_mask, 
                      tgt_mask=tgt_mask, 
                      src_padding_mask=src_padding_mask, 
                      tgt_padding_mask=tgt_padding_mask, 
                      memory_key_padding_mask=src_padding_mask)
        
        # Remove first token from target sequence for loss calculation
        tgt_out = tgt[:, 1:]


        # Original logits shape: [128, 50, 32000]
        # After reshape: [6400, 32000]
        # This means:
        # - 6400 total tokens to predict (128 sequences * 50 tokens each)
        # - Each token has 32000 possible vocabulary choices

        # Original tgt_out shape: [128, 50]
        # After reshape: [6400]
        # This means:
        # - 6400 total tokens (128 sequences * 50 tokens each)
        # - Each number represents the correct vocabulary index for that position

        # Now each row in logits (6400 rows) corresponds to exactly one target token

        # logits.reshape(-1, logits.shape[-1])
        # -1 in the first dimension means "automatically calculate this dimension"
        # logits.shape[-1] keeps the last dimension (vocabulary_size) unchanged
        # This effectively flattens the first two dimensions (batch_size and sequence_length) into one dimension

        # tgt_out.reshape(-1)
        # -1 in the first dimension means "automatically calculate this dimension"
        # This effectively flattens the second dimension (sequence_length) into one dimension

        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    
    return losses / len(list(val_dataloader))

if __name__ == "__main__":

    # Data Loader
    train_dataloader, _ = get_translation_dataloader(batch_size=128, flip_batch=True)

    # Print the number of batches in the dataloader
    print(f"Number of batches in dataloader: {len(train_dataloader)}")

    # First, let's look at a few samples
    print("\nLooking at the first sample:")
    for n, (german, english) in enumerate(train_dataloader):
        if n >= 1:  # Only look at first sample
            break
        print(f"\nsample {n}")
        print("german input")
        german_texts = index_to_de(german)
        for i, text in enumerate(german_texts):
            print(f"Sequence {i}: {text}")
        print("\nenglish target")
        english_texts = index_to_eng(english)
        for i, text in enumerate(english_texts):
            print(f"Sequence {i}: {text}")
        print("_________")
        
    # Initialize model
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                   NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    # Initialize model parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    transformer = transformer.to(DEVICE)

    # Define loss function and optimizer
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    # Training loop
    train_losses = []
    val_losses = []
    NUM_EPOCHS = 10

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_model(transformer, optimizer, train_dataloader)
        train_losses.append(train_loss)
        end_time = timer()
        
        val_loss = evaluate(transformer)
        val_losses.append(val_loss)
        
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
              f"Epoch time = {(end_time - start_time):.3f}s")

    # Save the trained model
    torch.save(transformer.state_dict(), 'transformer_de_to_en_model.pt')

    # Plot training and validation losses
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.plot(epochs, val_losses, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()











