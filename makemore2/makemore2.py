import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt # for making figures


words = open('names.txt', 'r').read().splitlines()

# Standardize data: 

tokens = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(tokens)}
stoi["."] = 0  
itos = {i:s for s,i in stoi.items()}

vocab_size = len(tokens)

def build_dataset(data, block_size): 

    X, Y = [], []
    #block_size = 3 
    for word in data: 
        #print(word)
        context = [0] * block_size

        for char in word + ".": 
            
            #print(f"Current Context: {''.join(itos[i] for i in context)}, -> {context}")
            ix = stoi[char]
            #print(f"Next character is: '{char}', String to number: {char} -> {xi}")

            X.append(context)
            Y.append(ix)   

            #print(f"X to Y mapping: {context}, -> {ix}")
            # update context
            context = context[1:] + [ix]

            #print(''.join(itos[i] for i in context), '----->', itos[ix], "\n") 

    
    return(torch.tensor(X), torch.tensor(Y))

# Create Dataset: 
block_size = 3 # context length: how many characters do we take to predict the next one?
X, Y = build_dataset(data=words, block_size=block_size)

# Embedding:
emb_dim = 2
C = torch.rand((vocab_size, emb_dim))

# Initialise hidden Layer: 
n_hl_vars = 100 

W1 = torch.randn((emb_dim * block_size, n_hl_vars))
b1 = torch.randn(n_hl_vars)
W1.shape


emb = C[X]
emb.shape