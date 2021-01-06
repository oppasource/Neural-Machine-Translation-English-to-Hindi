import pdb
import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import model as m
import coverage_model as cm

##### Loading the dictionary for both the corpus ####
with open('data/english_dict.pickle', 'rb') as handle:
    english_dict = pickle.load(handle)

with open('data/hindi_dict.pickle', 'rb') as handle:
    hindi_dict = pickle.load(handle)
# Invert the hindi_dict for id2word
hindi_dict_reverse = {v: k for k, v in hindi_dict.items()}


############ Parameters ############
embedding_size = 128
hidden_size = 128

vocab_size_english = len(english_dict)
vocab_size_hindi = len(hindi_dict)

save_path = 'trained_models/'

######### Initialize model ########

# Using gpu if available else cpu
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# model = m.seq2seq(embedding_size, hidden_size, vocab_size_english, vocab_size_hindi, device)
# model = m.attention(embedding_size, hidden_size, vocab_size_english, vocab_size_hindi, device)
# model = m.global_attention_1(embedding_size, hidden_size, vocab_size_english, vocab_size_hindi, device)
# model = m.global_attention_2(embedding_size, hidden_size, vocab_size_english, vocab_size_hindi, device)
# model = m.global_attention_3(embedding_size, hidden_size, vocab_size_english, vocab_size_hindi, device)
model = cm.coverage(embedding_size, hidden_size, vocab_size_english, vocab_size_hindi, device)
model = model.to(device)

trained_model_name = 'coverage.pt'

if torch.cuda.is_available():
	model.load_state_dict(torch.load(save_path + trained_model_name))
else:
	model.load_state_dict(torch.load(save_path + trained_model_name, map_location='cpu'))

e = input('enter sentence: ')
e = e.lower().strip().split()
inp = []

for i in e:
	try:
		inp.append(english_dict[i])
	except:
		inp.append(english_dict['<unk>'])

e.reverse()
inp = torch.tensor(inp).to(device)
out_seq = [hindi_dict['<SOS>']]

# Break loop when <EOS> appears
while(1):

	decoder_in_seq = torch.tensor(out_seq).to(device)
	out = model(inp, decoder_in_seq)
	next_word_id = int(torch.max(out, 1)[1].cpu().numpy()[-1])
	
	#print(hindi_dict_reverse[next_word_id])
	if next_word_id == hindi_dict['<EOS>']:
		break
	else:
		out_seq.append(next_word_id)

out_seq = out_seq[1:]
out_seq = [hindi_dict_reverse[i] for i in out_seq]
out_sentence = ' '.join(out_seq)
print(out_sentence)

	

