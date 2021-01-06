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



############ Parameters ############
epochs = 20
lr = 1e-4
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

############# Loss Function ############
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

dev_prev_loss = 1000000   # infinite loss

# Keeping track of loss for a epoch
total_loss = 0
count = 0

#### Reading parallel lines from both the files ####
english_file = open('data/train.en', 'r')
hindi_file = open('data/train.hi', 'r')
# Number of lines(sentences)
num_lines = sum(1 for line in open('data/train.en', 'r'))


for e in range(epochs):
	# Load again if whole file is completed
	if count == 49300:
		# Keeping track of loss for a epoch
		total_loss = 0
		count = 0

		#### Reading parallel lines from both the files ####
		english_file = open('data/train.en', 'r')
		hindi_file = open('data/train.hi', 'r')
		# Number of lines(sentences)
		num_lines = sum(1 for line in open('data/train.en', 'r'))

	for e_line in tqdm(english_file, total = num_lines):
		h_line = hindi_file.readline()

		# Converting sequence to id numbers
		e_line = e_line.lower().split()
		h_line = h_line.split()

		inp = []
		target = []

		for i in e_line:
			try:
				inp.append(english_dict[i])
			except:
				inp.append(english_dict['<unk>'])

		for i in h_line:
			try:
				target.append(hindi_dict[i])
			except:
				target.append(hindi_dict['<unk>'])

				
		target.insert(0, hindi_dict['<SOS>'])
		target.append(hindi_dict['<EOS>'])
		inp.reverse()

		decoder_in_seq = target[:-1]
		decoder_target_seq = target[1:]

		inp = torch.tensor(inp).to(device)
		decoder_in_seq = torch.tensor(decoder_in_seq).to(device)
		decoder_target_seq = torch.tensor(decoder_target_seq).to(device)

		out = model(inp, decoder_in_seq)

		# Backpropagation
		optimizer.zero_grad()
		loss = criterion(out, decoder_target_seq)
		loss.backward()
		optimizer.step()

		total_loss += float(loss.data.cpu().numpy())
		count += 1

		if count%4000 == 0:
			break

	avg_loss = total_loss/count
	print('Epoch: ' + str(e + 1) + '/' + str(epochs) + ' Loss: ' + str(avg_loss) + '\n\n')

	####################### Cross Validation #######################
	# If dev loss has improved then save the model
	dev_loss = 0
	count_dev = 0

	#### Reading parallel lines from DEV files ####
	english_file_dev = open('data/dev.en', 'r')
	hindi_file_dev = open('data/dev.hi', 'r')
	# Number of lines(sentences)
	num_lines_dev = sum(1 for line in open('data/dev.en', 'r'))

	for e_line in tqdm(english_file_dev, total = num_lines_dev):
		h_line = hindi_file_dev.readline()

		# Converting sequence to id numbers
		el_line = e_line.lower().split()
		hl_line = h_line.split()
		e_seq = []
		h_seq = []

		for i in el_line:
			try:
				e_seq.append(english_dict[i])
			except:
				e_seq.append(english_dict['<unk>'])

		for i in hl_line:
			try:
				h_seq.append(hindi_dict[i])
			except:
				h_seq.append(hindi_dict['<unk>'])

		h_seq.insert(0, hindi_dict['<SOS>'])
		h_seq.append(hindi_dict['<EOS>'])
		e_seq.reverse()

		decoder_in_seq = h_seq[:-1]
		decoder_target_seq = h_seq[1:]

		e_seq = torch.tensor(e_seq).to(device)
		decoder_in_seq = torch.tensor(decoder_in_seq).to(device)
		decoder_target_seq = torch.tensor(decoder_target_seq).to(device)

		out = model(e_seq, decoder_in_seq)
		# Calculating loss
		loss = criterion(out, decoder_target_seq)

		dev_loss += float(loss.data.cpu().numpy())
		count_dev += 1

	avg_dev_loss = dev_loss/count_dev
	print('Epoch: ' + str(e + 1) + '/' + str(epochs) + ' Dev Loss: ' + str(avg_dev_loss) + '\n\n')

	if avg_dev_loss < dev_prev_loss:
		print('Model has improved, saving')
		dev_prev_loss = avg_dev_loss
		path = save_path + type(model).__name__ + '_FinalLoss_' + str(avg_dev_loss) + '.pt'
		torch.save(model.state_dict(), path)
