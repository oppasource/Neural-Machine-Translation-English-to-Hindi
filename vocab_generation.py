import pickle
import pdb
import collections

##### Creating a dictionary for both the corpus ####
english = open('data/train.en', 'r').read()
english = english.lower().split()

counter = collections.Counter(english)
english = [k for k,v in counter.items() if v > 1]

english = set(english)
english_dict = {}
for x,y in enumerate(english):
	english_dict[y] = x
english_dict['<unk>'] = len(english_dict)


hindi = open('data/train.hi', 'r').read()
hindi = hindi.split()

counter = collections.Counter(hindi)
hindi = [k for k,v in counter.items() if v > 1]

hindi = set(hindi)
hindi_dict = {}
for x,y in enumerate(hindi):
	hindi_dict[y] = x
hindi_dict['<unk>'] = len(hindi_dict)
hindi_dict['<SOS>'] = len(hindi_dict)
hindi_dict['<EOS>'] = len(hindi_dict)


with open('data/english_dict.pickle', 'wb') as handle:
    pickle.dump(english_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/hindi_dict.pickle', 'wb') as handle:
    pickle.dump(hindi_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(len(hindi_dict), len(english_dict))
# pdb.set_trace()