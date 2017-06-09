import config
from dataloader_memory import Reader
import code
import pickle
import os
import models
import torch
import torch.nn as nn
from torch.nn import Parameter 
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim


def test(model, crit, data, vocab={}, index2word={}, args={}):
	correct_count = 0.
	total_count = 0.
	total_loss = 0.
	f_extract_lstm = open(args.heatmap_file, "w")
	f_extract_lstm.write("<html><head><title>Extract LSTM</title></head><body>")

	for (x, x_mask, y) in data:
		B, T = x.shape
		x = Variable(torch.from_numpy(x), volatile=True).long()
		x_mask = Variable(torch.from_numpy(x_mask), volatile=True)
		y = Variable(torch.from_numpy(y), volatile=True).long()
		if args.use_cuda:
			x = x.cuda()
			y = y.cuda()
			x_mask = x_mask.cuda()
		if args.decompose_type == "beta":
			pred, out = model.decompose(x, x_mask)
		elif args.decompose_type == "gamma":
			pred, out = model.additive_decompose(x, x_mask)
		pred_y = pred.max(1)[1]

		score = (out[:,:,0] - out[:,:,1]).data
		score[x_mask.data == 0] = -99999
		negative_indicator = torch.max(score, 1)[1]
		score[x_mask.data == 0] = 99999
		positive_indicator = torch.min(score, 1)[1]
		out = torch.exp(out).data * 10
		lengths = x_mask.data.sum(1).long().view(-1)
		for i in range(B):
			f_extract_lstm.write("<p>predicted: " + str(pred_y.data[i][0]) + " correct label: " + str(y.data[i]))
			for j in range(lengths[i]):
				# code.interact(local=locals())
				if negative_indicator[i][0] == j:
					# print("negative")
					f_extract_lstm.write("<span style='font-size:" + str(out[i][j][pred_y.data[i]][0]) + \
						";color:green' title='" + str(out[i][j][pred_y.data[i]][0]) + "'>" + str(index2word[x.data[i][j]]) + " </span>")
				elif positive_indicator[i][0] == j:
					# print("positive")
					f_extract_lstm.write("<span style='font-size:" + str(out[i][j][pred_y.data[i]][0]) + \
						";color:red' title='" + str(out[i][j][pred_y.data[i]][0]) + "'>" + str(index2word[x.data[i][j]]) + " </span>")
				else:
					f_extract_lstm.write("<span style='font-size:" + str(out[i][j][pred_y.data[i]][0]) + \
						"' title='" + str(out[i][j][pred_y.data[i]][0]) + "'>" + str(index2word[x.data[i][j]]) + " </span>")
			# print("<span style='font-size:" + str(out[i][j][pred_y.data[i]][0]) + \
					# "' title='" + str(out[i][j][pred_y.data[i]][0]) + "'>" + str(index2word[x.data[i][j]]) + " </span>")
			f_extract_lstm.write("</p>\n")
		# code.interact(local=locals())
		
		correct_count += (pred_y == y).data.sum()
		total_count += B
		loss = crit(pred, y)
		total_loss += loss.data[0] * total_count

	f_extract_lstm.write("</body></html>")
	f_extract_lstm.close()
	return total_loss, correct_count, total_count

def main(args):
	if os.path.isfile("vocab.pkl"):
		vocab = pickle.load(open("vocab.pkl", "rb"))
	else:
		print("vocab file not found")
		exit(-1)

	reader = Reader(args)
	reader.load_test_data()
	reader.encode_test(vocab)
	test_data = reader.gen_examples(reader.test_data)
	pickle.dump(test_data, open("test_data.pkl", "wb"))

	index2word = {v: k for k, v in vocab.items()}

	if os.path.isfile(args.model_file):
		model = torch.load(args.model_file)
	else:
		print("model file %s not found" % args.model_file)

	crit = nn.CrossEntropyLoss()

	test_loss, test_correct_count, test_count = test(model, crit, test_data, vocab=vocab, index2word=index2word, args=args)
	test_acc = test_correct_count/test_count
	test_loss = test_loss/test_count
	print("test accuracy: %f, test loss: %f" % (test_acc, test_loss))
	return -1
	
if __name__ == "__main__":
	args = config.get_args()
	main(args)

