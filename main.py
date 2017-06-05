import config
from dataloader import Reader
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


def eval(reader, model, crit, data_split="dev", vocab={}, args={}):
	reader.reset_dev()
	correct_count = 0.
	total_count = 0.
	total_loss = 0.
	while 1:
		(x, x_mask, y), last_batch = reader.get_batch(data_split=data_split, vocab=vocab)
		B, T = x.shape
		x = Variable(torch.from_numpy(x)).long()
		x_mask = Variable(torch.from_numpy(x_mask))
		y = Variable(torch.from_numpy(y)).long()
		if args.use_cuda:
			x = x.cuda()
			y = y.cuda()
			x_mask = x_mask.cuda()
		pred = model(x, x_mask)
		pred_y = pred.max(1)[1]
		correct_count += (pred_y == y).data.sum()
		total_count += B
		loss = crit(pred, y)
		total_loss += loss.data[0] * total_count

	return total_loss, correct_count, total_count

def test(reader, model, crit, data_split="test", vocab={}, args={}):
	reader.reset_test()
	correct_count = 0.
	total_count = 0.
	total_loss = 0.
	while 1:
		(x, x_mask, y), last_batch = reader.get_batch(data_split=data_split, vocab=vocab)
		B, T = x.shape
		x = Variable(torch.from_numpy(x)).long()
		x_mask = Variable(torch.from_numpy(x_mask))
		y = Variable(torch.from_numpy(y)).long()
		if args.use_cuda:
			x = x.cuda()
			y = y.cuda()
			x_mask = x_mask.cuda()
		pred = model(x, x_mask)
		pred_y = pred.max(1)[1]
		correct_count += (pred_y == y).data.sum()
		total_count += B
		loss = crit(pred, y)
		total_loss += loss.data[0] * total_count

	return total_loss, correct_count, total_count

def main(args):
	reader = Reader(args)
	reader.init_train()
	if os.path.isfile(args.vocab_file):
		vocab = pickle.load(open("vocab.pkl", "rb"))
	else:
		vocab = reader.build_dict()
		pickle.dump(vocab, open("vocab.pkl", "wb"))

	index2word = {v: k for k, v in vocab.items()}

	reader.reset_train()
	reader.init_dev()

	if os.path.isfile(args.model_file):
		model = torch.load(args.model_file)
	else:
		model = getattr(models, args.model)(args)
	if args.use_cuda:
		model.cuda()

	crit = nn.CrossEntropyLoss()

	learning_rate = args.learning_rate
	optimizer = getattr(optim, args.optimizer)(model.parameters(), lr=learning_rate)

	best_dev_acc = 0.
	epoch = 0
	batch = 0
	while 1:
		(x, x_mask, y), last_batch = reader.get_batch(data_split="train", vocab=vocab)
		if last_batch:
			dev_loss, dev_correct_count, dev_count = eval(reader, model, crit, data_split="dev", vocab=vocab, args=args)
			dev_acc = dev_correct_count/dev_count
			dev_loss = dev_loss/dev_count
			print("dev accuracy: %f, dev loss: %f" % (dev_acc, dev_loss))
			if dev_acc > best_dev_acc:
				best_dev_acc = dev_acc
				print("best dev accuracy: %f" % best_dev_acc)
				torch.save(model, args.model_file)
				print("model saved.")

			epoch += 1

			if epoch >= args.num_epochs:
				break
			reader.reset_train()
		else:
			x = Variable(torch.from_numpy(x)).long()
			x_mask = Variable(torch.from_numpy(x_mask))
			y = Variable(torch.from_numpy(y)).long()
			if args.use_cuda:
				x = x.cuda()
				y = y.cuda()
				x_mask = x_mask.cuda()
			pred = model(x, x_mask)
			loss = crit(pred, y)

			# code.interact(local=locals())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if args.use_cuda:
				loss = loss.cpu()

			print("batch %d training loss: %f" %(batch, loss.data[0]))
			batch += 1
		# code.interact(local=locals())



if __name__ == "__main__":
	args = config.get_args()
	main(args)

