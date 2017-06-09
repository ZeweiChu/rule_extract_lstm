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


def eval(model, crit, data, vocab={}, args={}):
	correct_count = 0.
	total_count = 0.
	total_loss = 0.

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


		out = torch.exp(out).data * 30
		lengths = x_mask.data.sum(1).long().view(-1)
		for i in range(B):
			f_extract_lstm.write("<p>predicted: " + str(pred_y.data[i][0]) + " correct label: " + str(y.data[i]))
			for j in range(lengths[i]):
				f_extract_lstm.write("<span style='font-size:" + str(out[i][j][pred_y.data[i]][0]) + \
					"' title='" + str(out[i][j][pred_y.data[i]][0]) + "'>" + str(index2word[x.data[i][j]]) + " </span>")
			# print("<span style='font-size:" + str(out[i][j][pred_y.data[i]][0]) + \
					# "' title='" + str(out[i][j][pred_y.data[i]][0]) + "'>" + str(index2word[x.data[i][j]]) + " </span>")
			f_extract_lstm.write("</p>")
		# code.interact(local=locals())
		
		correct_count += (pred_y == y).data.sum()
		total_count += B
		loss = crit(pred, y)
		total_loss += loss.data[0] * total_count

	f_extract_lstm.write("</body></html>")
	f_extract_lstm.close()
	return total_loss, correct_count, total_count

	

def main(args):
	if os.path.isfile("vocab.pkl") and os.path.isfile("train_data.pkl") and os.path.isfile("dev_data.pkl"):
		vocab = pickle.load(open("vocab.pkl", "rb"))
		train_data = pickle.load(open("train_data.pkl", "rb"))
		dev_data = pickle.load(open("dev_data.pkl", "rb"))
	else:
		reader = Reader(args)
		reader.load_data()
		vocab = reader.build_dict()
		pickle.dump(vocab, open("vocab.pkl", "wb"))
		reader = Reader(args)
		reader.load_data()
		reader.encode(vocab)
		train_data = reader.gen_examples(reader.train_data)
		pickle.dump(train_data, open("train_data.pkl", "wb"))
		dev_data = reader.gen_examples(reader.dev_data)
		pickle.dump(dev_data, open("dev_data.pkl", "wb"))

	index2word = {v: k for k, v in vocab.items()}

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
	dev_loss, dev_correct_count, dev_count = eval(model, crit, dev_data, vocab=vocab, args=args)
	dev_acc = dev_correct_count/dev_count
	dev_loss = dev_loss/dev_count
	print("dev accuracy: %f, dev loss: %f" % (dev_acc, dev_loss))

	if dev_acc > best_dev_acc:
		best_dev_acc = dev_acc
		print("best dev accuracy: %f" % best_dev_acc)
		torch.save(model, args.model_file)
		print("model saved.")
	epoch = 0
	total_train_batch = len(train_data)

	iteration = 0
	while 1:
		x, x_mask, y = train_data[iteration % total_train_batch]
		
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

		print("iteration %d training loss: %f" %(iteration, loss.data[0]))
		iteration += 1
		# code.interact(local=locals())
		if iteration % args.eval_iterations == 0:
			dev_loss, dev_correct_count, dev_count = eval(model, crit, dev_data, vocab=vocab, args=args)
			dev_acc = dev_correct_count/dev_count
			dev_loss = dev_loss/dev_count
			print("dev accuracy: %f, dev loss: %f" % (dev_acc, dev_loss))
			if dev_acc > best_dev_acc:
				best_dev_acc = dev_acc
				print("best dev accuracy: %f" % best_dev_acc)
				torch.save(model, args.model_file)
				print("model saved.")

			if iteration / total_train_batch >= args.num_epochs:
				break



if __name__ == "__main__":
	args = config.get_args()
	main(args)


