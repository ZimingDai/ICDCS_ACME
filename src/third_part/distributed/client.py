import os,sys
import time
import torch
import copy
import numpy as np
from logger import Logger
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from utils.group_lasso_optimizer import group_lasso_decay
from pruning_engine import pytorch_pruning, PruningConfigReader, prepare_pruning_list
from utils.utils import save_checkpoint, adjust_learning_rate, AverageMeter, accuracy, load_model_pytorch, dynamic_network_change_local, get_conv_sizes, connect_gates_with_parameters_for_flops
class Client:
    def __init__(self, args, model, train_dataset, test_datset, train_idxs, test_idxs, id=-1,train_writer=None):
        self.args = args
        self.local_model = model.cuda()
        self.client_id = id
        self.train_dataset = train_dataset
        self.test_datset = test_datset
        self.id = id
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=args.local_bs,
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            list(train_idxs)))
        self.test_loader = torch.utils.data.DataLoader(self.test_datset, batch_size=args.local_bs,
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            list(test_idxs)),shuffle = False)
        self.use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.train_writer = train_writer

    def prepare_model(self):
        # aux function to get size of feature maps
        # First it adds hooks for each conv layer
        # Then runs inference with 1 image
        self.output_sizes = get_conv_sizes(self.args, self.local_model)

        if self.use_cuda and not self.args.mgpu:
            self.local_model = self.local_model.to(self.device)
        elif self.args.distributed:
            self.local_model.cuda()
            print("\n\n WARNING: distributed pruning was not verified and might not work correctly")
            self.local_model = torch.nn.parallel.DistributedDataParallel(self.local_model)
        elif self.args.mgpu:
            self.local_model = torch.nn.DataParallel(self.local_model).cuda()
        else:
            self.local_model = self.local_model.to(self.device)

        self.args.distributed = self.args.world_size > 1
        print("model is set to device: use_cuda {}, args.mgpu {}, agrs.distributed {}".format(self.use_cuda, self.args.mgpu,
                                                                                              self.args.distributed))
        self.weight_decay = self.args.wd
        # fix network for oracle or criteria computation,不会对模型的权重施加任何正则化惩罚
        if self.args.fixed_network:
            self.weight_decay = 0.0

        # remove updates from gate layers, because we want them to be 0 or 1 constantly
        if 1:
            self.parameters_for_update = []
            self.parameters_for_update_named = []
            for name, m in self.local_model.named_parameters():
                # if "gate" not in name:
                if name!='classifier.five.2.weight' and name!='classifier.five.4.weight' and name!='classifier.five.8.weight' and name!='classifier.five.10.weight':
                    self.parameters_for_update.append(m)
                    self.parameters_for_update_named.append((name, m))
                else:
                    print("skipping parameter", name, "shape:", m.shape)

        total_size_params = sum([np.prod(par.shape) for par in self.parameters_for_update])
        print("Total number of parameters, w/o usage of bn consts: ", total_size_params)

        self.optimizer = optim.SGD(self.parameters_for_update, lr=self.args.lr, momentum=self.args.momentum, weight_decay=self.weight_decay)

        # 帮助优化器实现组lasso（带有非常小的权重，不会影响训练）
        # 将用于计算网络中剩余的FLOPs和参数数量
        if 1:
            # helping optimizer to implement group lasso (with very small weight that doesn't affect training)
            # will be used to calculate number of remaining flops and parameters in the network
            self.group_wd_optimizer = group_lasso_decay(self.parameters_for_update, group_lasso_weight=self.args.group_wd_coeff,
                                                   named_parameters=self.parameters_for_update_named,
                                                   output_sizes=self.output_sizes)

        cudnn.benchmark = True

        # define objective
        self.criterion = nn.CrossEntropyLoss()

        ###=======================added for pruning
        # logging part
        self.log_save_folder = "%s" % self.args.name
        if not os.path.exists(self.log_save_folder):
            os.makedirs(self.log_save_folder)

        if not os.path.exists("%s/models%s" % (self.log_save_folder, str(self.id))):
            os.makedirs("%s/models%s" % (self.log_save_folder, str(self.id)))

        time_point = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        textfile = "%s/log%s_%s.txt" % (self.log_save_folder, str(self.id),time_point)
        stdout = Logger(textfile)
        sys.stdout = stdout
        print(" ".join(sys.argv))

    def prepare_pruning(self):
        # initializing parameters for pruning
        # we can add weights of different layers or we can add gates (multiplies output with 1, useful only for gradient computation)
        self.pruning_engine = None
        if self.args.pruning:
            self.pruning_settings = dict()
            if not (self.args.pruning_config is None):
                self.pruning_settings_reader = PruningConfigReader()
                self.pruning_settings_reader.read_config(self.args.pruning_config)
                self.pruning_settings = self.pruning_settings_reader.get_parameters()

            has_attribute = lambda x: any([x in a for a in sys.argv])

            if has_attribute('pruning-momentum'):
                self.pruning_settings['pruning_momentum'] = vars(self.args)['pruning_momentum']
            if has_attribute('pruning-method'):
                self.pruning_settings['method'] = vars(self.args)['pruning_method']

            self.pruning_parameters_list = prepare_pruning_list(self.pruning_settings, self.local_model, model_name=self.args.model,
                                                           pruning_mask_from=self.args.pruning_mask_from, name=self.args.name)
            print("Total pruning layers:", len(self.pruning_parameters_list))

            self.folder_to_write = "%s" % self.log_save_folder + "/"
            self.log_folder = self.folder_to_write

            self.pruning_engine = pytorch_pruning(self.pruning_parameters_list, pruning_settings=self.pruning_settings,
                                             log_folder=self.log_folder,id = self.id,train_writer = self.train_writer)

            self.pruning_engine.connect_tensorboard(self.train_writer)
            self.pruning_engine.dataset = self.args.dataset
            self.pruning_engine.model = self.args.model
            self.pruning_engine.pruning_mask_from = self.args.pruning_mask_from
            self.pruning_engine.load_mask()
            gates_to_params = connect_gates_with_parameters_for_flops(self.args.model, self.parameters_for_update_named)
            self.pruning_engine.gates_to_params = gates_to_params

    def train(self,args, model, device, train_loader, optimizer, epoch, criterion, train_writer=None,is_prune=False):
        """Train for one epoch on the training set also performs pruning"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        loss_tracker = 0.0
        acc_tracker = 0.0
        loss_tracker_num = 0
        res_pruning = 0
        sum_train_acc = 0.0
        sum_train_loss = 0.0

        model.train()
        if args.fixed_network:
            # if network is fixed then we put it to eval mode
            model.eval()

        end = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # make sure that all gradients are zero
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.detach_()  # 将梯度张量从计算图中分离
                    p.grad.zero_()  # 将梯度张量清零

            output = model(data)
            logits = output.logits
            loss = criterion(logits, target)

            if args.pruning:
                # useful for method 40 and 50 that calculate oracle
                self.pruning_engine.run_full_oracle(model, data, target, criterion, initial_loss=loss.item())

            # measure accuracy and record loss
            losses.update(loss.item(), data.size(0))
            output = output.logits
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))
            acc_tracker += prec1.item()

            loss_tracker += loss.item()

            loss_tracker_num += 1

            if args.pruning:
                if self.pruning_engine.needs_hessian:
                    self.pruning_engine.compute_hessian(loss)

            if not (args.pruning and args.pruning_method == 50):  # 剪枝但是剪枝方法不是50
                self.group_wd_optimizer.step()

            loss.backward()

            # add gradient clipping
            if not args.no_grad_clip:
                # found it useless for our experiments
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # step_after will calculate flops and number of parameters left, step_after将计算FLOPS和剩余参数的数量
            # needs to be launched before the main optimizer, 需要在主优化器之前启动
            # otherwise weight decay will make numbers not correct, 否则权重衰减会使数值不正确
            if not (args.pruning and args.pruning_method == 50):
                if batch_idx % args.log_interval == 0:
                    self.group_wd_optimizer.step_after()

            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_interval == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, batch_idx, len(train_loader), batch_time=batch_time,
                    loss=losses, top1=top1, top5=top5))
            sum_train_acc +=top1.avg
            sum_train_loss+=losses.avg
            if args.pruning:
                # pruning_engine.update_flops(stats=group_wd_optimizer.per_layer_per_neuron_stats)
                if batch_idx == 0:
                    sum_importances = self.pruning_engine.do_step(loss=loss.item(), optimizer=optimizer)
                else:
                    new_importances = self.pruning_engine.do_step(loss=loss.item(), optimizer=optimizer)
                    for i in range(len(sum_importances)):
                        sum_importances[i] += new_importances[i]
                if args.model == "resnet20" or args.model == "resnet101" or args.dataset == "Imagenet":
                    if (
                            self.pruning_engine.maximum_pruning_iterations == self.pruning_engine.pruning_iterations_done) and self.pruning_engine.set_moment_zero:
                        for group in optimizer.param_groups:
                            for p in group['params']:
                                if p.grad is None:
                                    continue
                                param_state = optimizer.state[p]
                                if 'momentum_buffer' in param_state:
                                    del param_state['momentum_buffer']

                        self.pruning_engine.set_moment_zero = False

            # if not (args.pruning and args.pruning_method == 50):
            #     if batch_idx % args.log_interval == 0:
            #         group_wd_optimizer.step_after()
        for i in range(len(sum_importances)):
            sum_importances[i] = sum_importances[i] / len(train_loader)
        return sum_importances,(sum_train_acc/len(train_loader)),(sum_train_loss/len(train_loader))

    def pruning_model(self,contributions):
        self.pruning_engine.do_prune_step(optimizer= self.optimizer,contributions=contributions)

    def neuron_stats(self,args):
        neurons_left = int(self.group_wd_optimizer.get_number_neurons(print_output=args.get_flops))
        flops = int(self.group_wd_optimizer.get_number_flops(print_output=args.get_flops))

        return neurons_left, flops

    def validate(self,args, test_loader, model, device, criterion, epoch, train_writer=None):
        """Perform validation on the validation set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        with torch.no_grad():
            for data_test in test_loader:
                data, target = data_test

                data = data.to(device)

                output = model(data)

                if args.get_inference_time:
                    iterations_get_inference_time = 100
                    start_get_inference_time = time.time()
                    for it in range(iterations_get_inference_time):
                        output = model(data)
                    end_get_inference_time = time.time()
                    print("time taken for %d iterations, per-iteration is: " % (iterations_get_inference_time),
                          (end_get_inference_time - start_get_inference_time) * 1000.0 / float(
                              iterations_get_inference_time), "ms")

                target = target.to(device)
                logits = output.logits
                loss = criterion(logits, target)

                prec1, prec5 = accuracy(logits.data, target, topk=(1, 5))
                losses.update(loss.item(), data.size(0))
                top1.update(prec1.item(), data.size(0))
                top5.update(prec5.item(), data.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

        print(
            ' * Prec@1 {top1.avg:.3f}, Prec@5 {top5.avg:.3f}, Time {batch_time.sum:.5f}, Loss: {losses.avg:.3f}'.format(
                top1=top1, top5=top5, batch_time=batch_time, losses=losses))
        train_writer.add_scalar(f'Client {self.id}/val_loss', losses.avg, epoch)
        train_writer.add_scalar(f'Client {self.id}/val_acc', top1.avg, epoch)
        return top1.avg, losses.avg

        ###=======================end for pruning
    def local_training(self,epoch):
        if epoch ==0:
            # loading model file
            if (len(self.args.load_model) > 0) and (not self.args.dynamic_network):
                if os.path.isfile(self.args.load_model):
                    load_model_pytorch(self.local_model, self.args.load_model, self.args.model)
                else:
                    print("=> no checkpoint found at '{}'".format(self.args.load_model))
                    exit()

        for l_epoch in range(1, self.args.local_epochs + 1):
            if self.args.distributed:
                self.train_sampler.set_epoch(l_epoch)
            adjust_learning_rate(self.args, self.optimizer, l_epoch, self.args.zero_lr_for_epochs, self.train_writer)

            if not self.args.run_test and not self.args.get_inference_time:
                cur_importances,train_acc,train_loss = self.train(self.args, self.local_model, self.device, self.train_loader, self.optimizer, l_epoch, self.criterion, train_writer=self.train_writer)
                if l_epoch==1:
                    avg_importances = cur_importances
                else:
                    for i in range(len(cur_importances)):
                        avg_importances[i] = avg_importances[i] + cur_importances[i]

            if self.args.pruning:
                # skip validation error calculation and model saving
                if self.pruning_engine.method == 50: continue
        for i in range(len(avg_importances)):
            avg_importances[i] = avg_importances[i] /self.args.local_epochs

        if self.train_writer is not None:
            self.train_writer.add_scalar(f'Client {self.id}/train_loss', train_loss, epoch)
            self.train_writer.add_scalar(f'Client {self.id}/train_acc', train_acc, epoch)

        return copy.deepcopy(avg_importances),train_acc,train_loss

    def local_training_alone(self,epoch):
        if epoch ==0:
            # loading model file
            if (len(self.args.load_model) > 0) and (not self.args.dynamic_network):
                if os.path.isfile(self.args.load_model):
                    load_model_pytorch(self.local_model, self.args.load_model, self.args.model)
                else:
                    print("=> no checkpoint found at '{}'".format(self.args.load_model))
                    exit()

        for l_epoch in range(1, self.args.local_epochs + 1):
            if self.args.distributed:
                self.train_sampler.set_epoch(l_epoch)
            adjust_learning_rate(self.args, self.optimizer, l_epoch, self.args.zero_lr_for_epochs, self.train_writer)

            if not self.args.run_test and not self.args.get_inference_time:
                cur_importances,train_acc,train_loss = self.train(self.args, self.local_model, self.device, self.train_loader, self.optimizer, l_epoch, self.criterion, train_writer=self.train_writer)

            if self.train_writer is not None:
                self.train_writer.add_scalar(f'Client {self.id}/train_loss', train_loss, l_epoch-1)
                self.train_writer.add_scalar(f'Client {self.id}/train_acc', train_acc, l_epoch-1)

            prec1, loss = self.validate(self.args, self.test_loader, self.local_model, self.device, self.criterion,
                                        l_epoch - 1, train_writer=self.train_writer)

        return loss,prec1,train_acc,train_loss
