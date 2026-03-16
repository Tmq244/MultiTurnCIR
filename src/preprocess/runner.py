import torch
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
from irbench.irbench import IRBench
from irbench.evals.eval_helper import EvalHelper
import os
import random
class Trainer(object):
    def __init__(self,
                 args,
                 data_loader,
                 model,
                 summary_writer):
        self.args = args
        self.data_loader = data_loader
        self.model = model
        self.summary_writer = summary_writer
        self.processed_images = 0
        self.global_step = 0

    def __adjust_lr__(self,epoch,warmup=True):
        lr = self.args.lr * self.args.batch_size/16.0
        if warmup:
            warmup_images = 10000
            lr = min(self.processed_images * lr / float(warmup_images),lr)
        for e in self.args.lr_decay_steps:
            if epoch>=e:
                lr*=self.args.lr_decay_factor
        self.model.adjust_lr(lr)
        self.cur_lr = lr

    def __logging__(self,log_data):
        msg = '[Train][{}]'.format(self.args.expr_name)
        msg += '[Epoch: {}]'.format(self.epoch)
        msg += '[Lr:{:.6f}]'.format(self.cur_lr)
        log_data['lr'] = self.cur_lr
        for k, v in log_data.items():
            if not self.summary_writer is None:
                self.summary_writer.add_scalar(k, v, self.global_step)
            if isinstance(v, float):
                msg += '{}:{:.6f}'.format(k, v)
            else:
                msg += '{}:{}'.format(k, v)
        print(msg)

    def train(self, epoch):
        self.epoch = epoch
        self.model.train()
        metric_sums = {}
        metric_count = 0
        for bidx, input in enumerate(tqdm(self.data_loader, desc='Train')):
            self.global_step += 1
            self.processed_images += input[0][0].size(0)
            self.__adjust_lr__(epoch, warmup=self.args.warmup)
            for i in range(self.args.max_turn_len+2):
                input[i][0] = Variable(input[i][0]).cuda()
                input[i][1] = Variable(input[i][1]).cuda()
            #input[0][0] = Variable(input[0][0])
            #input[0][1] = Variable(input[0][1])
            #input[1][0] = Variable(input[1][0])
            # input[1][1] = Variable(input[1][1])
            # input[2][0] = Variable(input[2][0])
            #input[self.args.max_turn_len+1] = Variable(input[self.args.max_turn_len+1]).cuda()
            output = self.model(input)
            log_data = self.model.update(output)
            metric_count += 1
            for key, value in log_data.items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
            if (bidx % self.args.print_freq) == 0:
                self.__logging__(log_data)
        epoch_summary = {'num_batches': metric_count, 'lr': float(self.cur_lr)}
        if metric_count > 0:
            for key, value in metric_sums.items():
                epoch_summary[key] = value / float(metric_count)
        return epoch_summary
        #torch.save(self.model,'Attr/'+str(epoch)+str(self.args.target)+'.pkl')    
class Evaluator(object):
    def __init__(self,args,data_loader,model,summary_writer,eval_freq):
        self.args = args
        self.data_loader = data_loader
        self.model = model
        self.summary_writer = summary_writer
        self.eval_freq = eval_freq
        self.best_score = -1.0
        self.repo_path = os.path.join('./repo',args.expr_name)
        if not os.path.exists(self.repo_path):
            os.makedirs(self.repo_path)
        self.best_ckpt_path = os.path.join(self.repo_path, 'best_model.pth')
        self.last_ckpt_path = None
        self.targets = list(self.data_loader.keys())

    def _save_best_checkpoint(self, epoch, metric_value, overall_result):
        state_dict = self.model.state_dict()
        ckpt = {
            'epoch': epoch,
            'metric_name': 'R10R50',
            'metric_value': float(metric_value),
            'overall': overall_result,
            'state_dict': state_dict,
        }
        torch.save(ckpt, self.best_ckpt_path)
        epoch_ckpt_name = 'best_model_epoch_{:03d}_r10r50_{:.4f}.pth'.format(epoch, metric_value)
        self.last_ckpt_path = os.path.join(self.repo_path, epoch_ckpt_name)
        torch.save(ckpt, self.last_ckpt_path)

    def test(self, epoch):
        ir_config = {}
        ir_config['srch_method'] = 'bf'
        ir_config['srch_libs'] = None
        ir_config['desc_type'] = 'global'
        irbench = IRBench(ir_config)
        irdict = {}
        self.epoch = epoch
        model = self.model.eval()
        r1 = 0.
        r5 = 0.
        r8 = 0.
        r10 = 0.
        r20 = 0.
        r50 = 0.
        r10r50 = 0.
        mrr = 0.
        target_results = {}
        for target, data_loader in self.data_loader.items():
            query_targets = {}

            irbench.clean()
            eval_helper = EvalHelper()

            # add index features.
            data_loader.dataset.set_mode('index')
            for bidx, input in enumerate(tqdm(data_loader, desc='Index')):
                if self.args.method == 'combine':
                    data = input[2]
                    data1 = Variable(input[0].cuda())
                else:
                    input[0] = Variable(input[0].cuda())  # input[0] = (x, image_id)
                #input[2] = Variable(input[2].cuda())
                    data = input[0]
                image_id = input[1]
                with torch.no_grad():
                    if self.args.method == 'combine':
                        output = model.get_original_combined_feature(data,data1)
                    else:
                        output = model.get_original_image_feature(data)
                for i in range(output.size(0)):
                    _iid = image_id[i]
                    _feat = output[i].squeeze().cpu().numpy()
                    irbench.feed_index([_iid, _feat])
                    irdict[_iid] = _feat
            data_loader.dataset.set_mode('query')
            for bidx, input in enumerate(tqdm(data_loader, desc='Query')):
                for i in range(self.args.max_turn_len+1):
                    input[i][0] = Variable(input[i][0]).cuda()
                    input[i][1] = Variable(input[i][1]).cuda()
                input[self.args.max_turn_len+1][1] = Variable(input[self.args.max_turn_len+1][1]).cuda()
                with torch.no_grad():
                    output = self.model(input)[0]
                    #output = model.get_original_tag_feature()
                    # output = model.get_manipulated_image_feature(data)
                for i in range(output.size(0)):
                    _qid = input[self.args.max_turn_len+1][0][i]
                    _feat = output[i].squeeze().cpu().numpy()
                    irbench.feed_query([_qid, _feat])

                    _w_key = input[self.args.max_turn_len+1][0][i]
                    _tid = input[self.args.max_turn_len][2][i]
                    query_targets[_w_key] = _tid
                    eval_helper.feed_gt([_w_key, [_tid]])
            res = irbench.search_all(top_k=None)
            res = irbench.render_result(res)
            eval_helper.feed_rank_from_dict(res)
            score = eval_helper.evaluate(metric=['top_k_acc'], kappa=[1, 5, 8, 10, 20, 50])
            print('Target: {}'.format(target))
            print(score)
            _r1 = score[0][str(1)]['top_k_acc']
            _r5 = score[0][str(5)]['top_k_acc']
            _r8 = score[0][str(8)]['top_k_acc']
            _r10 = score[0][str(10)]['top_k_acc']
            _r20 = score[0][str(20)]['top_k_acc']
            _r50 = score[0][str(50)]['top_k_acc']
            _r10r50 = 0.5 * (_r10 + _r50)
            _mrr = 0.
            for query_id, target_id in query_targets.items():
                ranked_ids = res.get(query_id, [])
                if target_id in ranked_ids:
                    _mrr += 1.0 / (ranked_ids.index(target_id) + 1)
            if query_targets:
                _mrr /= float(len(query_targets))
            target_results[target] = {
                'R1': _r1,
                'R5': _r5,
                'R8': _r8,
                'R10': _r10,
                'R20': _r20,
                'R50': _r50,
                'R10R50': _r10r50,
                'MRR': _mrr,
            }
            r1 += _r1
            r8 += _r8
            r10 += _r10
            r20 += _r20
            r50 += _r50
            r10r50 += _r10r50
            r5 += _r5
            mrr += _mrr
            if (bidx % self.args.print_freq) == 0 and self.summary_writer is not None:
                self.summary_writer.add_scalar('{}/R1'.format(target), _r1, epoch)
                self.summary_writer.add_scalar('{}/R5'.format(target), _r5, epoch)
                self.summary_writer.add_scalar('{}/R8'.format(target), _r8, epoch)
                self.summary_writer.add_scalar('{}/R10'.format(target), _r10, epoch)
                self.summary_writer.add_scalar('{}/R20'.format(target), _r20, epoch)
                self.summary_writer.add_scalar('{}/R50'.format(target), _r50, epoch)
                self.summary_writer.add_scalar('{}/R10R50'.format(target), _r10r50, epoch)
                self.summary_writer.add_scalar('{}/MRR'.format(target), _mrr, epoch)

            # mean score.
        r1 /= len(self.data_loader)
        r10r50 /= len(self.data_loader)
        r8 /= len(self.data_loader)
        r10 /= len(self.data_loader)
        r20 /= len(self.data_loader)
        r50 /= len(self.data_loader)
        r5 /= len(self.data_loader)
        mrr /= len(self.data_loader)
        overall_result = {
            'R1': r1,
            'R5': r5,
            'R8': r8,
            'R10': r10,
            'R20': r20,
            'R50': r50,
            'R10R50': r10r50,
            'MRR': mrr,
        }
        is_best = r10r50 > self.best_score
        if is_best:
            self.best_score = r10r50
            self._save_best_checkpoint(epoch, r10r50, overall_result)
            print('[Checkpoint] New best R10R50={:.4f} at epoch {}, saved to {}'.format(r10r50, epoch, self.best_ckpt_path))
        print('Overall>> R1:{:.4f}\tR5:{:.4f}\tR8:{:.4f}\tR10:{:.4f}\tR20:{:.4f}\tR50:{:.4f}\tR10R50:{:.4f}\tMRR:{:.4f}'.format(r1, r5, r8, r10, r20, r50, r10r50, mrr))
        return {
            'targets': target_results,
            'overall': overall_result,
            'best': {
                'is_best': is_best,
                'best_metric_name': 'R10R50',
                'best_metric_value': self.best_score,
                'best_ckpt_path': self.best_ckpt_path,
                'last_best_epoch_ckpt_path': self.last_ckpt_path,
            },
        }
        #max_r10 = 0
        #max_r50 = 0
        #if (r10r50>max_r10r50):
         #   print 'y'
          #  max_r10r50 = r10r50
           # torch.save(self.model,'Text_Only/'+str(epoch)+'model.pkl')
        
                                   
