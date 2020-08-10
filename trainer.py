from __future__ import absolute_import, division, print_function, unicode_literals
import glob, os
import torch
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.optim import Adam
from tqdm import tqdm, trange, tqdm_notebook, tnrange


class Trainer:
    def __init__(self,
                 args,
                 config,
                 model, 
                 criterion, 
                 train_dataloader, 
                 valid_dataloader,
                 logger,
                 save_path,
                 tb_writer):
        
        self.args = args
        self.config = config
        self.model = model
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.logger = logger
        self.save_path = save_path
        self.tb_writer = tb_writer
        
        self.t_total = len(self.train_dataloader)*self.args.epoch
        self.device = self.config.device
        # self.optimizer = AdamW(self.get_model_parameters(), lr=self.config.learning_rate)
        self.optimizer = Adam(self.get_model_parameters(), lr=self.config.learning_rate)
        self.scheduler = WarmupLinearSchedule(self.optimizer, 0.1*self.t_total, self.t_total)
           
        self.global_step = 0
        self.best_eval_loss = 7.0

    def get_model_parameters(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}]
        
        return optimizer_grouped_parameters
        
        
    def train(self):
        for epoch in range(self.args.epoch):
            self.train_epoch(epoch)
            self.evaluation(epoch)
            self.write_to_tb()
            self.save_model(epoch)
        self.tb_writer.close()

        
    def transform_to_bert_input(self,batch):

        input_ids = torch.from_numpy(batch[0]).to(self.device) 
        valid_length = batch[1].clone().detach().to(self.device)
        token_type_ids = torch.tensor( batch[2]).long().to(self.device)
        label = torch.from_numpy(batch[3]).to(self.device)

        return input_ids, valid_length, token_type_ids, label
        
        
    def train_epoch(self, epoch):       
        self.model.to(self.device)
        self.model.train()  
    
        tr_correct_cnt, tr_total_cnt = 0,0
        tr_loss = 0.0      
        train_loader = self.train_dataloader

        for step, batch in enumerate(train_loader):     
            self.model.zero_grad()   
              
            input_idx, valid_length, token_type_idx, label = self.transform_to_bert_input(batch)
            output  = self.model(input_idx, valid_length, token_type_idx)
            loss = self.criterion(output.view(-1, output.size(-1)), label.view(-1))

            tr_loss += loss.item()
            loss.backward()            
            
            if step>0 and (step) % self.config.gradient_accumulation_steps == 0:
                self.global_step += self.config.gradient_accumulation_steps

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.tr_avg_loss = tr_loss / step

                if self.global_step % 100==0:
                    self.logger.info('epoch : {} /{}, global_step : {} /{}, loss: {:.3f}, tr_avg_loss: {:.3f}'.format(
                        epoch+1, self.args.epoch, self.global_step, self.t_total, loss.item(), self.tr_avg_loss))
                    

    def evaluation(self, epoch):  
        self.model.eval()
        eval_loss = 0.0
        eval_step=1
        
        self.logger.info('*****************Evaluation*****************')
        valid_loader = tqdm(self.valid_dataloader)
        for step, batch in enumerate(valid_loader):    
            with torch.no_grad():   
                input_idx, valid_length, token_type_idx, label = transform_to_bert_input(batch)
                output  = model(input_idx, valid_length, token_type_idx)
                
            loss = self.criterion(output.view(-1, output.size(-1)), label.view(-1))                     
            eval_loss += loss.item()
            eval_step += 1.0

        self.eval_avg_loss = eval_loss/eval_step
        self.logger.info('epoch : {} /{}, global_step : {} /{}, eval_loss: {:.3f}'.format(
            epoch+1, self.args.epoch, self.global_step, self.t_total, self.eval_avg_loss))                
                
    def save_model(self, epoch):
        if self.eval_avg_loss < self.best_eval_loss:
            self.best_eval_loss = self.eval_avg_loss

            self.model.to(torch.device('cpu'))
            state = {'epoch': epoch+1,
                     'model_state_dict': self.model.state_dict()}

            save_model_path = '{}/epoch_{}_step_tr_loss_{:.3f}_eval_loss_{:.3f}.pt'.format(
                        self.save_path, epoch+1, self.global_step, self.tr_avg_loss, self.eval_avg_loss)
                
                
            # Delte previous checkpoint
            if len(glob.glob(self.save_path+'/epoch*.pt'))>0:
                os.remove(glob.glob(self.save_path+'/epoch*.pt')[0])
            torch.save(state, save_model_path)
            self.logger.info(' Model saved to {}'.format(save_model_path))
            
            os.mkdir(self.save_path+'/epoch_{}_eval_loss_{:.3f}'.format(epoch+1, self.eval_avg_loss))  


    def write_to_tb(self):
        self.tb_writer.add_scalars('loss', {'train': self.tr_avg_loss, 'val': self.eval_avg_loss}, self.global_step)