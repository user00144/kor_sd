import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import CLIPProcessor

# optimizer, scheduler, dataloader , loss_fn

device = "cuda"

def train(dl_train, dl_val, student_model , teacher_model , CONFIG) :
    loss_fn = nn.MSELoss()
    
    params = list(student_model.text_model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
            "weight_decay": CONFIG.WD,
        },
        {
            "params": [p for n, p in params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=CONFIG.LR)
    
    
    warmup_ratio = 0.1
    t_total = len(dl_train) * CONFIG.EPOCHS
    warmup_step = int(t_total * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = warmup_step, num_training_steps = t_total)
    
    #scaler = torch.cuda.amp.GradScaler()
    
    teacher_model.to(device)
    student_model.text_model.to(device)
    
    best_val_loss = 5
    
    for epoch in range(CONFIG.EPOCHS) :
        epoch_train_loss, epoch_ko_en_loss, epoch_en_en_loss = train_fn(dl_train, teacher_model, student_model, optimizer, loss_fn, scheduler)
        epoch_val_loss = eval_fn(dl_val, teacher_model, student_model, loss_fn)
        print(f'EPOCH {epoch} - Train loss : {epoch_train_loss} / ko_en : {epoch_ko_en_loss} / en_en : {epoch_en_en_loss} \n Val loss : {epoch_val_loss}')
        
        if best_val_loss > epoch_val_loss :
            processor = CLIPProcessor.from_pretrained(CONFIG.TOKENIZER_MODEL_NAME)
            student_model.save_pretrained(CONFIG.SAVE_DIR)
            processor.save_pretrained(CONFIG.SAVE_DIR)

    


def train_fn(dl_train, teacher_model, student_model , optimizer , loss_fn, scheduler) :
    teacher_model.eval()
    student_model.train()
    
    epoch_train_loss = 0
    epoch_ko_en_loss = 0
    epoch_en_en_loss = 0

    for i, batch in enumerate(tqdm(dl_train)) :
        
        optimizer.zero_grad()
        
        # batch
        
        #with torch.cuda.amp.autocast():

        ko_token, en_ko_token, en_en_token = batch
        ko_token, en_ko_token, en_en_token = ko_token.to(device), en_ko_token.to(device), en_en_token.to(device)
            
        en_ko_out = student_model.text_model(**en_ko_token, output_hidden_states = True)
        ko_out = student_model.text_model(**ko_token, output_hidden_states = True)
        en_en_out = teacher_model.text_model(**en_en_token, output_hidden_states = True)

        # pooled_prompt_embeds = prompt_embeds[0]
        # prompt_embeds = prompt_embeds.hidden_states[-2]
            
        en_en_emb = en_en_out.hidden_states[-2]
        en_ko_emb = en_ko_out.hidden_states[-2]
        ko_emb = ko_out.hidden_states[-2]
            
        pooled_en_en_emb = en_en_out[0]
        pooled_en_ko_emb = en_ko_out[0]
        pooled_ko_emb = ko_out[0]
                
        ko_en_loss = loss_fn(ko_emb, en_en_emb)
        en_en_loss = loss_fn(en_ko_emb, en_en_emb)
            
        pooled_ko_en_loss = loss_fn(pooled_ko_emb, pooled_en_en_emb)
        pooled_en_en_loss = loss_fn(pooled_en_ko_emb, pooled_en_en_emb)
        
        loss = (ko_en_loss + pooled_ko_en_loss) * 0.5 + (en_en_loss + pooled_en_en_loss) * 0.5
        
        epoch_val_loss += loss
        
        epoch_ko_en_loss += ko_en_loss
        epoch_en_en_loss += en_en_loss
        
        epoch_train_loss += loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        
    epoch_train_loss /= len(dl_train)
    epoch_ko_en_loss /= len(dl_train)
    epoch_en_en_loss /= len(dl_train)
    
    return epoch_train_loss, epoch_ko_en_loss, epoch_en_en_loss

def eval_fn(dl_val, teacher_model, student_model, loss_fn) :
    teacher_model.eval()
    student_model.eval()
    
    epoch_val_loss = 0
    with torch.no_grad() :
        for i, batch in enumerate(tqdm(dl_val)) :
            # batch
            ko_token, en_ko_token, en_en_token = batch
            ko_token, en_ko_token, en_en_token = ko_token.to(device), en_ko_token.to(device), en_en_token.to(device)
            
            en_en_out = teacher_model.text_model(**en_en_token, output_hidden_states = True)
            en_ko_out = student_model.text_model(**en_ko_token, output_hidden_states = True)
            ko_out = student_model.text_model(**ko_token, output_hidden_states = True)
            
            # pooled_prompt_embeds = prompt_embeds[0]
            # prompt_embeds = prompt_embeds.hidden_states[-2]
            
            en_en_emb = en_en_out.hidden_states[-2]
            en_ko_emb = en_ko_out.hidden_states[-2]
            ko_emb = ko_out.hidden_states[-2]
            
            pooled_en_en_emb = en_en_out[0]
            pooled_en_ko_emb = en_ko_out[0]
            pooled_ko_emb = ko_out[0]
                
            ko_en_loss = loss_fn(ko_emb, en_en_emb)
            en_en_loss = loss_fn(en_ko_emb, en_en_emb)
            
            pooled_ko_en_loss = loss_fn(pooled_ko_emb, pooled_en_en_emb)
            pooled_en_en_loss = loss_fn(pooled_en_ko_emb, pooled_en_en_emb)
            
            loss = (ko_en_loss + pooled_ko_en_loss) * 0.5 + (en_en_loss + pooled_en_en_loss) * 0.5
            
            epoch_val_loss += loss
        
        epoch_val_loss /= len(dl_val)
    
    return epoch_val_loss

        
        

        