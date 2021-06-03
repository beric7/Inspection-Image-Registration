import csv
import copy
import time
from tqdm import tqdm
import torch
import numpy as np
import os

def train_model(model, criterion, dataloaders, optimizer, metrics, bpath, num_epochs=3):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    best_f1_score = 0
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Test_loss'] + \
        [f'Train_{m}' for m in metrics.keys()] + \
        [f'Test_{m}' for m in metrics.keys()]
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}
        torch.cuda.empty_cache()
        for phase in ['Train', 'Test']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            # Iterate over data.
            for batch in tqdm(iter(dataloaders[phase])):
                # These lines appear to be correct.
                images = batch['image'].to(device)
                true_masks = batch['mask'].to(device, dtype=torch.long)

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    # currently giving a tensor of [batch,dimension,dimension,channel]
                    # expects [batch, channel, dim, dim]
                    images = images.permute(0, 3, 1, 2).contiguous()
                    out, lowLevel = model(images)
                    y_pred_tensor_out = out
                    y_pred_tensor_lowLevel = lowLevel
                    
                    pred_out = torch.argmax(y_pred_tensor_out, dim=1)
                    y_pred_out = pred_out.data.cpu().numpy()

                    
                    # This is set up perfectly for the use of determining accuracy
                    y_pred_out = y_pred_out.ravel()
                    y_true = true_masks.data.cpu().numpy().ravel()
                    
                    # outputs['out']
                    loss_1 = criterion(y_pred_tensor_out, true_masks)
                    loss_2 = criterion(y_pred_tensor_lowLevel, true_masks)
                    loss = loss_1*0.5 + loss_2*0.5
                    
                    for name, metric in metrics.items():
                        if name == 'f1_score':
                            batchsummary[f'{phase}_{name}'].append(metric(y_true, y_pred_out, average='weighted'))
                        if name == 'jaccard_score':
                            batchsummary[f'{phase}_{name}'].append(metric(y_true, y_pred_out, average='weighted'))
                        else:
                            batchsummary[f'{phase}_{name}'].append(metric(y_true.astype('uint8'), y_pred_out, average='weighted'))

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss
            batchsummary[f'{phase}_loss'] = epoch_loss.item()
            print('{} Loss: {:.4f}'.format(phase, loss))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        print(batchsummary)
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Test' and batchsummary['Test_f1_score'] > best_f1_score:
                best_f1_score = batchsummary['Test_f1_score']
                best_model_wts = copy.deepcopy(model.state_dict())
                model.load_state_dict(best_model_wts)
                torch.save(model, os.path.join(bpath, 'weights_'+ str(epoch) + '.pt'))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
