import os
import torch
import numpy as np
from time import time

import utils as u

#########################################################################################################

def fit(model=None, optimizer=None, scheduler=None, epochs=None,
        dataloaders=None, early_stopping_patience=None, verbose=False):
    
    def get_accuracy(y_pred, y_true):
        y_pred, y_true = torch.argmax(y_pred, dim=1).detach(), y_true.detach()
        return torch.count_nonzero(y_true == y_pred).item() / len(y_pred)
    
    u.breaker()
    u.myprint("Training ...", "cyan")
    u.breaker()

    Losses, Accuracies = [], []
    bestLoss = {"train" : np.inf, "valid" : np.inf}
    bestAccs = {"train" : 0.0, "valid" : 0.0}

    model.to(u.DEVICE)
    start_time = time()
    for e in range(epochs):
        e_st = time()
        epochLoss = {"train" : np.inf, "valid" : np.inf}
        epochAccs = {"train" : 0.0, "valid" : 0.0}

        for phase in ["train", "valid"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            
            lossPerPass, accsPerPass = [], []

            for X, y in dataloaders[phase]:
                X, y = X.to(u.DEVICE), y.to(u.DEVICE).view(-1)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    output = model(X)
                    loss = torch.nn.NLLLoss()(output, y)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                lossPerPass.append(loss.item())
                accsPerPass.append(get_accuracy(output, y))
            epochLoss[phase] = np.mean(np.array(lossPerPass))
            epochAccs[phase] = np.mean(np.array(accsPerPass))
        Losses.append(epochLoss)
        Accuracies.append(epochAccs)

        if early_stopping_patience:
            if epochLoss["valid"] < bestLoss["valid"]:
                bestLoss = epochLoss
                BLE = e + 1
                torch.save({"model_state_dict": model.state_dict(),
                            "optim_state_dict": optimizer.state_dict()},
                           os.path.join(u.CHECKPOINT_PATH, "state.pt"))
                early_stopping_step = 0
            else:
                early_stopping_step += 1
                if early_stopping_step > early_stopping_patience:
                    u.myprint("\nEarly Stopping at Epoch {}".format(e + 1), "green")
                    break
        
        if epochLoss["valid"] < bestLoss["valid"]:
            bestLoss = epochLoss
            BLE = e + 1
            torch.save({"model_state_dict" : model.state_dict(),
                        "optim_state_dict" : optimizer.state_dict()},
                        os.path.join(u.CHECKPOINT_PATH, "state.pt"))
        
        if epochAccs["valid"] > bestAccs["valid"]:
            bestAccs = epochAccs
            BAE = e + 1
        
        if scheduler:
            scheduler.step(epochLoss["valid"])
        
        if verbose:
            u.myprint("Epoch: {} | Train Loss: {:.5f} | Valid Loss: {:.5f} |\
Train Accs: {:.5f} | Valid Accs: {:.5f} | Time: {:.2f} seconds".format(e+1, 
                                                                       epochLoss["train"], epochLoss["valid"], 
                                                                       epochAccs["train"], epochAccs["valid"], 
                                                                       time()-e_st), "cyan")
    
    u.breaker()
    u.myprint("Best Validation Loss at Epoch {}".format(BLE), "cyan")
    u.breaker()
    u.myprint("Best Validation Accs at Epoch {}".format(BAE), "cyan")
    u.breaker()
    u.myprint("Time Taken [{} Epochs] : {:.2f} minutes".format(len(Losses), (time()-start_time)/60), "cyan")
    u.breaker()
    u.myprint("Training Completed", "cyan")
    u.breaker()

    return Losses, Accuracies, BLE, BAE

#########################################################################################################

def predict(model, dataloader):
    model.load_state_dict(torch.load(os.path.join(u.CHECKPOINT_PATH, "state.pt"))["model_state_dict"])
    model.to(u.DEVICE)
    model.eval()

    y_pred = torch.zeros(1, 1).to(u.DEVICE)
    for X in dataloader:
        X = X.to(u.DEVICE)
        with torch.no_grad():
            output = torch.argmax(model(X), dim=1).view(-1, 1)
        y_pred = torch.cat((y_pred, output), dim=0)
    
    return y_pred[1:].detach().cpu().numpy()

#########################################################################################################
