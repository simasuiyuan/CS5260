# Colossalai and LR Range Test Report
## training framework: Colossalai
## source code: 

## 1. Exponentially increase the Learning rate (base)
### Learning rate
    def lrs(batch):
        low = math.log2(1e-5)
        high = math.log2(10)
        return 2**(low+(high-low)*batch/len(train_dataloader)/gpc.config.NUM_EPOCHS)
<img src=".\images\ExponentiallyIncLR.PNG" alt="ExponentiallyIncLR" width="400"/>

### train & valid loss
<img src=".\images\ExponentiallyIncLoss.PNG" alt="ExponentiallyIncLoss" width="550"/>

## 2. Decays the learning rate of each parameter group by gamma=0.5 every 100 epochs.
### Learning rate (vs Exponentially decrease Lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
<img src=".\images\2_legend.PNG" alt="2_legend" width="150" align="left"/><img src=".\images\StepDecLR.PNG" alt="ExponentiallyDecLR" width="400"/>  

### train & valid loss (vs Exponentially decrease Lr)
<img src=".\images\StepDecLoss.PNG" alt="ExponentiallyDecLoss" width="550"/>

## 3. Decays the learning rate of each parameter group by gamma every epoch.
### Learning rate (comparing different gamma - decaying rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
<img src=".\images\3_legend.PNG" alt="3_legend" width="150" align="left"/><img src=".\images\ExponentiallyDecLR.PNG" alt="ExponentiallyDecLR" width="400"/>  

### train & valid loss (comparing different gamma - decaying rate)
<img src=".\images\ExponentiallyDecLoss.PNG" alt="ExponentiallyDecLoss" width="550"/>

## 4. Sets the learning rate of each parameter group according to the 1cycle learning rate policy.
### Learning rate 
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1.1, steps_per_epoch=len(train_dataloader), epochs=config["NUM_EPOCHS"])

* Compare to  Exponentially decrease Lr

<img src=".\images\4_legend.PNG" alt="4_legend" width="150" align="left"/><img src=".\images\OneCycleLR.PNG" alt="OneCycleLR" width="400"/>  

### train & valid loss (comparing different gamma - decaying rate)
<img src=".\images\OneCycleLoss.PNG" alt="OneCycleLoss" width="550"/>

* Compare different max_lr (1 vs 1.1)

<img src=".\images\5_legend.PNG" alt="5_legend" width="150" align="left"/><img src=".\images\OneCycleLR_DiffMax.PNG" alt="OneCycleLR_DiffMax" width="400"/>  

### train & valid loss (comparing different gamma - decaying rate)
<img src=".\images\OneCycleLoss_DiffMax.PNG" alt="OneCycleLoss_DiffMax" width="550"/>

## 5. Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR).
### Learning rate 
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=1, step_size_up=50, step_size_down=100)

* Compare to  Exponentially decrease Lr

<img src=".\images\6_legend.PNG" alt="6_legend" width="150" align="left"/><img src=".\images\CyclicLR.PNG" alt="CyclicLR" width="400"/>  

### train & valid loss (comparing different gamma - decaying rate)
<img src=".\images\CyclicLoss.PNG" alt="CyclicLoss" width="550"/>

* Compare different learning schedules

<img src=".\images\7_legend.PNG" alt="7_legend" width="150" align="left"/><img src=".\images\compare3bestLR.PNG" alt="compare3bestLR" width="400"/>  

### train & valid loss (comparing different gamma - decaying rate)
<img src=".\images\7_legend.PNG" alt="7_legend" width="150" align="left"/><img src=".\images\compare3bestLoss.PNG" alt="compare3bestLoss" width="550"/>

<img src=".\images\7_legend.PNG" alt="7_legend" width="150" align="left"/><img src=".\images\compare3bestTestLoss.PNG" alt="compare3bestTestLoss" width="550"/>

# github link: 
