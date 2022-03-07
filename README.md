###### tags: `AI Capstone` `Pj1`
> [hackmd link](https://hackmd.io/@libao3128/B1_6u8jl5)
# Programming Assignment #1
Name: 黃立鈞
ID: 0816086

## Public Image Dataset
### Dataset
The dataset I choose for public image data is bird species classifier. The dataset is provided by [Colab Competition](https://competitions.codalab.org/competitions/35668?secret_key=09789b13-35ec-4928-ac0f-6c86631dda07#participate-get_starting_kit), which contains 6033 images belonging to 200 bird species. The task is to train a neural network by 3000 labeled image. You can go to my [github](https://github.com/libao3128/Bird_Classifier) to get more information about code and dataset. 
### Methodology
Before training, I seperate the labeled dataset into train, validate and test dataset with ratio 8:1:1 randomly. The function I used or split dataset is built in pytorch and the random seed is fixed to 0 to make sure the training process is reproducible and the test dataset will not be shuffled into others dataset when the training process is separate to many rounds.

As I mentioned above, the methodology I used for image preprocessing include:
1. **Image Padding**:
    - To make the image padding to the specified size of square and make the target object at the middle of the image.
2. **Image Resize**:
    - To resize the image with specified size.

3. **Image Crop**(random and center):
    - The random image crop is used when training to avoid the model overfitting.
    - The center image crop is used for testing because we assume most of the image is at the center of the image.
4. **Image Flip**(only for training):
    - Flip the image horizontally or vertically randomly to avoid overfitting.
5. **Pixel Normalization**:
    - Common methodology used for image preprocessing

Methodology for model:

1. Attention
    - Add an attention layer in the model to make the model focus on the features.

Methodology for model or training:
1. **MixUp**
    - This will overlay two image with the function :
        $x = \lambda x_i + (1-\lambda)x_j$
        where x is the mixup image, and $x_i$ and $x_j$ are two random select images. $\lambda$ is a random factor produce by beta function with $\alpha=\beta=0.2$.
    - The label for predicting and calculating loss function is also transformed by the function. 
    - For example:
        If there is an image belonging to class 1 and MixUp with an image with class 2. And $\lambda=0.2$. It means that every pixel of the new image is calculate by $x=0.2*x_1+0.8*x_2$ and the label of the new image will be $[0.2,0.8]$.
2. **Learning rate scheduler**
    - I use an exponential learning rate scheduler with decay rate 0.9.
### Algorithm
The algorithm I use for bird classifiers is CNN-based neural network. I use the pretrained resnet-152 model provided by pytorch library.
### The final hyper-parameter I used is:
#### For model:
| Model      | Loss Function |  Optimizer   | Attention |
| ---------- | ------------- | --- | --------- |
| resnet-152 | Cross Entropy |  Adam   | True    |

#### For date preprocessing:
##### Train dataset&Val dataset
    

| Resize | Crop     | Flip     | Normalization |
| -------------------- | -------- | -------- | ------------- |
| True                 | Randomly | Randomly | True          |



The **random crop** strategy I used:
- Since the size of the image for training is not fixed, I can not crop the fixed pixel size.
- The crop size I use is **0.8*the image** size I specified above.

The **random flip** strategy I used:
- Randomly flip the image vertically or horizontally.

The **normalization and standardize**e parameters I used:
- According to the parameter that people usually use for resnet-50 and resnet 152.
- Mean:$[0.485,0.456,0.406]$.
- Standard deviation:$[0.229,0.224,0.225]$.

##### Test dataset:


| Resize | Crop   | Flip  | Normalization |
| ------ | ------ | ----- | ------------- |
| False  | Center | False | True          |

As you can see, resizing is not used for test dataset, which means that the size of the image is fixed.
Therefor, it is possible for us to crop the test image into fixed size and I use center crop with size [375,375]. The reduction of random parameters allow us to reproduce the predicting result.
The parameter for normalization is the same as the train dataset:
- Mean:$[0.485,0.456,0.406]$.
- Standard deviation:$[0.229,0.224,0.225]$.

#### For training


| batch size | Progressively Resize | MixUp |
| ---------- | -------------------- | ----- |
| 5          | True                 | True  |

The **progressively resizing** strategy I used:
- The shorter edge of the image will fit the size I specified.
- Initial size 64 pixels.
- After 10 epochs, 128 pixels.
- After 20 epochs, 196 pixels.
- After 30 epochs, 296 pixels.
- After 40 epochs, 375 pixels.

The **MixUp** strategy I used:
I use a mixup strategy for training in order to make the model see the combination of the images rather than to focus on the specific one.
However, it is not appropriate to mix up every training image since the model will be confused and distracted.Therefore, I mix up the training images for approximately every 5 images whenever image_idx%5==0.
### Analysis
After training, my model get **0.66601** accuracy on the test dataset for the competition.
![](https://i.imgur.com/Dwnqf5p.png)
### Discuss
In this experiment, I have tried many image preprocessing techniques and found out that the performance of the model is highly related to the dataset quality. Besides, since the training dataset is not very large, for the high complexity model like CNN network, it is very likely for it to overfit the dataset. Therefore, I also tried many technique to prevent overfitting like **MixUp**, **random crop** and **random flip**.
Although the performance of my model is not very brilliant, I did learn a lot through the experiment. If I got more time, I would like to try different CNN model and see if I can get better result or not.

## Public Non-Image Dataset

### Dataset
I use the stock price of AAPL from 2020-1-1 to 2021-12-31 as public non-image dataset. The dataset contain 5536 rows and 6 columns. The columns' information is shown below:
![](https://i.imgur.com/jrFpo0p.png =500x)

The chart of the close prices is shown below. 
![](https://i.imgur.com/8wSsARa.png =300x)


Our goal is to predict whether the next day's price will be higher than today's price. Therefore, the label of the dataset will be $Close_{t+1}-Close_{t}>0$, where $t$ is the date. After computing the trend of prices, the label distribution looks like:
| True | False |
| ---- | ----- |
| 2868 | 2634  |

,which means that along 5536 days, there are 2868 days' prices growing higher than yesterday and 2634 descending on the other hand.

### Methodology
As mentioned above, our dataset is the record of the stock price with the time series. It is very hard for us to predict tomorrow's trend directly by today's price. Therefore, I use the indicators which are commonly used in the field of technical analysis. You can see table 1 in appendix for more information.




For each indicator, it has its own property through which traders generally predict the stock's trend. After computing the value of all indicators each day, I transform the value into binary form to represent the prediction of each of them. You can see Table 2. in appendix for more information


The dataset after transforming will be like:
![](https://i.imgur.com/igyLeZp.png =500x)

Due to some indicator requiring sufficient day length of data, the sample with NaN will be dropped.
![](https://i.imgur.com/XCuRDjn.png =500x)
It contains a total 2487 samples with 11 features and 1 label.
After preprocessing the dataset, I split the sample after the year 2021 as test data. All the analysis below will be evaluated by test data.
### Algorithm
#### Random forest
For the first machine learning algorithm, I used a random forest. For its hyper-parameters, I fixed its random state to 42 for reproducibility and used grid search cross validation to find the best estimators from 10 to 200 and the number of estimators with the best performance is 200, the second one is 30 and the third one is 40. I will compare the result with a different number of estimators in the Analysis part.
For other hyper-parameters, they are set as the default option in [sklearn library](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).
#### Support Vector Machine Classifier
For the second algorithm, I use SVM. For its hyper-parameter, I have try the different combination of sets below by grid search cross validation.


| Hyper-parameter | Option 1     | Option 2 | Option 3 | Option 4 |
| --------------- | ------------ | -------- | -------- | -------- |
| kernel          | linear       | poly     | rbf      | sigmoid   |
| C               | 0.1          | 1        | 10       | 100      |
| Degree          | from 1 to 10 |          |          |          |

Other hyper-parameters not listed above are set as the default option in [sklearn library](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

### Analysis
For Analysis, I use the performance metrics below:
| Performance Metric       | Definition                                                |
| ------------------------ | --------------------------------------------------------- |
| Percentage Accuracy(ACC) | (TP + TN) / (TP + TN + FP + FN)                           |
| Recall                   | TP / (TP + FN)                                            |
| Precision                | TP / (TP + FP)                                            |
| F1-score                 | $\frac{2\times Recall\times Precision}{Recall+Precision}$ |
#### Random forest
With different number of estimators:
![](https://i.imgur.com/VisRPeC.png =500x)



With different amount of training data:
Use random forest with 200 estimators as example
![](https://i.imgur.com/MW6JE8R.png =500x)

#### Support Vector Machine
With different hyper-parameter(first 10 places setting in validation):
![](https://i.imgur.com/SU9oPh9.png =500x)


With different amount of training data:
Use SVM with their best hyper-parameters: {'C': 0.1, 'degree': 2, 'kernel': 'poly'} as example.
![](https://i.imgur.com/t0ju7ps.png =500x)


### Discussion
For this experiment, I found out that with a few technical indicators, we can really predict the trend of tomorrow's price. In addition to this, I also find out that the more data doesn't mean the better performance. As you can see above, the best amount of data is around 5 years, neither more or less data is good for the prediction.
Besides the performance metrics I have done above, I also try to simulate the strategy based on model prediction.
![](https://i.imgur.com/7sTaiH8.png)
As you can see above is the return on investment based on random forest prediction given 2015~2020 train data. The strategy simply all in if the model predict that tomorrow will go up and do nothing if go down. As the result, the strategy can beat the long-term holding strategy.
If I got more time, I would like to try the dataset with different time scale like 5 days, 20 days, 15 mins and so on. Besides, I would like to compare the result given different stock and different time.


## Self-made dataset
### Dataset
For a self-made dataset, I use the questionnaire survey I have done in LINE Fresh 2021. The dataset is about the travel habits and their preference of the spots. The dataset contains 784 samples and 34 columns. Our goal is to predict whether a person will like to know more about the spot or not(column 33).
![](https://i.imgur.com/wr7eq5P.png =200x)
### Methodlogy
First of all, not every feature is useful for model prediction. Some of them are totally not related to the label, like timestamp and email. After removing the unnecessary feature, the features remained are:
![](https://i.imgur.com/CetwWwI.png =200x)
Then, I mapped the column name into a more convenient one and also mapped their legal value into processable form too.
![](https://i.imgur.com/Y9nCpD9.png =100x)
Third, I did one-hot encoding for feature **age**, **hometown**, **job**, **info_source**, **travel_date** and also **label**. Because they are either the categorical feature or the multiple choice question in the survey, it is necessary for them to do one-hot encoding.
After all, the training dataset contains 23 features and the label contains 10 columns(each for a choosable spot).
### Algorithm
In this experiment, I choose K-nearest neighbors as my classifier. I have tried different K and compared the result. Besides, since some of the spots are not very popular in the survey, some of the labels are imbalanced. I use random over sampling to balance the distribution of each label.
Before training, I split the dataset into train and test dataset with ratio 0.8:0.2.
### Analysis
![](https://i.imgur.com/UprE3Qe.png =250x)

The plot box above is the performance on a test dataset of each label with K=20. We can find out that the model did not work very well on a specific label. After checking the label, I realize it is an imbalance label with only 10% of the sample preferring to go.

![](https://i.imgur.com/8ByGsC0.png =250x)![](https://i.imgur.com/9wEHZKG.png =250x)
The chart above is the performance with K=5 and K=10. Both of them can not solve the problem of imbalance data although the over sampling technique is used.

### Discussion
In this experiment, we can predict the preference of people about spots. Although the accuracy is around 50%, it is still a great reference for us to find out people's tendencies. Since the label is the multiple choice question in the survey, some of them are very imbalanced, which is a very bad factor that will affect model performance. If I had more time, I would like to find the relationship between each spot. Then, we can advertise the specific spots in some places if we find out that most people will like both of them. Besides, collecting more useful information is the most important task to do before we start the prediction.

## Appendix
### Table
**Table 1.** 
| Indicator                                    | Formulas                                                                      |
| -------------------------------------------- | ----------------------------------------------------------------------------- |
| Simple Moving Average(SMA)                   | $\frac{C_t+C_{t-1}+...+C_{t-n}}{n}$                                           |
| Weighted Moving Average(WMA)                 | $\frac{nC_t+(n-1)C_{t-1}+...+C_{t-9}}{n+(n-1)+...1}$                          |
| Exponential Moving Average(EMA)               | $EMA(k)_t=EMA(k)_{t-1}+\alpha\times(C_t-EMA(k)_{t-1}), \alpha=\frac{2}{k+1}$  |
| Stochastic K%                                | $\frac{C_t-LL_{t-(n-1)}}{HH_{t-(n-1)}-LL_{t-(n-1)}}\times100$                 |
| Stochastic D%                                | $\frac{\sum_{i=0}^{n-1}K_{t-i}}{10}%$                                         |
| Moving Average Convergence Divergence (MACD) | $MACD(n)_{t-1}+\frac{2}{n+1}\times(DIFF_t-MACD(n)_{t-1})$                     |
| CCI (Commodity Channel Index)                | $\frac{M_t-SM_t}{0.015D_t}$                                                   |
| A/D (Accumulation/Distribution) Oscillator   | $\frac{H_t-C_{t-1}}{H_t-Lt}$                                                  |
| Larry William’s R%                           | $\frac{H_n-C_t}{H_n}-L_n \times100$                                           |
| Relative Strength Index (RSI)                | $100-\frac{100}{1+(\sum_{i=0}^{n-1}UP_{t-i}/n)/(\sum_{i=0}^{n-1}DW_{t-i}/n)}$ |
| Momentum                                     | $C_t-C_{t-9}$                                                                 |


$C_t$ is the closing price, $L_t$ is the low price and $H_t$ is the high price at time $t$.
$LL_t$ and $HH_t$ implies lowest low and highest high in the last t days, respectively.$DIFF_t=EMA(12)_t-EMA(26)_t$, where $EMA$ is the exponential moving average. $M_t=\frac{H_t+L_t+C_t}{3}, SM_t=\frac{\sum_{i=1}^{n}M_{t-i+1}}{n}, D_t=\frac{\sum_{i=1}^{n}|M_{t-i+1-SMt}|}{n}.$
$UP_t$ is upward price change while $DW_t$ is downward price change at time $t$. 

**Table 2.**
| Indicator                                    | True if | False if    |
| -------------------------------------------- | ---- | --- |
| Simple Moving Average(SMA)                   | $C_t>SMA(10)_t$     |$C_t\leq SMA(10)_t$      |
| Weighted Moving Average(WMA)                 | $C_t>WMA(10)_t$     | $C_t\leq WMA(10)_t$    |
| Exponential Moving Average(EMA)               | $C_t>EMA(10)_t$     | $C_t\leq EMA(10)_t$    |
| Stochastic K%                                | $K(10)_t>K(10)_{t-1}$     | $K(10)_t\leq K(10)_{t-1}$    |
| Stochastic D%                                |$D(10)_t>D(10)_{t-1}$      | $D(10)_t\leq D(10)_{t-1}$    |
| Moving Average Convergence Divergence (MACD) |$MACD(10)_t>MACD(10)_{t-1}$      | $MACD(10)_t\leq MACD(10)_{t-1}$ |
| CCI (Commodity Channel Index)                |$CCI(10)_t<200$ or$-200\leq CCI(10)_t\leq200$ and $CCI(10)_t>CCI(10)_{t-1}$      | $CCI(10)_t>-200$ or$-200\leq CCI(10)_t\leq200$ and $CCI(10)_t\leq CCI(10)_{t-1}$    |
| A/D (Accumulation/Distribution) Oscillator   |$AD(10)_t>AD(10)_{t-1}$      | $AD(10)_t\leq AD(10)_{t-1}$    |
| Larry William’s R%                           | $R(10)_t>R(10)_{t-1}$      | $R(10)_t\leq R(10)_{t-1}$    |
| Relative Strength Index (RSI)                |$RSI(10)_t<30$ or$30\leq RSI(10)_t\leq70$ and $RSI(10)_t>RSI(10)_{t-1}$      | $RSI(10)_t>70$ or$30\leq RSI(10)_t\leq70$ and $RSI(10)_t<RSI(10)_{t-1}$    |
| Momentum                                     |$MOM(10)_t>0$      | $MOM(10)_t\leq0$    |
### Code
#### Public Image Dataset
```python

import numpy as np
import pandas as pd
import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from PIL import Image
import os
import matplotlib.pyplot as plt



from torch.utils.data import DataLoader


batch_size = 5
Mixup = True
Progressive_Resizing = True
image_size = 375
log_file = open('log_info.txt','a')

class TrainData():
    """
    Train data with the required function for data loader in pytorch 
    """
    def __init__(
            self, class_file, img_file,
            transform=None, target_transform=None,
            is_train=False):
        self.get_labels()
        self.labels = pd.read_table(
            class_file, sep=" ",
            names=['img','label'],
            header=None
            )
        self.img_file = img_file
        self.transform = transform
        self.target_transform = target_transform
        self.is_train=is_train

    def __len__(self):
        return len(self.labels)

    def set_is_train(self,is_train):
        self.is_train = is_train

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_file, self.labels.iloc[idx, 0])
        image = Image.open(img_path)

        label_name = self.labels.iloc[idx,1]
        label_index = int((self.label_list[self.label_list['label'] == label_name]).index.values)

        label = torch.zeros(200)
        label[label_index] = 1.

        if self.transform is not None:
            image = self.transform(image)

        #print(self.is_train)
        # If data is for training, perform mixup, only perform mixup roughly on 1 for every 5 images
        if self.is_train and idx > 0 and idx%5 == 0:
            #print('mix')
            # Choose another image/label randomly

            mixup_idx = np.random.randint(0, len(self.labels)-1)
            img_path = os.path.join(self.img_file, self.labels.iloc[mixup_idx, 0])
            mixup_image  = Image.open(img_path)
            mixup_label_name =  self.labels.iloc[idx,1]
            mixup_label_index = int((self.label_list[self.label_list['label'] == mixup_label_name]).index.values)
            mixup_label = torch.zeros(200)
            mixup_label[mixup_label_index] = 1.
            
            if self.transform is not None:
                mixup_image = self.transform(mixup_image)


            # Select a random number from the given beta distribution
            # Mixup the images accordingly
            alpha = 0.2
            lam = np.random.beta(alpha, alpha)
            image = lam * image + (1 - lam) * mixup_image
            label = lam * label + (1 - lam) * mixup_label


        #plt.imshow(  image.permute(1, 2, 0)  )
        #plt.show()
            
        return image.to(device),label.to(device)

    def get_labels(self):
        self.label_list =  pd.read_table('classes.txt',names=['label'])
        

class TestData():
    """
    Test data with the required function for data loader in pytorch 
    """
    def __init__(
            self, class_file, img_file,
            transform=None, target_transform=None):
        self.get_labels()
        self.labels = pd.read_table(
            class_file, sep=" ",
            names = ['img'], header = None
            )
        self.img_file = img_file
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self,idx):
        img_path = os.path.join(self.img_file, self.labels.iloc[idx, 0])
        image = Image.open(img_path)
        
        if self.transform is not None:
            image = self.transform(image)
     
        return image.to(device),self.labels.iloc[idx, 0]

    def get_labels(self):
        self.label_list =  pd.read_table('classes.txt',names = ['label'])


class Attention(torch.nn.Module):
    """
    Attention block for CNN model.
    """
    def __init__(
            self, in_channels, out_channels, 
            kernel_size, padding):
        super(Attention, self).__init__()
        self.conv_depth = torch.nn.Conv2d(
            in_channels, out_channels, 
            kernel_size, padding = padding, 
            groups = in_channels
            )
        self.conv_point = torch.nn.Conv2d(
            out_channels, out_channels, 
            kernel_size = (1, 1)
            )
        self.bn = torch.nn.BatchNorm2d(
            out_channels, eps=1e-5, 
            momentum = 0.1, affine = True, 
            track_running_stats = True
            )
        self.activation = torch.nn.Tanh()

    def forward(self, inputs):
        x, output_size = inputs
        x = nn.functional.adaptive_max_pool2d(x, output_size=output_size)
        x = self.conv_depth(x)
        x = self.conv_point(x)
        x = self.bn(x)
        x = self.activation(x) + 1.0
        return x


class ResNet152Attention(torch.nn.Module):
    """
    Attention-enhanced ResNet-50 model.
    """
    weights_loader = staticmethod(models.resnet152)

    def __init__(
            self, num_classes=200, 
            pretrained=True, use_attention=True):
        super(ResNet152Attention, self).__init__()
        net = self.weights_loader(pretrained=pretrained)

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.use_attention = use_attention

        
        net.fc = torch.nn.Linear(
            in_features=net.fc.in_features,
            out_features=num_classes,
            bias=net.fc.bias is not None
            )
        self.net = net
        
        if self.use_attention:
            self.att1 = Attention(
                in_channels = 64,
                out_channels = 256,
                kernel_size = (3, 5),
                padding = (1, 2)
                )
            self.att2 = Attention(
                in_channels = 256,
                out_channels = 512,
                kernel_size = (5, 3),
                padding = (2, 1)
                )
            self.att3 = Attention(
                in_channels = 512, 
                out_channels = 1024, 
                kernel_size = (3, 5), 
                padding = (1, 2)
                )
            self.att4 = Attention(
                in_channels=1024, 
                out_channels=2048, 
                kernel_size=(5, 3), 
                padding=(2, 1)
                )

            if pretrained:
                self.att1.bn.weight.data.zero_()
                self.att1.bn.bias.data.zero_()
                self.att2.bn.weight.data.zero_()
                self.att2.bn.bias.data.zero_()
                self.att3.bn.weight.data.zero_()
                self.att3.bn.bias.data.zero_()
                self.att4.bn.weight.data.zero_()
                self.att4.bn.bias.data.zero_()

    def _forward(self, x):
        return self.net(x)
    
    def _forward_att(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x_a = torch.clone(x)
        x = self.net.layer1(x)
        x = x.mul(self.att1((x_a,x.shape[-2:])))

        x_a = x.clone()
        x = self.net.layer2(x)
        x = x * self.att2((x_a, x.shape[-2:]))

        x_a = x.clone()
        x = self.net.layer3(x)
        x = x * self.att3((x_a, x.shape[-2:]))

        x_a = x.clone()
        x = self.net.layer4(x)
        x = x * self.att4((x_a, x.shape[-2:]))

        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x
    
    def forward(self, x):
        if self.use_attention :
            return self._forward_att(x) 
        else :
            return self._forward(x)
        

def pad(img, size_max=500):
    """
    Pads images to the specified size (height x width). 
    """
    pad_height = max(0, size_max - img.height)
    pad_width = max(0, size_max - img.width)
    
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left
    
    return transforms.functional.pad(
        img,
        (pad_left, pad_top, pad_right, pad_bottom),
        fill=tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406))))


def train_loop(dataloader, model, loss_fn, optimizer):
    '''
    The loop for the model training
    '''

    def print_info(loss,train_loss,num_of_img,correct):
        loss, current = loss.item(), batch * len(X)+batch_size
        
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        log_file.write(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]\n")
        train_loss /=  num_of_img
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            train_loss, correct, num_of_img,
            100. * correct / num_of_img))
        log_file.write('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            train_loss, correct, num_of_img,
            100. * correct / num_of_img))
            
        

    model.train()

    size = len(dataloader.dataset)
    train_loss = 0
    correct = 0
    num_of_img = 0
    

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss.
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        train_loss += loss

        # Get the prediction and add one to correct if the prediction 
        # is correct.
        pred = torch.argmax(output, dim=1)
        correct += (pred == torch.argmax(y, dim=1)).float().sum() 
        num_of_img += batch_size

        # Backpropagation
        loss.backward()
        optimizer.step()

        
        
        if batch % 100 == 99:
            # Save the model weight and print training information
            # for every 100 loop.
            torch.save(model.state_dict(), 'model_weights.pth')
            print_info(loss,train_loss,num_of_img,correct)

            train_loss = 0
            correct = 0
            num_of_img = 0

    if batch % 100 != 99:
        torch.save(model.state_dict(), 'model_weights.pth')
        print_info(loss,train_loss,num_of_img,correct)
        

def val_loop(model, test_loader, is_test=False):
    '''
    The loop for the model validation
    '''

    model.eval()  # Set the model to evaluate mode

    test_loss = 0
    correct = 0

    with torch.no_grad(): 
        # disable gradient calculation for efficiency
        for X, y in test_loader:
            
            output = model(X)  # Get model prediction output

        
            test_loss += loss_fn(output, y)  # Add current loss to total loss
            pred = torch.argmax(output, dim=1)  # Get the prediction

            # Add one to correct if the prediction is correct
            correct += (pred == torch.argmax(y, dim=1)).float().sum()

    test_loss /= len(test_loader.dataset)  # Calculate average loss

    # Print testing information
    if is_test:
        print('Test set:')
        log_file.write('Test set:\n')
    else:
        print('Validation set:')
        log_file.write('Validation set:\n')
    print(' Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    log_file.write(' Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    model.train()  # Set the model to training mode


def test(model,test_loader):
    '''
    The loop for the model testing
    '''

    model.eval()

    file = open('answer.txt','w')
    label_list = pd.read_table('classes.txt',names=['label'])
   
    with torch.no_grad():
        for data, img_name in test_loader:
            # Prediction
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            for idx,name in enumerate(img_name):
                file.write(name)
                file.write(' ')
                file.write(label_list.loc[int(pred[idx]),'label'])
                file.write('\n')
            
    file.close()
    
    model.train()


def log_info():
    
    log_file.write('batch_size:'+str(batch_size)+'\n')
    log_file.write('MixUp:'+str(Mixup)+'\n')
    log_file.write('Progressive_Resizing:'+str(Progressive_Resizing)+'\n')
    if Progressive_Resizing:
        log_file.write('img_size:'+str(image_size)+'\n')

log_info()

# Initialize the model and transform used in model
train_transform = transforms.Compose([
   
   transforms.Resize([image_size], antialias=True),
   
   transforms.RandomOrder([
       transforms.RandomCrop((int(image_size),int(image_size))),
       transforms.RandomHorizontalFlip(),
       transforms.RandomVerticalFlip()
   ]),
   
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])

test_transform = transforms.Compose([
   
   
   transforms.CenterCrop((375,375)),
   transforms.ToTensor(),
   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CNN_model = models.resnet152(pretrained=True)
CNN_model.fc = torch.nn.Linear(
            in_features=CNN_model.fc.in_features,
            out_features=200
            )

CNN_model.to(device)
# Ask user if they want to load the weight trained before
print("load?(y/n)")
load=input()
if load=='y':
    try:
        CNN_model.load_state_dict(torch.load('model_weights.pth'))
        print("load weight")
    except:
        print("load failed")

# Get the mode from user's input
print("Choose mode: (1) train (2) test (3) train and test")
mode = int(input())
while not (mode>=1 and mode<=3):
    print("wrong input")
    mode = input()

# Training process
if mode==1 or mode==3:
    # Ask user the initial learning rate
    print("input init learning rate")
    learning_rate=float(input())

    # Ask user how much epoch they want to train
    print("how many epoch?")
    epoch = int(input())

    # Load data and set optimizer, scheduler and loss function
    optimizer = torch.optim.Adam(CNN_model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
    loss_fn = nn.CrossEntropyLoss()

    # Set random seed to ensure the training process reproducible
    torch.manual_seed(0)
    Data = TrainData("training_labels.txt",'training_images',test_transform)
    _,test_dataset = random_split(Data,[2700,300])

    Data = TrainData("training_labels.txt",'training_images',train_transform)
    Data.is_train = Mixup
    torch.manual_seed(0)
    Data,_ = random_split(Data,[2700,300])
    test_dataloader = DataLoader(
            test_dataset,
            batch_size = batch_size, 
            shuffle = True)  
    
   
    
    for i in range(epoch):
        # Loop for 'epoch' times to train the model

        epoch_info = 'epoch:'+ str(i)+ ' learning rate:'+ str(scheduler.get_last_lr()[0])
        print(epoch_info)  
        log_file.write(epoch_info+'\n')

        # Split the data into training data and validation data randomly
        train_dataset,validation_dataset = random_split(Data,[2400,300])
        
        
        train_dataloader = DataLoader( 
            train_dataset, 
            batch_size = batch_size, 
            shuffle = True)
        val_dataloader = DataLoader(
            validation_dataset,
            batch_size = batch_size,
            shuffle = True)         

        # Go to train loop function to train the model
        train_loop(train_dataloader,CNN_model,loss_fn,optimizer)
        # Go to validation loop function to validate the model
        val_loop(CNN_model,val_dataloader)
        # Go to validation loop function to test the model
        val_loop(CNN_model,test_dataloader,True)
        print()
        log_file.write('\n')
        
        scheduler.step() # Decrease learning rate by scheduler 
        

# Testing Process     
if mode==2 or mode==3:
    file_name = 'testing_img_order.txt'
    folder_name = 'testing_images'

    # Load test data from file
    test_dataset = TestData(file_name, folder_name, test_transform)
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size = batch_size, 
        shuffle = False)     

    # Go to test function to test and output predict result 
    test(CNN_model,test_dataloader)

log_file.close()






```


#### Public Non-Image Dataset
```python
import pandas as pd
import numpy as np
import stockstats as ss
import time
import yfinance as yf
import talib as ta

target_id = 'AAPL'
start_date = "2012-01-01"
end_date = "2021-12-31"
test_date = '2021'
day_length = 1
time_period = 5

target = yf.download(target_id, start=start_date, end=end_date, interval="1d")

target.columns = target.columns.map({
    'Open':'open',
    'High':'high',
    'Low':'low',
    'Adj Close':'adj close',
    'Volume':'volume',
    'Close':'close'
})

target['LongShort'] = target['close'].shift(-day_length)
num_label = pd.DataFrame(index=target.index)
num_label['LongShort'] = (target['LongShort']-target['close'])/target['close']
num_label.plot.box()

Indicator = pd.DataFrame(index=target.index)
Det_Indicator = pd.DataFrame(index=target.index)
from talib.abstract import STOCH as KD
KD_ = KD(target,time_period).dropna()

print(KD_)
KD_shift = KD_.shift(1)
print(KD_shift)
KD_['LongShort'] = (KD_['slowk']-KD_['slowd'])
Indicator['KD_k'] = KD_['slowk']
Indicator['KD_d'] = KD_['slowd']
Det_Indicator['KD_k'] = KD_['slowk']>KD_shift['slowk']
Det_Indicator['KD_d'] = KD_['slowd']>KD_shift['slowd']

from talib.abstract import MACD
MACD_ = MACD(target,12,26,time_period)
MACD_ = MACD_.dropna()
MACD_shift = MACD_.shift(1)
MACD_['LongShort'] = MACD_['macdhist']>0
Indicator['MACD'] =  MACD_['macd']
MACD_['macdhist'].plot()
Det_Indicator['MACD'] = MACD_['macd']>MACD_shift['macd']

from talib import SMA

SMA_ = pd.DataFrame(SMA(target['close'],time_period),columns=['SMA'])
SMA_ = SMA_.dropna()
SMA_['LongShort'] = (target['close']-SMA_['SMA'])/SMA_['SMA']
Indicator['SMA'] = SMA_['LongShort']
Det_Indicator['SMA'] = target.loc[SMA_.index,'close']>SMA_['SMA']

from talib import EMA

EMA_ = pd.DataFrame(EMA(target['close'],time_period),columns=['EMA'])
EMA_ = EMA_.dropna()
EMA_['LongShort'] = (target['close']-EMA_['EMA'])/EMA_['EMA']
Indicator['EMA'] = EMA_['LongShort']
Det_Indicator['EMA'] = target.loc[EMA_.index,'close']>EMA_['EMA']

from talib import WMA

WMA_ = pd.DataFrame(WMA(target['close'],time_period),columns=['WMA'])
WMA_  = WMA_.dropna()
WMA_['LongShort'] = (target['close']-WMA_['WMA'])/WMA_['WMA']
Indicator['WMA'] = WMA_['LongShort']
Det_Indicator['WMA'] = target.loc[WMA_.index,'close']>WMA_['WMA']

from talib import RSI
RSI_ = pd.DataFrame(RSI(target['close'],time_period),columns=['RSI'])
RSI_ = RSI_.dropna()
Indicator['RSI'] = RSI_['RSI']

RSI_shift = RSI_.shift(1)

Det_Indicator['RSI'] = False
RSI_['below30'] = RSI_['RSI']<30
RSI_['Increase'] = RSI_['RSI']>RSI_shift['RSI']

Det_Indicator['RSI'] = Det_Indicator['RSI']|RSI_['below30']|RSI_['Increase']

Det_Indicator['RSI'].value_counts()

from talib import MOM
MOM_ = MOM(target['close'], timeperiod=time_period)
MOM_ = MOM_.dropna()

Indicator['MOM'] = MOM_
Det_Indicator['MOM'] = MOM_>0

from talib import CCI
CCI_ = pd.DataFrame(CCI(target['high'],target['low'],target['close'],time_period),columns=['CCI'])
CCI_=CCI_.dropna()

CCI_shift = CCI_.shift(1)

Indicator['CCI'] = CCI_
Det_Indicator['CCI'] = False
CCI_['below200'] = CCI_['CCI']<200
CCI_['Increase'] = CCI_['CCI']>CCI_shift['CCI']
Det_Indicator['CCI'] = Det_Indicator['CCI']|CCI_['Increase']|CCI_['below200'] 

from talib import WILLR
WILLR_ = WILLR(target['high'],target['low'],target['close'],time_period)
WILLR_=WILLR_.dropna()
WILLR_shift=WILLR_.shift(1) 
Indicator['WILLR'] = WILLR_
Det_Indicator['WILLR'] = WILLR_>WILLR_shift

from talib import AD
AD_ = AD(target['high'],target['low'],target['close'],target['volume'])
AD_shift = AD_.shift(1)
Det_Indicator['AD'] = AD_>AD_shift

five_days = pd.concat([Det_Indicator,pd.DataFrame(num_label['LongShort'])],axis=1)
five_days

five_days.dropna(inplace=True)
five_days

from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score
def get_performance(true,pred,out=False):
    matrix = confusion_matrix(true,pred,labels=[True,False])
    result = {
        
        'acc':accuracy_score(true,pred),
        'win_rate':matrix[0][0]/(matrix[0][0]+matrix[1][0]),
        'precision':precision_score(true,pred),
        'recall':recall_score(true,pred)
    }
    return result

from sklearn.model_selection import train_test_split


X = five_days.iloc[:,:-1]
y = five_days['LongShort']>0

X_train = X[five_days.index<test_date]
X_test = X[five_days.index>=test_date]
y_train = y[five_days.index<test_date]
y_test = y[five_days.index>=test_date]

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import GridSearchCV
clf = RFC(random_state=42,n_jobs=5)

parameters = {
    'n_estimators': np.arange(10, 210, 10).tolist(),
    
    
    }

GS = GridSearchCV(clf, parameters,n_jobs=5)
GS.fit(X_train, y_train)



print(GS.best_params_)

from sklearn.ensemble import RandomForestClassifier
clf =  RandomForestClassifier(**GS.best_params_,n_jobs=5,random_state=42)
clf.fit(X_train,y_train)
result = clf.predict(X_test)

print(get_performance(y_test,result))
print(confusion_matrix(y_test,result,labels=[True,False]))
y_test.value_counts()

print(np.average(five_days.loc[:,'LongShort']))
print(np.average(five_days.loc[y_test[result].index,'LongShort']))

print(np.average(five_days.loc[y_test[np.array(y_test) & result ].index,'LongShort']))
print(np.average(five_days.loc[y_test[~np.array(y_test) & result ].index,'LongShort']))

import matplotlib.pyplot as plt

earn =np.log(pd.DataFrame(five_days.loc[y_test[result].index,'LongShort'])+1)

earn = earn.cumsum()

earn =np.exp(earn) 
earn.plot()


price_reward =  target[target.index>=test_date]['close'].copy()
price_reward/=price_reward[0]

price_reward.plot(label='Stock Price')
plt.legend()
print('Final reward:',earn.iloc[-1])
print('Stock grow:',price_reward.iloc[-1])

```
#### Self-Made Dataset 
```python
# %%
import pandas as pd
import numpy as np

dataset = pd.read_excel('pj1\Dataset\LINE FRESH.xlsx')


# %%
pd.Series(dataset.columns)

# %%
dataset = dataset.drop(dataset.columns[[0,1]+list(range(6,23))],axis=1)
dataset = dataset.drop(dataset.columns[15],axis=1)

# %%
pd.Series(dataset.columns)

# %%
columns = dataset.columns
dataset.columns=['gender','age','hometown','job','rank_spot','rank_trans','rank_spend','rank_food','rank_hotel','rank_shopping','rank_amusement','rank_atmosphere','info_source','travel_date','label']


# %%
pd.Series(dataset.columns)

# %%
dataset['gender'] = dataset['gender'].map({
    '男':'0',
    '女':'1'
})
dataset['age'] = dataset['age'].map({
    '小於 18 歲':0,
    '18~22 歲':1,
    '23~30 歲':2,
    '31~40 歲':3,
    '41~50 歲':4,
    '51~60 歲':5,
    '大於 60 歲':6
})
dataset['hometown'] = dataset['hometown'].map({
    '北北基':'north',
    '桃竹苗':'chu',
    '中彰投':'mid',
    '雲嘉南':'south',
    '高屏':'ping',
    '宜花東':'east',
    '日本':'foreign',
    '阿姆斯特丹':'foreign',
    '新加坡':'foreign',
    '美國':'foreign',
    '離島地區（金門、馬祖、澎湖、綠島、蘭嶼等）':'island',
    '美國加州':'foreign',
    'United States':'foreign'
})
dataset['job'] = dataset['job'].map({
    '學生':'student',
    '工商業':'industry and commerce',        
    '軍公教'       :'public employee',
    '服務業'       :'service',
    '自由業'       :'free',
    '家管'         :'free',
    '工程師'       :'technology',
    '退休人士'     :'no',
    '科技業'       :'technology',
    '製造業'       :'industry and commerce',
    '研究助理'     :'technology',
    '傳產技術員'   :'technology',
    '醫療業'       :'health',
    '待業'         :'no',
    '保險業'       :'industry and commerce',
    '半導體業'     :'technology',
    '金融業'       :'industry and commerce',
    'Engineer'     :'technology',
    '待業中'       :'no',
    '園藝'         :'free',
    '保母'         :'free',
    '半導體'       :'technology',
    '半導體製造業' :'technology',
    '電子'         :'technology',
    '程式設計師'   :'technology',
    '無業QQ'       :'no',
    '無業'         :'no'
})

# %%
from sklearn.preprocessing import OneHotEncoder
age_onehot = OneHotEncoder()
age = pd.DataFrame(age_onehot.fit_transform(np.array(dataset.loc[:,['age']]).reshape(-1, 1)).toarray(),columns=age_onehot.get_feature_names_out(['age']))
hometown_onehot = OneHotEncoder()
hometown = pd.DataFrame(age_onehot.fit_transform(np.array(dataset.loc[:,['hometown','job']])).toarray(),columns=age_onehot.get_feature_names_out(['hometown','job']))

X = pd.concat([dataset.loc[:,['gender']],age,hometown],axis=1)

# %%
info_source = pd.DataFrame(index=dataset.index)
info_source['info_social_media'] =  dataset['info_source'].str.contains('社群媒體')
info_source['info_internet'] =  dataset['info_source'].str.contains('網路搜尋')
info_source['info_youtube'] =   dataset['info_source'].str.contains('Youtuber')
info_source['info_map'] =   dataset['info_source'].str.contains('Map')
info_source['info_friend'] =   dataset['info_source'].str.contains('親友')
info_source['info_app'] =   dataset['info_source'].str.contains('旅遊app')
info_source['info_news'] =   dataset['info_source'].str.contains('新聞報導')
info_source['info_magzine'] =   dataset['info_source'].str.contains('旅遊雜誌')
info_source['info_line'] = (  dataset['info_source'].str.contains('LINE 旅遊')) | (  dataset['info_source'].str.contains('LINE 熱點'))

# %%
travel_date = pd.DataFrame(index=dataset.index)
travel_date['date_weekday'] =  dataset['travel_date'].str.contains('平日') 
travel_date['date_weekend'] =  dataset['travel_date'].str.contains('周末') 
travel_date['date_vacation'] =  dataset['travel_date'].str.contains('連續假期')
travel_date['date_winter_summer_vacation'] =  dataset['travel_date'].str.contains('寒暑假')
travel_date['date_holiday'] =  dataset['travel_date'].str.contains('當地舉行特色活動時')

# %%
dataset.describe()

# %%
X = pd.concat([X,info_source,travel_date],axis=1)
y = pd.DataFrame(index=dataset.index)

# %%
y['story_library'] = dataset['label'].str.contains('台東故事館')
y['blue'] =  dataset['label'].str.contains('台東藍晒圖')
y['music_market'] =  dataset['label'].str.contains('鐵花村音樂聚落市集')
y['tample'] = dataset['label'].str.contains( '台東天后宮')
y['rail'] =  dataset['label'].str.contains('台東舊鐵道路廊')
y['dong_dong'] = dataset['label'].str.contains('東東市' )
y['museum'] =  dataset['label'].str.contains('台東美術館')
y['forest_park'] =  dataset['label'].str.contains('台東森林公園')
y['fish_park'] =  dataset['label'].str.contains('鯉魚山公園')
y['coastal_park'] =  dataset['label'].str.contains('海濱公園')

# %%
X.describe()

# %%
X

# %%
X=X.dropna()
y=y.loc[X.index,:]

# %%
X

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)

# %%
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
def get_performance(true,pred,out=False):
    #
    result = {
        'matrix':confusion_matrix(true,pred,labels=[True,False]),
        'acc':accuracy_score(true,pred),
        
        'precision':precision_score(true,pred),
        'recall':recall_score(true,pred),
        'f1_score':f1_score(true,pred)
    }
    return result
    

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
neighbor = 20
performance = {}
for i in range(len(y_train.columns)):
    ros =RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(X_train, y_train.iloc[:,i])
    clf = KNeighborsClassifier(n_neighbors=neighbor)
    clf.fit(X=X_resampled,y=  y_resampled)
    print(y_train.iloc[:,i].value_counts())
    result = clf.predict(X_test)
    performance[y_train.columns[i]] = get_performance(y_test.iloc[:,i],result)
    

# %%
performance = pd.DataFrame(performance)
performance[1:].T.plot.box(title='{} nearest neighbors'.format(neighbor))

performance.T

```
