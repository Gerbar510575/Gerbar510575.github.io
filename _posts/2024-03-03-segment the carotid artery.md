# 深度學習方法解決醫學影像實務問題之流程

## 1. 現實問題之釐清
- 本專案旨在自動從頸部超音波影像中標註頸動脈區域，以協助醫師進行非侵入式、快速且準確的血管判讀。
- 該任務屬於典型的**語義分割問題（semantic segmentation）**，需要為輸入影像中每一個像素預測其是否屬於頸動脈。
- 本任務源自課堂作業：提供三位受試者的頸部超音波影片，從中各擷取 100 張影格作為訓練資料，另提供另一位受試者影像 100 張作為測試資料。
- 所有影像皆經由放射科醫師標註頸動脈區域，作為訓練時的標籤資料。

## 2. 轉換成深度學習問題
- 任務定義：輸入為 RGB 超音波影像，輸出為與影像等尺寸的二值遮罩圖（1 為頸動脈區域，0 為背景）。
- 本專案同時實作兩種模型：
  - FCN-8s（具 skip connection 機制）
  - U-Net（醫學影像常用架構）
- 預測準確度以 **Dice coefficient** 作為指標，並作為 Kaggle 評分依據。
- 使用交叉熵損失函數（CrossEntropyLoss）進行監督式訓練。

## 3. 資料蒐集與前處理
- 訓練資料來自 `train/pre` 資料夾（影像）與 `train/post`（標註遮罩），測試資料來自 `test/pre`。
- 每張影像之對應遮罩名稱為 `(原檔名)_ROI.bmp`。
- 使用 `train.csv` 與 `test.csv` 對應影像檔名與預測輸出命名。
- 資料增強與標準化流程：
  - 調整尺寸至固定大小（如 448x448）
  - Normalize（使用 ImageNet 預設均值與標準差）
  - Tensor 化（使用 `ToTensorV2()`）

## 4. 資料探索 (EDA)
### 4.1 資料分佈與可視化
- 將訓練影像與遮罩重疊視覺化，檢查遮罩貼合品質與異常。
- 標註區域大多集中於中央橫條紋狀區域，符合理論上頸動脈所在位置。
- 樣本變異度適中，未見嚴重偏態。

### 4.2 訓練樣本劃分
- 將 300 張訓練影像劃分為 80% 訓練集、20% 驗證集，使用固定 seed 保證可重現。
- 使用 `SonoDataset` 包裝索引抽樣，確保相同資料不同切分不重複使用。

## 5. 模型建構與訓練
### 5.1 模型設計
- **FCN-8s**
  - 使用多層 CNN 卷積與 max pooling 下採樣後，利用 skip connection 結合不同尺度特徵圖，並透過 transpose convolution 還原輸出尺寸。
- **U-Net**
  - Encoder-Decoder 架構，結合對稱 encoder 與 decoder 並有跨層連結（concatenate），強化空間定位資訊。
  - Backbone 採用 `ResNet18`，DecoderBlock 實作反卷積與卷積層混合模組。

### 5.2 訓練設定
- 批次大小：8 或 16（Kaggle Sweep 測試）
- Optimizer：Adam（lr = 5e-5, weight decay = 1e-8）
- Epoch：10
- 使用 `wandb` 記錄訓練過程與指標（Loss、Dice、mIoU）
- 儲存最佳模型檢核點（best_ckpt.pth）與最後一輪模型（last_ckpt.pth）

## 6. 模型評估與視覺化
- 驗證指標包括：
  - 全域準確率（global accuracy）
  - 平均 IoU（mean IoU）
  - 平均 Dice Score（mean Dice）
- 額外可視化圖包含：
  - 原圖
  - Ground Truth Mask
  - 模型預測機率圖
  - 預測二值圖（閾值 0.5）
  - 錯誤融合圖（紅：FP、綠：TP、藍：FN）

## 7. 是否解決現實問題？
- 在 Kaggle 測試集上，模型於 Dice Score 可達 0.87 左右，展現頗佳的 segmentation 能力。
- 可於臨床應用中協助醫師快速初步判讀頸動脈狹窄或病變位置。
- 模型訓練參數可壓至百萬級（~6M），可於嵌入式系統或邊緣裝置上運行。
- 未來可延伸至即時影片解析，搭配滑動視窗完成即時血管追蹤。

## Introduction
Deep learning models have shown remarkable success in medical image analysis tasks, including semantic segmentation.
This project aims to develop predictive models to label carotid artery areas 
of input sonography images which are obtained from EDA hospital.
I used the deep learning model ResNet18 Unet under Pytorch framework to do the analysis and not only got the amazing testing dice cofficient of 0.96505 but also ranked the top three among competitors from Statistics, Computer Science background.
[Kaggle Competition](https://www.kaggle.com/competitions/mia-hw4/leaderboard) (Ranking: 3/31, Testing Dice Cofficient: 0.96505)
![](/images/ranking.png "My Competition Ranking")

## Challenge
1. **Choosing between Last or Best parameter setting?**
   - After training each epoch, model parameters were logged and the best one was chosen in the condition that smallest validation loss. However, parameters from last epoch are not need to be the best durning the whole training process. So, how to determine whether training with more epochs or staying in the best in the current epoch is the trade-off between prediction accuracy and the time consumption.

2. **Try different combinations of hyperparameters?**
   - There are infinite combinations of hyperparameters, in my project I mainly focus on trying different combinations of learning rate and weight decay using grid search method. With the help of visualization provided by wandb, it is easier to compare and understand the performance of each combination comprehensively and quickly.

3. **Validation loss remain the same in the each training epoch**
   - I encounter the problem of the not declined validation loss. The predictions are identical, increasing the epochs may not improve the model's performance. Trying the different learning rate help to avoid getting stuck in a local minimum of the loss function. 

4. Trying to use binary entropy loss always feels like it could lead to better performance in binary classification tasks, but due to time constraints, this attempt was not successful, instead common cross entropy loss was used as a criterion for computing loss.

5. Because the characteristic of this set of images, training the model with more images (increasing from 240 to 270) allow the model to better capture the characteristics of the images.

6. Without the help of Cross Validation, it's impossible to know if there's a possibility of overfitting.
7. Due to the lack of data, data augmentation may help to enhance training result.




## Data Preparation
Sonography videos were taken from left and right necks of three volunteers. We randomly extracted 100 image frames from each volunteer’s video to form in total 300 training sonography images. The test data were taken from another volunteer and 100 test sonography images were created. Several radiologists to label carotid artery area from each image frame. 

After collecting the 300 training images and 100 testing images, I present 16 images combined with mask(yellow portion) labeled by the experts.
![](/images/artery.png "images with masks")
Next, per image corresponding mask are transformed by resizing, normalizing and numericalizing into a tensor which is the standard data format used as an input of deep learning model.
Then, Appropriate Dataloder are created for loading training data
```python
class DeepMedical(torch.utils.data.Dataset):
    def __init__(self, images, transforms = None):
        self.transforms = transforms   
        
        self.image_paths = images
        self.mask_paths = [image.replace('pre', 'post') + '_ROI.bmp' for image in images]

        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = np.array(Image.open(image_path).convert('RGB'))
        mask = np.array(Image.open(mask_path)) * 1

        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
            
        return image, mask
```

## Model Building
FCN model was first being used, however it's testing dice accuracy not surpassing 0.8 threshold. Thus, I turned into Unet model proposed by  . Especially, ResNet18 was used as encoder block and the decoder block follwed by original paper. Pre-trained model was deployed because of the small amount of data. Fine-tuning process was displayed visualizing in the wandb api dashboard. After tring the various combination of hyperparameter such as learning rate and weight decay etc. I used the set of parameter listed below as my hyperparamter setting. Include code snippets or references to notebooks where the model building process is documented.

**Key Concepts**

1. Feature extractor

In segmentation models, feature extractor layers are used throughout the network, allowing it to process spatial information efficiently. These layers capture local patterns and structures in the input data.

2. Skip Connections

To recover fine-grained details lost during downsampling, Unet uses skip connections. These connections combine feature maps from early layers with those from later layers, aiding in the reconstruction of high-resolution information.

3. Upsampling

Segmentation models employ upsampling techniques to restore the spatial resolution of the feature maps. Transposed convolutions or bilinear interpolation can be used for this purpose.


#### U-net

![](/images/UNET.png "fast.ai's logo")

```python
class ResNet18Unet(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(ResNet18Unet, self).__init__()
        base = resnet18(pretrained=pretrained)

        self.firstconv = base.conv1
        self.firstbn = base.bn1
        self.firstrelu = base.relu
        self.firstmaxpool = base.maxpool
        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        out_channels = [32, 64, 128, 256]

        self.center = DecoderBlock(
            in_channels=out_channels[3],
            out_channels=out_channels[3],
            kernel_size=3,
        )
        self.decoder4 = DecoderBlock(
            in_channels=out_channels[3] + out_channels[2],
            out_channels=out_channels[2],
            kernel_size=3,
        )
        self.decoder3 = DecoderBlock(
            in_channels=out_channels[2] + out_channels[1],
            out_channels=out_channels[1],
            kernel_size=3,
        )
        self.decoder2 = DecoderBlock(
            in_channels=out_channels[1] + out_channels[0],
            out_channels=out_channels[0],
            kernel_size=3,
        )
        self.decoder1 = DecoderBlock(
            in_channels=out_channels[0] + out_channels[0],
            out_channels=out_channels[0],
            kernel_size=3,
        )

        self.finalconv = nn.Sequential(
            nn.Conv2d(out_channels[0], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            #nn.Dropout2d(0.1, False),
            nn.Conv2d(32, num_classes, 1),
        )

    def forward(self, x):
        # stem
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x_ = self.firstmaxpool(x)

        # Encoder
        e1 = self.encoder1(x_)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        center = self.center(e4)

        d4 = self.decoder4(torch.cat([center, e3], 1))
        d3 = self.decoder3(torch.cat([d4, e2], 1))
        d2 = self.decoder2(torch.cat([d3, e1], 1))
        d1 = self.decoder1(torch.cat([d2, x], 1))

        f = self.finalconv(d1)

        return f
```

```python
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, in_channels // 4, kernel_size, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv = nn.ConvTranspose2d(
            in_channels // 4,
            in_channels // 4,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(
            in_channels // 4,
            out_channels,
            kernel_size=kernel_size,
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.deconv(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        return x
```

## Metrics

Two commonly used metrics to evaluate the performance of segmentation algorithms are Dice Coefficient and Intersection over Union (IoU).

### Dice Coefficient

The Dice Coefficient, also known as the F1 Score, is a measure of the similarity between two sets. In the context of image segmentation, it is used to quantify the agreement between the predicted segmentation and the ground truth.

The formula for Dice Coefficient is given by:

$$ Dice = \frac{2 \times |X \cap Y|}{|X| + |Y|} $$

where:
- $X$ is the set of pixels in the predicted segmentation,
- $Y$ is the set of pixels in the ground truth,
- $|\cdot|$ denotes the cardinality of a set (i.e., the number of elements).

Dice Coefficient ranges from 0 to 1, where 1 indicates a perfect overlap between the predicted and ground truth segmentations.

### Intersection over Union (IoU)

IoU, also known as the Jaccard Index, is another widely used metric for segmentation evaluation. It measures the ratio of the intersection area to the union area between the predicted and ground truth segmentations.

The formula for IoU is given by:

$$ IoU = \frac{|X \cap Y|}{|X \cup Y|} $$

where:
- $X$ is the set of pixels in the predicted segmentation,
- $Y$ is the set of pixels in the ground truth.

Similar to Dice Coefficient, IoU ranges from 0 to 1, with 1 indicating a perfect overlap.

![](https://www.mathworks.com/help/vision/ref/jaccard.png)

### Interpretation

- **High Values**: A higher Dice Coefficient or IoU indicates better segmentation performance, as it signifies a greater overlap between the predicted and ground truth regions.

- **Low Values**: Lower values suggest poor segmentation accuracy, indicating a mismatch between the predicted and ground truth segmentations.

### Implementation

Dice coefficient and IoU can be calculated by confusion matrix. Therefore, the initial step is to build an confusion matrix from scratch.

**Algorithm: Building Confusion Matrix $M$**

**Input:**
- a: Target labels tensor
- b: Predicted labels tensor
- num_classes: Number of classes

**Procedure:**
1. Initialize the confusion matrix (self.mat) if it is not already created:
   - Create a square matrix of zeros with shape (num_classes, num_classes) and dtype=torch.int64.
   - Place the matrix on the same device as the input tensor `a`.

2. Update the confusion matrix using the update method:

   a. Check for valid class indices:
      - Create a boolean mask k, where elements are True if a is in the range [0, num_classes) and False otherwise.
      
   b. Calculate indices for updating the confusion matrix:
      - Convert the valid elements of `a` and `b` to torch.int64 and calculate the indices using the formula n * a[k] + b[k], where n is the number of classes.
         - **We represent class-i pixels classify to class-j as $\to n * i + j$**
      - Increment the corresponding elements in the confusion matrix using torch.bincount.

3. Compute segmentation metrics using the compute method:
   - Convert the confusion matrix to a float tensor h.
   - Extract correct predictions along the diagonal of the matrix.
   - Compute metrics from $M$
      - `acc` $\to M_{ii}/M_{i\cdot}$
      - `global_acc` $\to sum(M_{ii})/M_{\cdot\cdot}$
      - `dice` $\to \frac{2M_{ii}}{M_{i\cdot}+M_{\cdot i}}$
      - `iou` $\to \frac{M_{ii}}{M_{i\cdot}+M_{\cdot i}-M_{ii}}$

**Output:**
- The confusion matrix is updated and segmentation metrics are computed.
- The $(i, j)-$ terms of the $M$ represents class-i pixels classify to class-j

Note: In practice, we often omit the metrics from the background!!

