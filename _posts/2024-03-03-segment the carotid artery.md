# Deep learning in medical image analysis_semantic segmentation

## Introduction
Provide a brief introduction to the project, This project aims to develop predictive models to label carotid artery areas 
of input sonography images, objectives, and any relevant background information.

## Data Preparation
Describe the process of collecting or obtaining the data for the project. Include details such as the We had taken sonography videos from left and right necks of three volunteers. We randomly extracted 100 image frames from each volunteer’s video to from in total 300 training sonography images. The test data were taken from another volunteer and 100 test sonography images were created. We had asked several radiologists to label carotid artery area from each image frame.

```python
# Example code block for data preparation
The data preprocessing techniques including Run-Length Encoding (RLE) which is a simple and efficient method for representing segmented masks in computer vision and image processing. In the context of image segmentation, RLE is often used to compress and store binary masks, where each pixel is either part of the segmented object (foreground) or not (background).
```

## Exploratory Data Analysis (EDA)
![](/images/artery.png "fast.ai's logo")
Conduct exploratory data analysis to gain insights into the data. Include visualizations and summary statistics to describe the data's characteristics, distributions, and relationships between variables.

## Model Building
Outline the process of building machine learning or statistical models for the project. Describe the model selection, training, and hyperparameter tuning procedures. Include code snippets or references to notebooks where the model building process is documented.

**Key Concepts**

1. Feature extractor

In segmentation models, feature extractor layers are used throughout the network, allowing it to process spatial information efficiently. These layers capture local patterns and structures in the input data.

2. Skip Connections

To recover fine-grained details lost during downsampling, FCN uses skip connections. These connections combine feature maps from early layers with those from later layers, aiding in the reconstruction of high-resolution information.

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

## Evaluation
Evaluate the performance of the models using appropriate metrics and techniques. Discuss the strengths and limitations of the models and any insights gained from the evaluation process.

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

$$
IoU = \frac{|X \cap Y|}{|X \cup Y|} 
$$

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
- The $(i, j)-$terms of the $M$ represents class-i pixels classify to class-j

Note: In practice, we often omit the metrics from the background!!

## Conclusion
Summarize the key findings and conclusions of the project. Reflect on the challenges faced, lessons learned, and potential future directions for the project.

## References
List any references, datasets, libraries, or other resources used in the project.


