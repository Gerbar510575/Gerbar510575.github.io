# 統計方法解決實務問題之流程

## 1、現實問題之釐清
本研究的動機，出於對極右翼政治立場政黨 "Alternative für Deutschland" （AfD） 在 2025 年德國聯邦議院選舉 (Bundestagswahl 2025) ，以 20.6 選區得票率、20.8 政黨得票率的表現，一舉躍升為第21 屆聯邦議會 (the 21st Bundestag) 第二大黨，這個現象的好奇。而之所以感興趣，正是因為百年前於 1930 年第五屆德意志國國會選舉，希特勒 (Adolf Hitler) 領導的極右翼政治立場政黨納粹黨 (Nationalsozialistische Deutsche Arbeiterpartei) 就是以 18.25 得票率表現，一舉躍升為國會第二大黨。然而，極右翼政黨興起的成因錯綜複雜，我們觀察德國 16 邦於2025 年德國聯邦議院選舉，各邦得票率最高之政黨示意圖，發現存在 AfD 於前東德五邦中的支持率遠高過前西德各邦的現象。因此，本研究將以「前東德/前西德」這個角度切入分析極右翼政黨興起的成因。

## 2、轉換成統計問題
過往許多文獻從不同領域不同層面討論造成 AfD 於前東德/前西德邦支持率差異的原因。其中，Götzel 對 2021 年德國官方蒐集的 GLES 資料，使用邏輯斯迴歸驗證「居住地位於前東德/前西德」因素，是否會對「支持 AfD 政黨與否」二元應變數，具有顯著的影響？研究結果顯示，儘管在控制住主觀、客觀經濟變數、宗教教派變數、性別年齡變數以及民粹 (populist) 意識形態變數後，居住於前東德的選民，仍舊相較於其前西德同胞，平均上來說，統計上顯著地高出百分之 4.4 的機率支持AfD。再多控制住一個本土 (nativist) 意識形態變數後，依然也還有高出百分之 3.2 的機率支持AfD。其中，作者發現控制住民粹、本土意識形態變數，對居住於前東德的選民支持AfD 機率造成的影響最大。因此，本研究提出統計模型探討「居住地位於前東德/前西德」與「民粹、本土意識形態變數」的關係，我們將 Götzel 未使用「對觀測值間強烈的自相關性建模」以及「對迴歸係數、個體效應建立混合狀態模型」整合進面板模型。

## 3、資料蒐集與前處理
本研究針對 GLES 面板第 22 至第 27 波次（共六期）的問卷資料，擷取在六期中皆有出現之受訪者識別碼，最終獲得 5,156 位合格選民。接著，針對「居住地」變數，排除於不同波次中出現居住地不一致的受訪者；對於居住地一致但存在缺失值者，則進行資料填補處理。再者，針對衡量民粹與本土意識形態的四個問卷題項：文化融合（Cultural Integration）、移民政策（Immigration Policy）、國家情緒（National Sentiment）、以及強人領導支持（Strong Leadership Support）（詳見表~\ref{tab:var_summary}），我們排除完全未填答之受訪者，並對部分波次填答缺漏者進行「向前填補」（forward fill），以降低樣本損失的程度。此外，為確保上述四個變數在主成分分析（PCA）詮釋方向上的一致性，亦針對文化融合題項進行反向編碼處理。最後，我們將說明本研究中應變數、解釋變數與控制變數之取得方式。

## 4、資料探索 (EDA)
### 4.1 視覺化
### 4.2 初步模型配適

## 5、配適混合自迴歸面板模型 (自行復刻的演算法)

## 6、模型評估 (有母數標準誤差)

## 7、是否解決現實問題？

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
I define a class `FedMinScraper` aimed at extracting monthly US Federal Reserve minutes from the Federal Reserve website. It takes a list of dates as input, specifying the periods for which transcripts are to be extracted. The class utilizes multithreading for faster extraction and parsing of the transcripts. 

```python
class FedMinScraper(object):
    """
    The purpose of this class is to extract monthly US federal reserve minutes
    
    Parameters
    ----------
    dates: list('yyyy'|'yyyy-mm')
        List of strings/integers referencing dates for extraction
        Example:
        dates = [min_year] -> Extracts all transcripts for this year
        dates = [min_year,max_year] -> Extracts transactions for a set of years
        dates = ['2020-01'] -> Extracts transcripts for a single month/year

    nthreads: int
        Set of threads used for multiprocessing
        defaults to None

    Returns
    --------
    transcripts: txt files

    """

    url_parent = r"https://www.federalreserve.gov/monetarypolicy/"
    url_current = r"fomccalendars.htm"

    # historical transcripts are stored differently
    url_historical = r"fomchistorical{}.htm"
    # each transcript has a unique address, gathered from url_current or url_historical
    url_transcript = r"fomcminutes{}.htm"
    href_regex = re.compile("(?i)/fomc[/]?minutes[/]?\d{8}.htm")

    def __init__(self, dates, nthreads=5, save_path=None):

        # make sure user has given list with strings
        if not isinstance(dates, list):
            raise TypeError("dates should be a list of yyyy or yyyymm str/int")

        elif not all([bool(re.search(r"^\d{4}$|^\d{6}$", str(d))) for d in dates]):
            raise ValueError("dates should be in a yyyy or yyyymm format")

        self.dates = dates
        self.nthreads = nthreads
        self.save_path = save_path

        self.ndates = len(dates)
        self.years = [int(d[:4]) for d in dates]
        self.min_year, self.max_year = min(self.years), max(self.years)
        self.transcript_dates = []
        self.transcripts = {}
        self.historical_date = None

        self._get_transcript_dates()

        self.start_threading()

        if save_path:
            self.save_transcript()

    def _get_transcript_dates(self):
        """
        Extract all links for
        """

        r = requests.get(urljoin(FedMinScraper.url_parent, FedMinScraper.url_current))
        soup = BeautifulSoup(r.text, "lxml")
        # dates are given by yyyymmdd

        tdates = soup.findAll("a", href=self.href_regex)
        tdates = [re.search(r"\d{8}", str(t))[0] for t in tdates]
        self.historical_date = int(min(tdates)[:4])
        # find minimum year

        # extract all of these and filter
        # tdates can only be applied to /fomcminutes
        # historical dates need to be applied to federalreserve.gov

        if self.min_year < self.historical_date:
            # just append the years i'm interested in
            for y in range(self.min_year, self.historical_date + 1):

                r = requests.get(
                    urljoin(
                        FedMinScraper.url_parent, FedMinScraper.url_historical.format(y)
                    )
                )
                soup = BeautifulSoup(r.text, parser="lxml")
                hdates = soup.find_all("a", href=self.href_regex)
                tdates.extend([re.search(r"\d{8}", str(t))[0] for t in hdates])

        self.transcript_dates = tdates

    def get_transcript(self, transcript_date):

        transcript_url = urljoin(
            FedMinScraper.url_parent,
            FedMinScraper.url_transcript.format(transcript_date),
        )
        r = requests.get(transcript_url)

        if not r.ok:
            transcript_url = urljoin(
                FedMinScraper.url_parent.replace("/monetarypolicy", ""),
                r"fomc/minutes/{}.htm".format(transcript_date),
            )
            r = requests.get(transcript_url)

        soup = BeautifulSoup(r.content, "lxml")
        main_text = soup.findAll(name="p")

        clean_main_text = "\n\n".join(t.text.strip() for t in main_text)

        # reduce double spaces to one
        clean_main_text = re.sub(r"  ", r" ", clean_main_text)

        self.transcripts[transcript_date] = clean_main_text

    def start_threading(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.nthreads) as executor:
            executor.map(self.get_transcript, self.transcript_dates)

    def save_transcript(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        for fname, txt in self.transcripts.items():
            with open(
                os.path.join(self.save_path, fname + ".txt"), "w", encoding="utf-8"
            ) as o:
                o.write(txt)
                o.close()
```

The class uses BeautifulSoup for web scraping, fetching the URLs of the transcripts based on the provided dates. It then extracts the main text content from the fetched transcripts, cleans the text, and stores the cleaned transcripts in a dictionary with keys representing the respective dates. 

The extracted transcripts can also be saved to a specified directory if provided. The class verifies the input dates and ensures they are in the correct format before proceeding with extraction. It distinguishes between current and historical transcripts based on the provided dates and constructs the appropriate URLs for extraction.

Overall, this class provides a convenient way to automate the retrieval and storage of Federal Reserve meeting transcripts for analysis or archival purposes.

After having the FOMC minutes text documents, I truncated the documents discarding the unnecessary information such as the list of attendants presented in the begining of the FOMC minute. 
![](/images/minute.png "Attendance info should be truncated before NLP")

## EDA

### Descriptive Statistics

![](/images/EDA.png "minute information")

### Paragraphs and words overtime

![](/images/year.png "paragraph and word over time")

## Model Building
FCN model was first being used, however it's testing dice accuracy not surpassing 0.8 threshold. Thus, I turned into Unet model proposed by  . Especially, ResNet18 was used as encoder block and the decoder block follwed by original paper. Pre-trained model was deployed because of the small amount of data. Fine-tuning process was displayed visualizing in the wandb api dashboard. After tring the various combination of hyperparameter such as learning rate and weight decay etc. I used the set of parameter listed below as my hyperparamter setting. Include code snippets or references to notebooks where the model building process is documented.

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
