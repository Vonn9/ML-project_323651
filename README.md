# ML-project
# Court Dynamics

**Team-26 Member:**
* **Yunbo Feng:** 323651  
* **Kaiyang Zhang:** 317461  
* **Jingyu Bai:** 325101  
* **Jiayi Ye:** E06080    

## Section 1: Introduction

This project utilizes unsupervised learning from machine learning to analyze the historical performance data of NBA players. We use three different clustering methods (K-Means, DBSCAN, and Hierarchical Clustering) to analyze player metrics such as age, position, pts, and asts, and their evolution, with the aim of redefining player roles based on the results.
## Section 2: Methods 
### 2.1 Data Preprocessing
The basketball.db database contains 12 tables. The 'player_regular_season' table records detailed data for each player for every season, including keys like 'ilkid', 'year', 'team', 'gp', 'pts', 'reb', and 'asts', making it the most suitable source for analyzing player development and trajectory over time. The 'players' table contains 'birthdate', which can be combined with 'year' to calculate player age. Selecting other tables that focus on career totals would prevent the analysis of temporal changes.

Therefore, to analyze player development trajectories, we selected the 'player_regular_season' and 'players' tables, merged them using 'ilkid' as the unique key, and then performed subsequent data processing.
#### 2.1.1 Age Calculation
First, we converted 'birthdate' to a datetime type, converted invalid values to NaN, and extracted the birth year. We calculated the player's age using the difference between the season year and the birth year. We focused our study on players aged 18-45 to facilitate the subsequent analysis of player performance changes with age.
#### 2.1.2 Handling Missing and Outlier Values
First, missing values in the 12 relevant features were imputed with 0, signifying no activity or scoring for that metric. Then, data rows with missing age values were deleted, as an unknown age would directly impact subsequent analysis.
#### 2.1.3 Standardizing Data
Due to the significant variation in minutes played by NBA players, directly comparing raw stats can be unfair to certain players and distort the final data. Therefore, all feature data was standardized to a "Per 36 Minutes" basis to eliminate the time factor's influence: $value_{36mins} = value_{all} / minutes \times 36$. A value of 36 minutes is commonly used as an average benchmark in NBA data processing. 
#### 2.1.4 Data Normalization
Given the wide differences in units (magnitudes) across various features, all feature data was normalized using StandardScaler to ensure the data is on a comparable scale.
### 2.2 Feature Selection
- We used PCA to reduce the dimensionality of 12 scoring-related features ['pts', 'fgm', 'fga', 'ftm', 'fta', 'asts', 'turnover', 'reb', 'stl', 'tpa', 'blk', 'tpm']. The importance results for each feature are as follows:

  - pts: importance = 0.3628
  - fgm: importance = 0.3568
  - fga: importance = 0.3522
  - ftm: importance = 0.3453
  - fta: importance = 0.3443
  - asts: importance = 0.2894
  - turnover: importance = 0.2757
  - reb: importance = 0.2705
  - stl: importance = 0.2567
  - tpa: importance = 0.1671
  - blk: importance = 0.1665
  - tpm: importance = 0.1571

- We selected the top 9 core features with an absolute loading value greater than 0.2:

  - `pts` (Points)
  - `fgm` (Field Goals Made)
  - `fga` (Field Goals Attempted)
  - `ftm` (Free Throws Made)
  - `fta` (Free Throws Attempted)
  - `asts` (Assists)
  - `turnover` (Turnovers)
  - `reb` (Rebounds)
  - `stl` (Steals)

- Simultaneously, we used Principal Component Analysis (PCA) to reduce the feature space to 2 principal components for visualization of the model results:
  - Principal Component 1 (X-axis): Offensive features like points (pts), field goals made (fgm), and assists (asts) have high positive loadings on PC1, thus representing overall offensive ability.
  - Principal Component 2 (Y-axis): Three-point shooting features (tpa, tpm) show high positive loadings on PC2, while interior features like rebounds (reb) and blocks (blk) show negative loadings. This reflects player type differentiation.
### 2.3 Problem Definition and Model Selection
This problem should be defined as a clustering problem, for the following reasons: 
The problem does not require predicting specific values like wins/losses or scores, nor does it involve rigidly assigning players to predefined labels. Therefore, it is neither a classification nor a regression problem. Since the goal is to explore the intrinsic structure of performance patterns across multiple seasons without defined labels, an unsupervised method is appropriate for discovering patterns in the data rather than relying on existing labels for prediction.

We selected three clustering models based on different principles for comparison:
- K-Means
- DBSCAN
- Hierarchical Clustering
### 2.4 Project Flowchart

![Flowchart](images/Flowchart.png)
*Figure 1: Flowchart*

### 2.5 Environment Configuration
name: nba_clu
channels:
  - defaults
  - conda-forge
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free

dependencies:
  - python=3.9.21
  - sqlite=3.45.3   
  - pip=25.1
  - pip:
      - numpy==2.0.2
      - pandas==2.3.3
      - scikit-learn==1.6.1
      - scipy==1.13.1 
      - matplotlib==3.9.4
      - pillow==11.3.0

## Section 3: Experimental Design 

### 3.1 Main purpose
The experiment will use three different clustering methods (K-Means, DBSCAN, Hierarchical Clustering) and hyperparameter tuning to find the optimal model, analyze player statistics, and explore new player roles defined by on-court function, rather than relying on fixed position labels. 
### 3.2 Experimental Data and design
- **Data split**: The data will be divided using train_test_split into a training:validation:test ratio of 70:15:15.
- **Design**: For K-Means, experiments will be conducted for $k$ values from 2 to 8, and the optimal hyperparameter will be determined based on the Elbow Method and Silhouette Score; For DBSCAN, experiments will be conducted for eps values between 1.0 and 2.0, and the optimal hyperparameter will be determined based on the Silhouette Score; For Hierarchical Clustering, experiments will be conducted for the linkage strategies ('ward', 'average', 'complete') and $k$ values from 2 to 8, with the optimal hyperparameters determined by the Silhouette Score. All hyperparameter tuning will only be performed on the training set, followed by validation and comparison of different models on the validation set. Finally, the optimal model will be tested on the test set.
### 3.3 Baseline
The K-Means model with default parameters (K=8) is set as the Baseline.

### 3.3 Evaluation Metrics (评估指标)

- **Primary Metric**: Silhouette Score
- **Reason**: For an unsupervised learning task where true player labels are absent, the Silhouette Score measures the cohesion within clusters and the separation between clusters. It ranges from [-1, 1], with values closer to 1 indicating better results.


## Section 4: Results

### 4.1 EDA
We deeply explored the scoring efficiency of players at different positions by age and the distribution of assists per 36 minutes.
#### 4.1.1 Average Points Per 36 Minites by Age and Position
The data_clean was grouped by Age and position, and the mean of the points list pts_36 was calculated. The table was then converted to have Age as rows and position as columns. A line chart was drawn with Age on the x-axis and Points on the y-axis, with blue (C) representing Centers, brown (F) representing Forwards, and cyan (G) representing Guards, illustrating the trend of scoring changes with age.
![eda_age_trajectory](images/eda_age_trajectory.png)
*Figure 2: Average Points Per 36 Minites by Age and Position*
- **Age Trend Analysis**: As shown in Figure 2, most players lack experience and are technically immature before age 20, peak around 25-29, and then decline year-over-year after age 30, with a sharp drop-off. A brief peak around age 40 might be due to a few individual talented outliers, followed by a dramatic decline afterward.
- The scoring curve for Guards (G) is steeper, Centers (C) is "flatter," and Forwards (F) is the most stable. 
- Guards (G) score relatively low early on, possibly because younger guards often play bench or defensive roles. Their scoring rises quickly after age 22 but drops sharply after age 35 due to physical decline. 
- Centers (C) show overall less fluctuation, potentially because their game relies on height and positioning, which are relatively stable skills. Data is sparse and low around age 40 due to fewer playing opportunities. 
- Forwards (F) are relatively stable, possessing both offensive and defensive versatility, and maintain high efficiency for a long period between ages 25-35, making them the most durable position on the court. 
#### 4.1.2 Distribution of Assists by Position
For each 'position', the assists asts_36 were extracted to form the list data2. A boxplot was drawn with position on the x-axis and assists asts_36 on the y-axis, showing the assist distribution for different positions. 
![eda_position_distribution](images/eda_position_distribution.png)
*Figure 3: Distribution of Assists by Position*
- **Position Distribution Analysis**: As shown in Figure 3, the assist data distribution for Guards (G) is markedly higher than for Forwards (F) and Centers (C). The box is also taller with more outliers, indicating Guards have the most assists and greater internal variability.
- The box for Centers (C) is the lowest and narrowest, indicating the fewest assists, with almost no outliers. 
- The box for Forwards (F) is moderate, with the median between Centers and Guards, indicating medium assisting ability and considerable potential. 
### 4.2 Model Performance Comparison

After hyperparameter tuning, the performance of each model on the training set is shown in the table below:
- K-Means：
  
| K Value | Silhouette Score |
|------|----------|
| 2 | 0.4375 |
| **3** | **0.4401** |
| 4 | 0.3470 |
| 5 | 0.3366 |
| 6 | 0.3325 |
| 7 | 0.3431 |
| 8 | 0.3179 |
- DBSCAN：

| Eps    | Silhouette Score |
|--------|----------|
| 1.0000 | 0.1055 |
| 1.0526 | 0.0360 |
| 1.1053 | 0.0003 |
| 1.1579 | -0.0087 |
| 1.2105 | 0.0300 |
| 1.2632 | 0.0830 |
| 1.3158 | 0.0738 |
| 1.3684 | 0.1020 |
| 1.4211 | 0.0734 |
| 1.4737 | 0.2167 |
| 1.5263 | 0.3038 |
| 1.5789 | 0.2007 |
| **1.6316** | **0.4196** |
| 1.6842 | 0.2189 |
| 1.7368 | 0.2615 |
| 1.7895 | 0.2800 |
| 1.8421 | 0.2550 |
| 1.8947 | 0.2517 |
| 1.9474 | 0.2827 |
| 2.0000 | 0.3660 |
- Hierarchical Clustering：

| Linkage  | K Value | Silhouette Score |
|----------|------|----------|
| ward | 2 | 0.4237 |
| ward | 3 | 0.4320 |
| ward | 4 | 0.1998 |
| ward | 5 | 0.2145 |
| ward | 6 | 0.2218 |
| ward | 7 | 0.2083 |
| ward | 8 | 0.2091 |
| **average** | **2** | **0.6168** |
| average | 3 | 0.5379 |
| average | 4 | 0.5340 |
| average | 5 | 0.5272 |
| average | 6 | 0.5159 |
| average | 7 | 0.5140 |
| average | 8 | 0.4818 |
| complete | 2 | 0.5137 |
| complete | 3 | 0.4757 |
| complete | 4 | 0.3849 |
| complete | 5 | 0.3975 |
| complete | 6 | 0.3536 |
| complete | 7 | 0.3511 |
| complete | 8 | 0.3720 |

- After 5-Fold cross-validation tuning, the performance of the optimal version of each model on the validation set is shown below: 

| Model       | Best Param | Val Score | Findings |
|--------------------|--------------------------|----------------------------|----------|
| K-Means            | n_clusters=3           | 0.4600                     | Slightly better than the Baseline. Players are mainly divided into three categories, with a relatively balanced structure.  |
| DBSCAN             | eps=1.6316             | 0.4466                    | The worst performance. Most data was identified as noise or mixed together, preventing effective clustering.  |
| Hierarchical       | k=2, linkage='average'      | 0.6075                    | Highest score. Formed clusters with extremely high separation, but the cluster distribution is highly imbalanced.  |

- Finally, the optimal Hierarchical Clustering model achieved a score of 0.7518 on the test set and 0.6749 on the entire dataset. 

### 4.3 Clustering Results Visualization

#### 4.3.1 K-Means 


| Figure4：K-Means Hyperparameter Tuning| Figure5: K-Means Final Result (K=3) |
|--------------------|------------------------|
| ![model_kmeans_tuning](images/model_kmeans_tuning.png) | ![model_kmeans_result](images/model_kmeans_result.png) |
- **Visualization Analysis**: As shown in Figure 4, Inertia decreases as the K value increases, but the Silhouette Score peaks at K=3. 
- **Findings**: As shown in Figure 5, the horizontal axis PC1 represents "Offensive Load" or "Usage Rate," where a larger value means the player is more central. The vertical axis PC2 represents "Defense/Interior Attribute," where a larger value indicates a greater defensive or interior player tendency. K-Means divides the data on the PCA plane into three clear regions. The purple cluster, concentrated in the positive PC1 area and near-zero or negative PC2, likely represents perimeter scoring core players, who are ball-dominant and suitable as tactical cores. The blue cluster, situated mid-to-high PC1 and mid-PC2, likely represents versatile forwards or organizing interior players, who are adaptable and suited for multiple roles. The yellow cluster, located near-zero or negative PC1 and positive PC2, likely represents defensive center players, who are cost-effective, high-efficiency, and cornerstones of championship teams. 

#### 4.3.2 DBSCAN 
![model_dbscan_result](images/model_dbscan_result.png)
*Figure 6: Model Dbscan Result*
- **Visualization Analysis**: As shown in Figure 6, the horizontal and vertical axes are the first and second principal components after PCA dimensionality reduction. Blue circles are successfully clustered players, and red crosses represent points identified as noise. 
- **Findings**：A large number of data points were marked as noise, with no typical cluster centers existing. This is because NBA player abilities change continuously, lacking obvious high and low-density regions. Therefore, DBSCAN could only classify most players as "noise." 
#### 4.3.3 Hierarchical Clustering
![model_hierarchical_dendrogram](images/model_hierarchical_dendrogram0.png)
*Figure 7: Dendrogram*
- **Visualization Analysis**: The x-axis represents different players, and the y-axis represents the Euclidean distance. The dendrogram divides the players into 3 major categories. The blue cluster merges at a high distance (10–12), representing players with high variance; the orange and red clusters merge at the lowest distance (1-3), potentially representing very similar players; and the green cluster merges at a medium distance (4-8), also representing relatively similar players. 
- **Findings**: All players are grouped into two major camps only near the highest layer, representing the offensive and defensive sides. The large volume of green under the blue branch suggests multiple sub-types within the two major categories of offensive core players and role players, representing different roles. The small red and yellow branches connecting two very close points suggest that some players are statistically almost identical, possibly representing players with completely the same style. 
#### 4.3.4 Final Result Visualization
The optimal Hierarchical Clustering model was clustered and visualized on the entire dataset.
![model_hierarchical_result](images/model_hierarchical_result.png)
*Figure 8：Scatter Plot（ALL DATA）*

As shown in Figure 8, blue points account for over 95% of the points in the chart, representing the mainstream player population. The red points are sparse, concentrated in the bottom right and right edge, representing "unicorn" players with extremely high offensive load and high defensive propensity. This further illustrates that NBA data is highly continuous, and most players are "functional players," with only a few being "overperforming core players."
![model_hierarchical_dendrogram](images/model_hierarchical_dendrogram.png)
*Figure 9：Dendrogram（ALL DATA）*

As shown in Figure 9, all players are divided into two major camps only near the highest layer, representing the offensive and defensive sides. The large number of brown under the blue branch suggests multiple sub-types representing different roles within the two major categories of offensive core players and role players. The small orange, green, red, and purple branches at the bottom connect very close points, indicating that some players are statistically virtually identical.

## Section 5: Conclusions 

### 5.1 Take-away point
This project compared three algorithms and found that NBA player data is continuously distributed in high-dimensional space. Although Hierarchical Clustering achieved the best score, K-Means provided a more balanced group division. Meanwhile, DBSCAN demonstrated that NBA player data is very continuous in high-dimensional space, lacking clear boundaries, thus making distance-based forced division more suitable than density-based division in this scenario.
  
- The project successfully constructed an NBA player dynamic analysis framework centered on unsupervised learning clustering, moving beyond traditional position labels to deeply understand a player's functional role in offense and defense, their growth trajectory, and anomalous performance. This was achieved by integrating age trend analysis, multi-dimensional statistical clustering, and visualization verification.
- The essential difference in player value lies in "functional positioning," not "nominal position." The model divided players into three major categories: high offensive load, versatile, and low offensive load.
- The framework can not only identify a player's mainstream functional role but also detect anomaly signals: for example, some players' rare, all-around, or overperforming seasons are statistically scarce, and these players often possess a unique value that can change the course of a game.
- Furthermore, ages 27–29 represent a critical period in player development. Coaching staff can use this pattern to start assigning more offensive responsibilities to players around age 25 and gradually adjust their roles toward organizing or defending after age 32.

### 5.2 Limitations & Future Work

#### 5.2.1 Limitations:

- **Per 36 Bias**: For marginal players with extremely low minutes played, "Per 36" data can be abnormally magnified, becoming outliers that interfere with the clustering results. 
- **Quantity Limitation**: Our analysis is based on season-aggregated data and did not model the role migration path of individual players over time. 
- **Feature Limitation**: The feature system is limited to basic technical statistics, lacking descriptions of game impact, space utilization, or possession type, which restricts the fine-grained identification of complex roles. 

#### 5.2.2 Future Work:

- **Weighted Filtering**: Introduce weighted filtering to remove samples with insufficient total minutes played. 
- **Feature Enhancement**: Add advanced statistical features for a more granular division of player roles. 
- **Longitudinal Modeling**: Extend the framework to time series clustering to track changes in an individual player's cluster assignment over their career. 
- **Decision Interface**：Develop an interactive dashboard to allow coaches to filter players based on role requirements and directly translate the analysis results into team building strategies.


