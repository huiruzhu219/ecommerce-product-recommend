# 电商用户价值分层与精准营销策略分析

本 Notebook 基于 `user_personalized_features.csv` 完成：

- 传统 RFM 用户价值评分
- 优化 RFM-I 模型：新增意向度、转化摩擦、活跃连接度、购买力背景
- 用户分层画像与精准营销建议
- 传统 RFM 与优化 RFM 的 ROI 对比实验
- 可视化：柱状图、散点图、折线/柱状对比图、热力图



```python
!pip install seaborn
!pip install IPython
!pip install matplotlib
!pip install pandas
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display


try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

sns.set_theme(style="whitegrid")

# 中文字体设置：Windows/Jupyter 常见字体优先
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


def show_fig(fig=None):
    """PyCharm/Jupyter ?????????? show_fig() ???"""
    if fig is None:
        fig = plt.gcf()
    display(fig)
    plt.close(fig)

DATA_PATH = Path(r"D:/archive/user_personalized_features.csv")

# 如果网页 Jupyter 无法访问 D 盘，可以把 csv 上传到 Notebook 同目录，然后启用下面这行：
# DATA_PATH = Path("user_personalized_features.csv")

OUT_DIR = Path("outputs_jupyter")
OUT_DIR.mkdir(exist_ok=True)

DATA_PATH

```

    Collecting seaborn
      Downloading seaborn-0.13.2-py3-none-any.whl.metadata (5.4 kB)
    Requirement already satisfied: numpy!=1.24.0,>=1.20 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from seaborn) (2.0.2)
    Collecting pandas>=1.2 (from seaborn)
      Downloading pandas-2.3.3-cp39-cp39-win_amd64.whl.metadata (19 kB)
    Collecting matplotlib!=3.6.1,>=3.4 (from seaborn)
      Downloading matplotlib-3.9.4-cp39-cp39-win_amd64.whl.metadata (11 kB)
    Collecting contourpy>=1.0.1 (from matplotlib!=3.6.1,>=3.4->seaborn)
      Downloading contourpy-1.3.0-cp39-cp39-win_amd64.whl.metadata (5.4 kB)
    Collecting cycler>=0.10 (from matplotlib!=3.6.1,>=3.4->seaborn)
      Downloading cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
    Collecting fonttools>=4.22.0 (from matplotlib!=3.6.1,>=3.4->seaborn)
      Downloading fonttools-4.60.2-cp39-cp39-win_amd64.whl.metadata (115 kB)
    Collecting kiwisolver>=1.3.1 (from matplotlib!=3.6.1,>=3.4->seaborn)
      Downloading kiwisolver-1.4.7-cp39-cp39-win_amd64.whl.metadata (6.4 kB)
    Requirement already satisfied: packaging>=20.0 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (25.0)
    Requirement already satisfied: pillow>=8 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (10.4.0)
    Collecting pyparsing>=2.3.1 (from matplotlib!=3.6.1,>=3.4->seaborn)
      Downloading pyparsing-3.3.2-py3-none-any.whl.metadata (5.8 kB)
    Requirement already satisfied: python-dateutil>=2.7 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (2.9.0.post0)
    Requirement already satisfied: importlib-resources>=3.2.0 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from matplotlib!=3.6.1,>=3.4->seaborn) (6.5.2)
    Requirement already satisfied: zipp>=3.1.0 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from importlib-resources>=3.2.0->matplotlib!=3.6.1,>=3.4->seaborn) (3.21.0)
    Requirement already satisfied: pytz>=2020.1 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from pandas>=1.2->seaborn) (2025.2)
    Collecting tzdata>=2022.7 (from pandas>=1.2->seaborn)
      Downloading tzdata-2026.2-py2.py3-none-any.whl.metadata (1.4 kB)
    Requirement already satisfied: six>=1.5 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from python-dateutil>=2.7->matplotlib!=3.6.1,>=3.4->seaborn) (1.17.0)
    Downloading seaborn-0.13.2-py3-none-any.whl (294 kB)
    Downloading matplotlib-3.9.4-cp39-cp39-win_amd64.whl (7.8 MB)
       ---------------------------------------- 0.0/7.8 MB ? eta -:--:--
       - -------------------------------------- 0.3/7.8 MB ? eta -:--:--
       -- ------------------------------------- 0.5/7.8 MB 1.5 MB/s eta 0:00:05
       -- ------------------------------------- 0.5/7.8 MB 1.5 MB/s eta 0:00:05
       -- ------------------------------------- 0.5/7.8 MB 1.5 MB/s eta 0:00:05
       -- ------------------------------------- 0.5/7.8 MB 1.5 MB/s eta 0:00:05
       -- ------------------------------------- 0.5/7.8 MB 1.5 MB/s eta 0:00:05
       -- ------------------------------------- 0.5/7.8 MB 1.5 MB/s eta 0:00:05
       -- ------------------------------------- 0.5/7.8 MB 1.5 MB/s eta 0:00:05
       -- ------------------------------------- 0.5/7.8 MB 1.5 MB/s eta 0:00:05
       -- ------------------------------------- 0.5/7.8 MB 1.5 MB/s eta 0:00:05
       -- ------------------------------------- 0.5/7.8 MB 1.5 MB/s eta 0:00:05
       -- ------------------------------------- 0.5/7.8 MB 1.5 MB/s eta 0:00:05
       -- ------------------------------------- 0.5/7.8 MB 1.5 MB/s eta 0:00:05
       -- ------------------------------------- 0.5/7.8 MB 1.5 MB/s eta 0:00:05
       ------------- -------------------------- 2.6/7.8 MB 790.4 kB/s eta 0:00:07
       ------------- -------------------------- 2.6/7.8 MB 790.4 kB/s eta 0:00:07
       ---------------------------------------- 7.8/7.8 MB 2.2 MB/s eta 0:00:00
    Downloading contourpy-1.3.0-cp39-cp39-win_amd64.whl (211 kB)
    Downloading cycler-0.12.1-py3-none-any.whl (8.3 kB)
    Downloading fonttools-4.60.2-cp39-cp39-win_amd64.whl (1.5 MB)
       ---------------------------------------- 0.0/1.5 MB ? eta -:--:--
       -------------------- ------------------- 0.8/1.5 MB 3.0 MB/s eta 0:00:01
       --------------------------- ------------ 1.0/1.5 MB 3.0 MB/s eta 0:00:01
       --------------------------------- ------ 1.3/1.5 MB 2.4 MB/s eta 0:00:01
       ---------------------------------------- 1.5/1.5 MB 2.1 MB/s eta 0:00:00
    Downloading kiwisolver-1.4.7-cp39-cp39-win_amd64.whl (55 kB)
    Downloading pandas-2.3.3-cp39-cp39-win_amd64.whl (11.4 MB)
       ---------------------------------------- 0.0/11.4 MB ? eta -:--:--
        --------------------------------------- 0.3/11.4 MB ? eta -:--:--
        --------------------------------------- 0.3/11.4 MB ? eta -:--:--
        --------------------------------------- 0.3/11.4 MB ? eta -:--:--
        --------------------------------------- 0.3/11.4 MB ? eta -:--:--
        --------------------------------------- 0.3/11.4 MB ? eta -:--:--
        --------------------------------------- 0.3/11.4 MB ? eta -:--:--
        --------------------------------------- 0.3/11.4 MB ? eta -:--:--
        --------------------------------------- 0.3/11.4 MB ? eta -:--:--
        --------------------------------------- 0.3/11.4 MB ? eta -:--:--
        --------------------------------------- 0.3/11.4 MB ? eta -:--:--
        --------------------------------------- 0.3/11.4 MB ? eta -:--:--
        --------------------------------------- 0.3/11.4 MB ? eta -:--:--
        --------------------------------------- 0.3/11.4 MB ? eta -:--:--
       ------- -------------------------------- 2.1/11.4 MB 674.9 kB/s eta 0:00:14
       ------- -------------------------------- 2.1/11.4 MB 674.9 kB/s eta 0:00:14
       ------- -------------------------------- 2.1/11.4 MB 674.9 kB/s eta 0:00:14
       ------- -------------------------------- 2.1/11.4 MB 674.9 kB/s eta 0:00:14
       ------- -------------------------------- 2.1/11.4 MB 674.9 kB/s eta 0:00:14
       ------------------------ --------------- 6.8/11.4 MB 1.7 MB/s eta 0:00:03
       ------------------------- -------------- 7.3/11.4 MB 1.8 MB/s eta 0:00:03
       ------------------------- -------------- 7.3/11.4 MB 1.8 MB/s eta 0:00:03
       ------------------------- -------------- 7.3/11.4 MB 1.8 MB/s eta 0:00:03
       ------------------------- -------------- 7.3/11.4 MB 1.8 MB/s eta 0:00:03
       ------------------------- -------------- 7.3/11.4 MB 1.8 MB/s eta 0:00:03
       ------------------------- -------------- 7.3/11.4 MB 1.8 MB/s eta 0:00:03
       ------------------------- -------------- 7.3/11.4 MB 1.8 MB/s eta 0:00:03
       ------------------------- -------------- 7.3/11.4 MB 1.8 MB/s eta 0:00:03
       ------------------------- -------------- 7.3/11.4 MB 1.8 MB/s eta 0:00:03
       ------------------------- -------------- 7.3/11.4 MB 1.8 MB/s eta 0:00:03
       ------------------------- -------------- 7.3/11.4 MB 1.8 MB/s eta 0:00:03
       ------------------------------- -------- 8.9/11.4 MB 1.4 MB/s eta 0:00:02
       ------------------------------- -------- 8.9/11.4 MB 1.4 MB/s eta 0:00:02
       ---------------------------------------  11.3/11.4 MB 1.6 MB/s eta 0:00:01
       ---------------------------------------- 11.4/11.4 MB 1.6 MB/s eta 0:00:00
    Downloading pyparsing-3.3.2-py3-none-any.whl (122 kB)
    Downloading tzdata-2026.2-py2.py3-none-any.whl (349 kB)
    Installing collected packages: tzdata, pyparsing, kiwisolver, fonttools, cycler, contourpy, pandas, matplotlib, seaborn
    
       ---------------------------------------- 0/9 [tzdata]
       ---- ----------------------------------- 1/9 [pyparsing]
       ------------- -------------------------- 3/9 [fonttools]
       ------------- -------------------------- 3/9 [fonttools]
       ------------- -------------------------- 3/9 [fonttools]
       ------------- -------------------------- 3/9 [fonttools]
       ------------- -------------------------- 3/9 [fonttools]
       ------------- -------------------------- 3/9 [fonttools]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       -------------------------- ------------- 6/9 [pandas]
       ------------------------------- -------- 7/9 [matplotlib]
       ------------------------------- -------- 7/9 [matplotlib]
       ------------------------------- -------- 7/9 [matplotlib]
       ------------------------------- -------- 7/9 [matplotlib]
       ------------------------------- -------- 7/9 [matplotlib]
       ------------------------------- -------- 7/9 [matplotlib]
       ------------------------------- -------- 7/9 [matplotlib]
       ------------------------------- -------- 7/9 [matplotlib]
       ----------------------------------- ---- 8/9 [seaborn]
       ---------------------------------------- 9/9 [seaborn]
    
    Successfully installed contourpy-1.3.0 cycler-0.12.1 fonttools-4.60.2 kiwisolver-1.4.7 matplotlib-3.9.4 pandas-2.3.3 pyparsing-3.3.2 seaborn-0.13.2 tzdata-2026.2
    Requirement already satisfied: IPython in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (8.18.1)
    Requirement already satisfied: decorator in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from IPython) (5.2.1)
    Requirement already satisfied: jedi>=0.16 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from IPython) (0.19.2)
    Requirement already satisfied: matplotlib-inline in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from IPython) (0.1.7)
    Requirement already satisfied: prompt-toolkit<3.1.0,>=3.0.41 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from IPython) (3.0.51)
    Requirement already satisfied: pygments>=2.4.0 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from IPython) (2.19.1)
    Requirement already satisfied: stack-data in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from IPython) (0.6.3)
    Requirement already satisfied: traitlets>=5 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from IPython) (5.14.3)
    Requirement already satisfied: typing-extensions in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from IPython) (4.13.2)
    Requirement already satisfied: exceptiongroup in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from IPython) (1.2.2)
    Requirement already satisfied: colorama in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from IPython) (0.4.6)
    Requirement already satisfied: wcwidth in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from prompt-toolkit<3.1.0,>=3.0.41->IPython) (0.2.13)
    Requirement already satisfied: parso<0.9.0,>=0.8.4 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from jedi>=0.16->IPython) (0.8.4)
    Requirement already satisfied: executing>=1.2.0 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from stack-data->IPython) (2.2.0)
    Requirement already satisfied: asttokens>=2.1.0 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from stack-data->IPython) (3.0.0)
    Requirement already satisfied: pure_eval in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from stack-data->IPython) (0.2.3)
    Requirement already satisfied: matplotlib in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (3.9.4)
    Requirement already satisfied: contourpy>=1.0.1 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from matplotlib) (1.3.0)
    Requirement already satisfied: cycler>=0.10 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from matplotlib) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from matplotlib) (4.60.2)
    Requirement already satisfied: kiwisolver>=1.3.1 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from matplotlib) (1.4.7)
    Requirement already satisfied: numpy>=1.23 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from matplotlib) (2.0.2)
    Requirement already satisfied: packaging>=20.0 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from matplotlib) (25.0)
    Requirement already satisfied: pillow>=8 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from matplotlib) (10.4.0)
    Requirement already satisfied: pyparsing>=2.3.1 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from matplotlib) (3.3.2)
    Requirement already satisfied: python-dateutil>=2.7 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from matplotlib) (2.9.0.post0)
    Requirement already satisfied: importlib-resources>=3.2.0 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from matplotlib) (6.5.2)
    Requirement already satisfied: zipp>=3.1.0 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from importlib-resources>=3.2.0->matplotlib) (3.21.0)
    Requirement already satisfied: six>=1.5 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from python-dateutil>=2.7->matplotlib) (1.17.0)
    Requirement already satisfied: pandas in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (2.3.3)
    Requirement already satisfied: numpy>=1.22.4 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from pandas) (2.0.2)
    Requirement already satisfied: python-dateutil>=2.8.2 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from pandas) (2.9.0.post0)
    Requirement already satisfied: pytz>=2020.1 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from pandas) (2025.2)
    Requirement already satisfied: tzdata>=2022.7 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from pandas) (2026.2)
    Requirement already satisfied: six>=1.5 in d:\anaconda3\envs\cup_2.5.0\lib\site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)
    




    WindowsPath('D:/archive/user_personalized_features.csv')



## 1. 读取数据与基础检查



```python
df_raw = pd.read_csv(DATA_PATH)
df_raw = df_raw.loc[:, ~df_raw.columns.str.contains(r"^Unnamed|^$", regex=True)]

print("数据规模：", df_raw.shape)
display(df_raw.head())
display(df_raw.describe(include="all").T)

```

    数据规模： (1000, 14)
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>Age</th>
      <th>Gender</th>
      <th>Location</th>
      <th>Income</th>
      <th>Interests</th>
      <th>Last_Login_Days_Ago</th>
      <th>Purchase_Frequency</th>
      <th>Average_Order_Value</th>
      <th>Total_Spending</th>
      <th>Product_Category_Preference</th>
      <th>Time_Spent_on_Site_Minutes</th>
      <th>Pages_Viewed</th>
      <th>Newsletter_Subscription</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#1</td>
      <td>56</td>
      <td>Male</td>
      <td>Suburban</td>
      <td>38037</td>
      <td>Sports</td>
      <td>5</td>
      <td>7</td>
      <td>18</td>
      <td>2546</td>
      <td>Books</td>
      <td>584</td>
      <td>38</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#2</td>
      <td>46</td>
      <td>Female</td>
      <td>Rural</td>
      <td>103986</td>
      <td>Technology</td>
      <td>15</td>
      <td>7</td>
      <td>118</td>
      <td>320</td>
      <td>Electronics</td>
      <td>432</td>
      <td>40</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#3</td>
      <td>32</td>
      <td>Female</td>
      <td>Suburban</td>
      <td>101942</td>
      <td>Sports</td>
      <td>28</td>
      <td>1</td>
      <td>146</td>
      <td>3766</td>
      <td>Apparel</td>
      <td>306</td>
      <td>1</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#4</td>
      <td>60</td>
      <td>Female</td>
      <td>Suburban</td>
      <td>71612</td>
      <td>Fashion</td>
      <td>18</td>
      <td>3</td>
      <td>163</td>
      <td>4377</td>
      <td>Apparel</td>
      <td>527</td>
      <td>29</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#5</td>
      <td>25</td>
      <td>Male</td>
      <td>Suburban</td>
      <td>49725</td>
      <td>Travel</td>
      <td>2</td>
      <td>5</td>
      <td>141</td>
      <td>4502</td>
      <td>Health &amp; Beauty</td>
      <td>53</td>
      <td>10</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>User_ID</th>
      <td>1000</td>
      <td>1000</td>
      <td>#1000</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>1000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>40.986</td>
      <td>13.497852</td>
      <td>18.0</td>
      <td>29.0</td>
      <td>42.0</td>
      <td>52.0</td>
      <td>64.0</td>
    </tr>
    <tr>
      <th>Gender</th>
      <td>1000</td>
      <td>2</td>
      <td>Male</td>
      <td>526</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Location</th>
      <td>1000</td>
      <td>3</td>
      <td>Suburban</td>
      <td>349</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Income</th>
      <td>1000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>81304.732</td>
      <td>37363.972753</td>
      <td>20155.0</td>
      <td>48715.5</td>
      <td>81042.0</td>
      <td>112694.5</td>
      <td>149951.0</td>
    </tr>
    <tr>
      <th>Interests</th>
      <td>1000</td>
      <td>5</td>
      <td>Sports</td>
      <td>213</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Last_Login_Days_Ago</th>
      <td>1000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>15.586</td>
      <td>8.205604</td>
      <td>1.0</td>
      <td>8.0</td>
      <td>16.0</td>
      <td>23.0</td>
      <td>29.0</td>
    </tr>
    <tr>
      <th>Purchase_Frequency</th>
      <td>1000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.631</td>
      <td>2.837762</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>7.0</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>Average_Order_Value</th>
      <td>1000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>104.039</td>
      <td>54.873569</td>
      <td>10.0</td>
      <td>58.0</td>
      <td>105.0</td>
      <td>150.0</td>
      <td>199.0</td>
    </tr>
    <tr>
      <th>Total_Spending</th>
      <td>1000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2552.957</td>
      <td>1420.985048</td>
      <td>112.0</td>
      <td>1271.75</td>
      <td>2542.0</td>
      <td>3835.5</td>
      <td>4999.0</td>
    </tr>
    <tr>
      <th>Product_Category_Preference</th>
      <td>1000</td>
      <td>5</td>
      <td>Apparel</td>
      <td>218</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Time_Spent_on_Site_Minutes</th>
      <td>1000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>297.36</td>
      <td>175.596123</td>
      <td>2.0</td>
      <td>144.0</td>
      <td>292.5</td>
      <td>449.25</td>
      <td>599.0</td>
    </tr>
    <tr>
      <th>Pages_Viewed</th>
      <td>1000.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>24.403</td>
      <td>14.02017</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>24.5</td>
      <td>36.0</td>
      <td>49.0</td>
    </tr>
    <tr>
      <th>Newsletter_Subscription</th>
      <td>1000</td>
      <td>2</td>
      <td>True</td>
      <td>507</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



```python
df = df_raw.copy()

bool_map = {"True": True, "False": False, True: True, False: False}
df["Newsletter_Subscription"] = df["Newsletter_Subscription"].map(bool_map).astype(bool)

numeric_cols = [
    "Age",
    "Income",
    "Last_Login_Days_Ago",
    "Purchase_Frequency",
    "Average_Order_Value",
    "Total_Spending",
    "Time_Spent_on_Site_Minutes",
    "Pages_Viewed",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols).reset_index(drop=True)

print("清洗后数据规模：", df.shape)
print("缺失值：")
display(df.isna().sum())

```

    清洗后数据规模： (1000, 14)
    缺失值：
    


    User_ID                        0
    Age                            0
    Gender                         0
    Location                       0
    Income                         0
    Interests                      0
    Last_Login_Days_Ago            0
    Purchase_Frequency             0
    Average_Order_Value            0
    Total_Spending                 0
    Product_Category_Preference    0
    Time_Spent_on_Site_Minutes     0
    Pages_Viewed                   0
    Newsletter_Subscription        0
    dtype: int64


## 2. 构建传统 RFM 与优化 RFM-I 指标

传统 RFM：

- R：最近登录越近越高
- F：购买频次越高越高
- M：总消费越高越高

新增因子：

- **意向度 I-score** = 停留时长分位分 * 0.5 + 浏览页数分位分 * 0.5
- **转化摩擦系数** = 浏览页数 / (购买次数 + 1)
- **活跃连接度** = 订阅状态 + 最近登录天数
- **购买力背景** = 收入分层 + 消费/收入比例，只做标签，不进入主评分



```python
def pct_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """返回 0-100 分位分。"""
    rank = series.rank(method="average", pct=True, ascending=True)
    if higher_is_better:
        return rank * 100
    return (1 - rank) * 100


def tier_by_quantile(series: pd.Series, labels: list[str]) -> pd.Series:
    return pd.qcut(series.rank(method="first"), q=len(labels), labels=labels)


def active_connection(row: pd.Series) -> str:
    subscribed = bool(row["Newsletter_Subscription"])
    days = row["Last_Login_Days_Ago"]
    if subscribed and days <= 7:
        return "高连接-订阅且7天内登录"
    if (not subscribed) and days <= 7:
        return "中连接-未订阅但7天内登录"
    if (subscribed and days >= 30) or days > 30:
        return "低连接-名义忠诚/不活跃"
    if subscribed:
        return "名义忠诚-需激活"
    return "弱连接-未订阅且近期未登录"


# 传统 RFM
df["R_Score"] = pct_score(df["Last_Login_Days_Ago"], higher_is_better=False)
df["F_Score"] = pct_score(df["Purchase_Frequency"], higher_is_better=True)
df["M_Score"] = pct_score(df["Total_Spending"], higher_is_better=True)
df["RFM_Score"] = 0.35 * df["R_Score"] + 0.30 * df["F_Score"] + 0.35 * df["M_Score"]

# 新增：意向度
df["Time_Score"] = pct_score(df["Time_Spent_on_Site_Minutes"], higher_is_better=True)
df["Pages_Score"] = pct_score(df["Pages_Viewed"], higher_is_better=True)
df["I_Score"] = 0.5 * df["Time_Score"] + 0.5 * df["Pages_Score"]

# 新增：转化摩擦
df["Conversion_Friction"] = df["Pages_Viewed"] / (df["Purchase_Frequency"] + 1)
df["Friction_Score"] = pct_score(df["Conversion_Friction"], higher_is_better=True)

# 新增：活跃连接度
df["Active_Connection"] = df.apply(active_connection, axis=1)

# 新增：购买力背景，只做标签
df["Income_Pct"] = pct_score(df["Income"], higher_is_better=True)
df["Income_Tier"] = tier_by_quantile(df["Income"], ["低收入", "中低收入", "中高收入", "高收入"])
df["Spend_Income_Ratio"] = df["Total_Spending"] / df["Income"]
df["Spend_Burden_Tier"] = tier_by_quantile(
    df["Spend_Income_Ratio"],
    ["低负担消费", "适中负担消费", "高负担消费", "超高负担消费"],
)
df["Purchasing_Power_Tag"] = df["Income_Tier"].astype(str) + "/" + df["Spend_Burden_Tier"].astype(str)

# 增强价值分
active_adj = df["Active_Connection"].map(
    {
        "高连接-订阅且7天内登录": 8,
        "中连接-未订阅但7天内登录": 5,
        "名义忠诚-需激活": 1,
        "弱连接-未订阅且近期未登录": -3,
        "低连接-名义忠诚/不活跃": -8,
    }
).fillna(0)

friction_adj = np.where(df["Friction_Score"] >= 75, -5, np.where(df["Friction_Score"] <= 25, 4, 0))

df["Enhanced_Value_Score"] = np.clip(
    0.55 * df["RFM_Score"]
    + 0.25 * df["I_Score"]
    + 0.20 * (100 - df["Friction_Score"])
    + active_adj
    + friction_adj,
    0,
    100,
)

display(
    df[
        [
            "User_ID",
            "RFM_Score",
            "I_Score",
            "Conversion_Friction",
            "Friction_Score",
            "Active_Connection",
            "Purchasing_Power_Tag",
            "Enhanced_Value_Score",
        ]
    ].head()
)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>User_ID</th>
      <th>RFM_Score</th>
      <th>I_Score</th>
      <th>Conversion_Friction</th>
      <th>Friction_Score</th>
      <th>Active_Connection</th>
      <th>Purchasing_Power_Tag</th>
      <th>Enhanced_Value_Score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>#1</td>
      <td>70.3550</td>
      <td>88.150</td>
      <td>4.750000</td>
      <td>55.55</td>
      <td>高连接-订阅且7天内登录</td>
      <td>低收入/超高负担消费</td>
      <td>77.622750</td>
    </tr>
    <tr>
      <th>1</th>
      <td>#2</td>
      <td>42.1450</td>
      <td>76.600</td>
      <td>5.000000</td>
      <td>58.55</td>
      <td>弱连接-未订阅且近期未登录</td>
      <td>中高收入/低负担消费</td>
      <td>47.619750</td>
    </tr>
    <tr>
      <th>2</th>
      <td>#3</td>
      <td>31.8500</td>
      <td>26.300</td>
      <td>0.500000</td>
      <td>4.30</td>
      <td>名义忠诚-需激活</td>
      <td>中高收入/高负担消费</td>
      <td>48.232500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>#4</td>
      <td>55.3525</td>
      <td>73.550</td>
      <td>7.250000</td>
      <td>72.25</td>
      <td>弱连接-未订阅且近期未登录</td>
      <td>中低收入/超高负担消费</td>
      <td>51.381375</td>
    </tr>
    <tr>
      <th>4</th>
      <td>#5</td>
      <td>81.3100</td>
      <td>14.775</td>
      <td>1.666667</td>
      <td>18.90</td>
      <td>高连接-订阅且7天内登录</td>
      <td>中低收入/超高负担消费</td>
      <td>76.634250</td>
    </tr>
  </tbody>
</table>
</div>


## 3. 用户分层规则



```python
def segment_user(row: pd.Series) -> str:
    high_income = row["Income_Pct"] >= 75
    low_income = row["Income_Pct"] <= 40
    high_spend = row["M_Score"] >= 70
    low_spend = row["M_Score"] <= 45
    active_recent = row["Last_Login_Days_Ago"] <= 7
    inactive = row["R_Score"] <= 35
    high_intent = row["I_Score"] >= 70
    very_high_intent = row["I_Score"] >= 80
    high_friction = row["Friction_Score"] >= 70
    low_purchase = row["Purchase_Frequency"] <= df["Purchase_Frequency"].median()

    if high_income and high_spend and inactive:
        return "高潜流失客"
    if high_income and high_intent and low_spend and high_friction:
        return "纠结土豪"
    if very_high_intent and active_recent and low_purchase and low_spend:
        return "高潜观望者"
    if high_intent and low_income and low_spend and row["Last_Login_Days_Ago"] <= 15:
        return "隐形活跃者"
    if row["Enhanced_Value_Score"] >= 78 and row["R_Score"] >= 45:
        return "核心用户"
    if row["Enhanced_Value_Score"] >= 62 or (high_intent and active_recent):
        return "潜力用户"
    if row["R_Score"] <= 30 and row["RFM_Score"] >= 55:
        return "沉睡价值用户"
    if row["RFM_Score"] <= 35 and row["I_Score"] <= 45:
        return "低价值用户"
    return "一般用户"


df["Enhanced_Segment"] = df.apply(segment_user, axis=1)

segment_summary = (
    df.groupby("Enhanced_Segment")
    .agg(
        用户数=("User_ID", "count"),
        平均消费=("Total_Spending", "mean"),
        平均客单价=("Average_Order_Value", "mean"),
        平均收入=("Income", "mean"),
        平均意向分=("I_Score", "mean"),
        平均摩擦分=("Friction_Score", "mean"),
        平均增强价值分=("Enhanced_Value_Score", "mean"),
        最近登录天数=("Last_Login_Days_Ago", "mean"),
    )
    .sort_values("用户数", ascending=False)
    .round(2)
)

display(segment_summary)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>用户数</th>
      <th>平均消费</th>
      <th>平均客单价</th>
      <th>平均收入</th>
      <th>平均意向分</th>
      <th>平均摩擦分</th>
      <th>平均增强价值分</th>
      <th>最近登录天数</th>
    </tr>
    <tr>
      <th>Enhanced_Segment</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>一般用户</th>
      <td>627</td>
      <td>2393.73</td>
      <td>104.35</td>
      <td>79697.40</td>
      <td>50.40</td>
      <td>56.03</td>
      <td>45.47</td>
      <td>16.68</td>
    </tr>
    <tr>
      <th>潜力用户</th>
      <td>187</td>
      <td>3194.65</td>
      <td>104.34</td>
      <td>83704.19</td>
      <td>51.94</td>
      <td>33.76</td>
      <td>67.51</td>
      <td>9.84</td>
    </tr>
    <tr>
      <th>低价值用户</th>
      <td>68</td>
      <td>1172.76</td>
      <td>98.54</td>
      <td>83971.90</td>
      <td>27.89</td>
      <td>42.65</td>
      <td>32.47</td>
      <td>22.84</td>
    </tr>
    <tr>
      <th>核心用户</th>
      <td>34</td>
      <td>3643.88</td>
      <td>98.38</td>
      <td>72581.03</td>
      <td>53.63</td>
      <td>27.96</td>
      <td>81.41</td>
      <td>4.29</td>
    </tr>
    <tr>
      <th>高潜流失客</th>
      <td>28</td>
      <td>4273.89</td>
      <td>114.68</td>
      <td>126868.14</td>
      <td>44.54</td>
      <td>44.58</td>
      <td>50.18</td>
      <td>24.21</td>
    </tr>
    <tr>
      <th>沉睡价值用户</th>
      <td>23</td>
      <td>4102.83</td>
      <td>110.04</td>
      <td>58787.61</td>
      <td>45.31</td>
      <td>35.25</td>
      <td>55.69</td>
      <td>24.57</td>
    </tr>
    <tr>
      <th>隐形活跃者</th>
      <td>16</td>
      <td>1062.31</td>
      <td>105.44</td>
      <td>42306.06</td>
      <td>78.80</td>
      <td>76.18</td>
      <td>48.56</td>
      <td>8.88</td>
    </tr>
    <tr>
      <th>纠结土豪</th>
      <td>12</td>
      <td>1212.33</td>
      <td>93.42</td>
      <td>133819.00</td>
      <td>84.63</td>
      <td>87.26</td>
      <td>39.14</td>
      <td>14.50</td>
    </tr>
    <tr>
      <th>高潜观望者</th>
      <td>5</td>
      <td>1094.40</td>
      <td>101.00</td>
      <td>63358.00</td>
      <td>90.22</td>
      <td>85.93</td>
      <td>53.12</td>
      <td>5.40</td>
    </tr>
  </tbody>
</table>
</div>


## 4. 可视化一：分层用户数量柱状图



```python
plt.figure(figsize=(12, 6))
order = segment_summary.index.tolist()
ax = sns.barplot(
    data=segment_summary.reset_index(),
    x="Enhanced_Segment",
    y="用户数",
    order=order,
    palette="Set2",
)
for p in ax.patches:
    ax.annotate(
        f"{int(p.get_height())}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom",
        fontsize=10,
    )
plt.title("更新 RFM-I 模型用户分层规模", fontsize=16, fontweight="bold")
plt.xlabel("用户分层")
plt.ylabel("用户数")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
show_fig()

```


    
![png](ecommerce_rfm_analysis_notebook_pycharm_fixed_files/ecommerce_rfm_analysis_notebook_pycharm_fixed_10_0.png)
    


## 5. 可视化二：各分层平均消费、收入、意向与摩擦对比



```python
metric_df = segment_summary.reset_index().melt(
    id_vars="Enhanced_Segment",
    value_vars=["平均消费", "平均收入", "平均意向分", "平均摩擦分", "平均增强价值分", "最近登录天数"],
    var_name="指标",
    value_name="数值",
)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for ax, metric in zip(axes, metric_df["指标"].unique()):
    data = metric_df[metric_df["指标"] == metric]
    sns.barplot(data=data, x="Enhanced_Segment", y="数值", ax=ax, palette="Blues_r")
    ax.set_title(metric)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=45)

plt.suptitle("不同用户分层的核心指标对比", fontsize=18, fontweight="bold", y=1.02)
plt.tight_layout()
show_fig()

```


    
![png](ecommerce_rfm_analysis_notebook_pycharm_fixed_files/ecommerce_rfm_analysis_notebook_pycharm_fixed_12_0.png)
    


## 6. 可视化三：意向度 vs 消费金额散点图

这张图可以识别：

- 高意向低消费：潜在转化对象
- 高消费低意向：可能已经满足或即将流失
- 高意向高消费：核心经营对象



```python
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df,
    x="I_Score",
    y="Total_Spending",
    hue="Enhanced_Segment",
    size="Income",
    sizes=(30, 260),
    alpha=0.75,
)
plt.axvline(70, color="red", linestyle="--", linewidth=1, label="高意向阈值")
plt.axhline(df["Total_Spending"].quantile(0.70), color="gray", linestyle="--", linewidth=1, label="高消费阈值")
plt.title("用户意向度 vs 总消费：识别高意向未转化人群", fontsize=16, fontweight="bold")
plt.xlabel("I-score 意向度")
plt.ylabel("Total Spending 总消费")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
show_fig()

```


    
![png](ecommerce_rfm_analysis_notebook_pycharm_fixed_files/ecommerce_rfm_analysis_notebook_pycharm_fixed_14_0.png)
    


## 7. 可视化四：转化摩擦 vs 意向度散点图

右上角代表：意向高、摩擦高，即“想买但纠结”。



```python
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df,
    x="I_Score",
    y="Friction_Score",
    hue="Enhanced_Segment",
    size="Purchase_Frequency",
    sizes=(30, 220),
    alpha=0.75,
)
plt.axvline(70, color="red", linestyle="--", linewidth=1)
plt.axhline(70, color="red", linestyle="--", linewidth=1)
plt.text(73, 92, "高意向 + 高摩擦\n重点转化区", color="red", fontsize=12)
plt.title("意向度 vs 转化摩擦：识别纠结土豪与高潜观望者", fontsize=16, fontweight="bold")
plt.xlabel("I-score 意向度")
plt.ylabel("Friction Score 转化摩擦分")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
show_fig()

```


    
![png](ecommerce_rfm_analysis_notebook_pycharm_fixed_files/ecommerce_rfm_analysis_notebook_pycharm_fixed_16_0.png)
    


## 8. 可视化五：RFM 分数与增强价值分对比

如果用户 RFM 不高但增强价值分高，说明传统模型可能低估了该用户。



```python
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df,
    x="RFM_Score",
    y="Enhanced_Value_Score",
    hue="Enhanced_Segment",
    alpha=0.75,
)
plt.plot([0, 100], [0, 100], color="gray", linestyle="--", label="RFM=增强价值")
plt.title("传统 RFM 分数 vs 增强价值分", fontsize=16, fontweight="bold")
plt.xlabel("传统 RFM Score")
plt.ylabel("Enhanced Value Score")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
show_fig()

```


    
![png](ecommerce_rfm_analysis_notebook_pycharm_fixed_files/ecommerce_rfm_analysis_notebook_pycharm_fixed_18_0.png)
    


## 9. 可视化六：购买力背景差异

同样消费金额，对不同收入用户意义不同。消费/收入比例越高，代表该消费对用户越“重”。



```python
plt.figure(figsize=(12, 7))
sns.scatterplot(
    data=df,
    x="Income",
    y="Total_Spending",
    hue="Spend_Burden_Tier",
    size="Spend_Income_Ratio",
    sizes=(30, 260),
    alpha=0.75,
)
plt.title("收入 vs 消费金额：识别购买力背景差异", fontsize=16, fontweight="bold")
plt.xlabel("Income 收入")
plt.ylabel("Total Spending 总消费")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
show_fig()

```


    
![png](ecommerce_rfm_analysis_notebook_pycharm_fixed_files/ecommerce_rfm_analysis_notebook_pycharm_fixed_20_0.png)
    



```python
plt.figure(figsize=(13, 6))
income_spend = (
    df.groupby(["Income_Tier", "Spend_Burden_Tier"])
    .size()
    .reset_index(name="用户数")
)
pivot = income_spend.pivot(index="Income_Tier", columns="Spend_Burden_Tier", values="用户数").fillna(0)
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("收入分层 × 消费负担：购买力背景热力图", fontsize=16, fontweight="bold")
plt.xlabel("消费负担")
plt.ylabel("收入分层")
plt.tight_layout()
show_fig()

```


    
![png](ecommerce_rfm_analysis_notebook_pycharm_fixed_files/ecommerce_rfm_analysis_notebook_pycharm_fixed_21_0.png)
    


## 10. 分层营销建议



```python
SEGMENT_STRATEGY = {
    "核心用户": "会员等级、专属新品、加价购、组合套装和复购周期提醒，提高客单价和复购频次。",
    "潜力用户": "围绕最近偏好类目做精准推荐，用阶梯满减和连续购买奖励把偶发消费变成习惯消费。",
    "高潜流失客": "优先召回并诊断流失原因，提供专属客服、个性化挽留券、新品优先体验。",
    "纠结土豪": "先降低决策成本：个性化榜单、同价位对比、专家推荐、套装方案，再给小额限时券。",
    "高潜观望者": "用低门槛首单券、足迹召回、购物车提醒和爆品推荐，促成首次或下一次购买。",
    "隐形活跃者": "鼓励分享裂变、评价晒单、内容互动；消费侧用低价爆品和拼团提升频次。",
    "沉睡价值用户": "轻触达测试兴趣，再根据点击和回访行为分层加码，避免一开始就高补贴。",
    "一般用户": "自动化运营和二次细分，用推荐、签到、常规促销培养偏好。",
    "低价值用户": "控制预算，低成本触达；只有出现高意向行为后再升级运营。",
}

strategy_table = segment_summary.reset_index()[["Enhanced_Segment", "用户数", "平均消费", "平均收入", "平均意向分", "平均摩擦分"]].copy()
strategy_table["营销建议"] = strategy_table["Enhanced_Segment"].map(SEGMENT_STRATEGY)
strategy_table = strategy_table.rename(columns={"Enhanced_Segment": "用户分层"})

display(strategy_table)

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>用户分层</th>
      <th>用户数</th>
      <th>平均消费</th>
      <th>平均收入</th>
      <th>平均意向分</th>
      <th>平均摩擦分</th>
      <th>营销建议</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>一般用户</td>
      <td>627</td>
      <td>2393.73</td>
      <td>79697.40</td>
      <td>50.40</td>
      <td>56.03</td>
      <td>自动化运营和二次细分，用推荐、签到、常规促销培养偏好。</td>
    </tr>
    <tr>
      <th>1</th>
      <td>潜力用户</td>
      <td>187</td>
      <td>3194.65</td>
      <td>83704.19</td>
      <td>51.94</td>
      <td>33.76</td>
      <td>围绕最近偏好类目做精准推荐，用阶梯满减和连续购买奖励把偶发消费变成习惯消费。</td>
    </tr>
    <tr>
      <th>2</th>
      <td>低价值用户</td>
      <td>68</td>
      <td>1172.76</td>
      <td>83971.90</td>
      <td>27.89</td>
      <td>42.65</td>
      <td>控制预算，低成本触达；只有出现高意向行为后再升级运营。</td>
    </tr>
    <tr>
      <th>3</th>
      <td>核心用户</td>
      <td>34</td>
      <td>3643.88</td>
      <td>72581.03</td>
      <td>53.63</td>
      <td>27.96</td>
      <td>会员等级、专属新品、加价购、组合套装和复购周期提醒，提高客单价和复购频次。</td>
    </tr>
    <tr>
      <th>4</th>
      <td>高潜流失客</td>
      <td>28</td>
      <td>4273.89</td>
      <td>126868.14</td>
      <td>44.54</td>
      <td>44.58</td>
      <td>优先召回并诊断流失原因，提供专属客服、个性化挽留券、新品优先体验。</td>
    </tr>
    <tr>
      <th>5</th>
      <td>沉睡价值用户</td>
      <td>23</td>
      <td>4102.83</td>
      <td>58787.61</td>
      <td>45.31</td>
      <td>35.25</td>
      <td>轻触达测试兴趣，再根据点击和回访行为分层加码，避免一开始就高补贴。</td>
    </tr>
    <tr>
      <th>6</th>
      <td>隐形活跃者</td>
      <td>16</td>
      <td>1062.31</td>
      <td>42306.06</td>
      <td>78.80</td>
      <td>76.18</td>
      <td>鼓励分享裂变、评价晒单、内容互动；消费侧用低价爆品和拼团提升频次。</td>
    </tr>
    <tr>
      <th>7</th>
      <td>纠结土豪</td>
      <td>12</td>
      <td>1212.33</td>
      <td>133819.00</td>
      <td>84.63</td>
      <td>87.26</td>
      <td>先降低决策成本：个性化榜单、同价位对比、专家推荐、套装方案，再给小额限时券。</td>
    </tr>
    <tr>
      <th>8</th>
      <td>高潜观望者</td>
      <td>5</td>
      <td>1094.40</td>
      <td>63358.00</td>
      <td>90.22</td>
      <td>85.93</td>
      <td>用低门槛首单券、足迹召回、购物车提醒和爆品推荐，促成首次或下一次购买。</td>
    </tr>
  </tbody>
</table>
</div>


## 11. ROI 对比实验

实验设计：

- 传统 RFM 策略：给 RFM_Score 前 20% 用户发券
- 优化 RFM-I 策略：给核心用户、潜力用户、纠结土豪、高潜观望者、高潜流失客发券
- 两组触达人数一致，券面预算一致
- 原始数据没有真实营销曝光与转化结果，因此这里用行为代理模型估算预期 ROI



```python
def roi_simulation(data: pd.DataFrame, coupon_value: float = 10.0, gross_margin: float = 0.35):
    target_n = max(1, int(np.floor(len(data) * 0.20)))
    budget = target_n * coupon_value

    active_bonus = data["Active_Connection"].map(
        {
            "高连接-订阅且7天内登录": 0.35,
            "中连接-未订阅但7天内登录": 0.25,
            "名义忠诚-需激活": 0.05,
            "弱连接-未订阅且近期未登录": -0.10,
            "低连接-名义忠诚/不活跃": -0.25,
        }
    ).fillna(0)

    segment_bonus = data["Enhanced_Segment"].map(
        {
            "纠结土豪": 0.55,
            "高潜观望者": 0.45,
            "高潜流失客": 0.40,
            "潜力用户": 0.25,
            "核心用户": 0.15,
            "隐形活跃者": 0.10,
        }
    ).fillna(0)

    affordability = np.clip(data["Income_Pct"] / 100, 0.05, 1.0)
    friction_penalty = np.clip(1 - data["Friction_Score"] / 180, 0.35, 1)

    logit = (
        -3.2
        + 0.018 * data["I_Score"]
        + 0.010 * data["F_Score"]
        + 0.008 * data["M_Score"]
        + active_bonus
        + 0.25 * affordability
    )
    base_prob = 1 / (1 + np.exp(-logit))

    coupon_uplift = np.clip(
        0.035
        * (0.55 + data["I_Score"] / 100)
        * friction_penalty
        * (0.75 + affordability)
        + segment_bonus / 20,
        0.005,
        0.18,
    )

    expected_90d_value = data["Average_Order_Value"] * (1 + data["Purchase_Frequency"] / 5)
    expected_coupon_cost = coupon_value * (base_prob + coupon_uplift)
    expected_incremental_revenue = coupon_uplift * expected_90d_value
    expected_incremental_profit = expected_incremental_revenue * gross_margin - expected_coupon_cost

    exp = data[
        [
            "User_ID",
            "RFM_Score",
            "Enhanced_Value_Score",
            "Enhanced_Segment",
            "Average_Order_Value",
        ]
    ].copy()
    exp["Base_Conversion_Proxy"] = base_prob
    exp["Coupon_Uplift_Proxy"] = coupon_uplift
    exp["Expected_Coupon_Cost"] = expected_coupon_cost
    exp["Expected_Incremental_Revenue"] = expected_incremental_revenue
    exp["Expected_Incremental_Profit"] = expected_incremental_profit

    traditional = exp.nlargest(target_n, "RFM_Score").assign(Experiment_Group="传统RFM-Top20%")

    optimized_pool = exp[
        exp["Enhanced_Segment"].isin(["核心用户", "潜力用户", "纠结土豪", "高潜观望者", "高潜流失客"])
    ]
    optimized = optimized_pool.nlargest(target_n, "Expected_Incremental_Profit").assign(Experiment_Group="优化RFM-核心与潜力")

    combined = pd.concat([traditional, optimized], ignore_index=True)

    summary = (
        combined.groupby("Experiment_Group")
        .agg(
            Target_Users=("User_ID", "count"),
            Budget=("User_ID", lambda s: len(s) * coupon_value),
            Avg_RFM=("RFM_Score", "mean"),
            Avg_Enhanced_Value=("Enhanced_Value_Score", "mean"),
            Avg_Base_Conversion=("Base_Conversion_Proxy", "mean"),
            Avg_Coupon_Uplift=("Coupon_Uplift_Proxy", "mean"),
            Expected_Coupon_Cost=("Expected_Coupon_Cost", "sum"),
            Expected_Incremental_Revenue=("Expected_Incremental_Revenue", "sum"),
            Expected_Incremental_Profit=("Expected_Incremental_Profit", "sum"),
        )
        .reset_index()
    )
    summary["Budget"] = budget
    summary["Expected_ROI"] = summary["Expected_Incremental_Profit"] / summary["Budget"]

    return combined, summary


roi_detail, roi_summary = roi_simulation(df)

display(roi_summary.round(4))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Experiment_Group</th>
      <th>Target_Users</th>
      <th>Budget</th>
      <th>Avg_RFM</th>
      <th>Avg_Enhanced_Value</th>
      <th>Avg_Base_Conversion</th>
      <th>Avg_Coupon_Uplift</th>
      <th>Expected_Coupon_Cost</th>
      <th>Expected_Incremental_Revenue</th>
      <th>Expected_Incremental_Profit</th>
      <th>Expected_ROI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>优化RFM-核心与潜力</td>
      <td>200</td>
      <td>2000.0</td>
      <td>65.4051</td>
      <td>65.8732</td>
      <td>0.3105</td>
      <td>0.0524</td>
      <td>725.6640</td>
      <td>2938.7781</td>
      <td>302.9083</td>
      <td>0.1515</td>
    </tr>
    <tr>
      <th>1</th>
      <td>传统RFM-Top20%</td>
      <td>200</td>
      <td>2000.0</td>
      <td>73.2254</td>
      <td>68.6470</td>
      <td>0.3319</td>
      <td>0.0442</td>
      <td>752.3305</td>
      <td>2169.3411</td>
      <td>6.9389</td>
      <td>0.0035</td>
    </tr>
  </tbody>
</table>
</div>


## 12. 可视化七：ROI 柱状 + 折线对比图



```python
roi_plot = roi_summary.copy()
roi_plot["Expected_ROI_%"] = roi_plot["Expected_ROI"] * 100

fig, ax1 = plt.subplots(figsize=(11, 6))

bar = sns.barplot(
    data=roi_plot,
    x="Experiment_Group",
    y="Expected_Incremental_Revenue",
    ax=ax1,
    palette=["#4C78A8", "#59A14F"],
)
ax1.set_ylabel("预期增量销售额")
ax1.set_xlabel("")
ax1.set_title("传统 RFM vs 优化 RFM-I：增量销售额与 ROI 对比", fontsize=16, fontweight="bold")

for p in bar.patches:
    ax1.annotate(
        f"{p.get_height():.0f}",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom",
        fontsize=11,
    )

ax2 = ax1.twinx()
ax2.plot(
    roi_plot["Experiment_Group"],
    roi_plot["Expected_ROI_%"],
    color="#E15759",
    marker="o",
    linewidth=3,
    label="预期 ROI",
)
ax2.set_ylabel("预期 ROI (%)")

for x, y in zip(roi_plot["Experiment_Group"], roi_plot["Expected_ROI_%"]):
    ax2.text(x, y + 0.3, f"{y:.2f}%", ha="center", color="#E15759", fontsize=11)

fig.tight_layout()
show_fig()

```


    
![png](ecommerce_rfm_analysis_notebook_pycharm_fixed_files/ecommerce_rfm_analysis_notebook_pycharm_fixed_27_0.png)
    


## 13. 可视化八：传统策略与优化策略触达人群差异



```python
touch_detail = roi_detail.merge(
    df[["User_ID", "Enhanced_Segment"]],
    on=["User_ID", "Enhanced_Segment"],
    how="left",
)

touch_segment = (
    touch_detail.groupby(["Experiment_Group", "Enhanced_Segment"])
    .size()
    .reset_index(name="触达人数")
)

plt.figure(figsize=(13, 7))
sns.barplot(
    data=touch_segment,
    x="Enhanced_Segment",
    y="触达人数",
    hue="Experiment_Group",
    palette=["#4C78A8", "#59A14F"],
)
plt.title("传统 RFM 与优化 RFM-I 触达人群结构对比", fontsize=16, fontweight="bold")
plt.xlabel("用户分层")
plt.ylabel("触达人数")
plt.xticks(rotation=30, ha="right")
plt.legend(title="")
plt.tight_layout()
show_fig()

```


    
![png](ecommerce_rfm_analysis_notebook_pycharm_fixed_files/ecommerce_rfm_analysis_notebook_pycharm_fixed_29_0.png)
    


## 14. 结论



```python
print("核心结论：")
print("1. 传统 RFM 更擅长找到历史消费高的人，但会漏掉高意向低消费、浏览多但未转化、以及高消费但正在流失的人。")
print("2. 优化 RFM-I 模型通过意向度、转化摩擦、活跃连接度和购买力背景标签，能识别更细的人群运营机会。")
print("3. 高潜观望者、纠结土豪、高潜流失客是传统 RFM 容易看不见，但很适合精准运营的人。")
print("4. ROI 对比中，优化策略把券投向更可能产生增量的人群，而不是只奖励历史高消费用户。")

display(roi_summary[["Experiment_Group", "Target_Users", "Budget", "Expected_Incremental_Revenue", "Expected_Incremental_Profit", "Expected_ROI"]].round(4))

```

    核心结论：
    1. 传统 RFM 更擅长找到历史消费高的人，但会漏掉高意向低消费、浏览多但未转化、以及高消费但正在流失的人。
    2. 优化 RFM-I 模型通过意向度、转化摩擦、活跃连接度和购买力背景标签，能识别更细的人群运营机会。
    3. 高潜观望者、纠结土豪、高潜流失客是传统 RFM 容易看不见，但很适合精准运营的人。
    4. ROI 对比中，优化策略把券投向更可能产生增量的人群，而不是只奖励历史高消费用户。
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Experiment_Group</th>
      <th>Target_Users</th>
      <th>Budget</th>
      <th>Expected_Incremental_Revenue</th>
      <th>Expected_Incremental_Profit</th>
      <th>Expected_ROI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>优化RFM-核心与潜力</td>
      <td>200</td>
      <td>2000.0</td>
      <td>2938.7781</td>
      <td>302.9083</td>
      <td>0.1515</td>
    </tr>
    <tr>
      <th>1</th>
      <td>传统RFM-Top20%</td>
      <td>200</td>
      <td>2000.0</td>
      <td>2169.3411</td>
      <td>6.9389</td>
      <td>0.0035</td>
    </tr>
  </tbody>
</table>
</div>


## 15. 导出结果



```python
df.to_csv(OUT_DIR / "user_value_segments_jupyter.csv", index=False, encoding="utf-8-sig")
segment_summary.to_csv(OUT_DIR / "segment_summary_jupyter.csv", encoding="utf-8-sig")
strategy_table.to_csv(OUT_DIR / "segment_marketing_strategy_jupyter.csv", index=False, encoding="utf-8-sig")
roi_summary.to_csv(OUT_DIR / "roi_experiment_summary_jupyter.csv", index=False, encoding="utf-8-sig")

print("已导出到：", OUT_DIR.resolve())

```

    已导出到： C:\Users\Lenovo\OneDrive\文档\New project\outputs_jupyter
    
