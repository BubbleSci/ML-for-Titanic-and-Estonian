{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7d4f9a6a-d93f-4d75-a2a5-2afef1b84ad3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e049ce4b-a969-4e70-9bf5-af442ef32d2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "4c919991-8ac1-472c-bf32-4a9b365faa86",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "import matplotlib"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b052d6de-a1a3-4a73-9a83-06edd009febc",
   "metadata": {},
   "outputs": [],
   "source": [
    "matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']\n",
    "matplotlib.rcParams['axes.unicode_minus'] = False #setting fonts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "38ae0719-37e7-4ad5-af52-31ac896ef27f",
   "metadata": {},
   "outputs": [],
   "source": [
    "train = pd.read_csv('train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "32395791-43da-4006-bb3b-9d773d730414",
   "metadata": {},
   "outputs": [],
   "source": [
    "test = pd.read_csv('test.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "f4f1be67-c4f5-4800-927b-7cac8ceb1e5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0caf6b39-f53a-421f-8576-8f7354654945",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 891 entries, 0 to 890\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  891 non-null    int64  \n",
      " 1   Survived     891 non-null    int64  \n",
      " 2   Pclass       891 non-null    int64  \n",
      " 3   Name         891 non-null    object \n",
      " 4   Sex          891 non-null    object \n",
      " 5   Age          714 non-null    float64\n",
      " 6   SibSp        891 non-null    int64  \n",
      " 7   Parch        891 non-null    int64  \n",
      " 8   Ticket       891 non-null    object \n",
      " 9   Fare         891 non-null    float64\n",
      " 10  Cabin        204 non-null    object \n",
      " 11  Embarked     889 non-null    object \n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 83.7+ KB\n"
     ]
    }
   ],
   "source": [
    "train.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a4e5cd3b-3cab-4fab-8e98-1b91a22d816c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "merged dataset: (1309, 12)\n"
     ]
    }
   ],
   "source": [
    "full = pd.concat([train, test], ignore_index=True)\n",
    "print('merged dataset:', full.shape) #merging train and test dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "dbab7a04-28ba-4249-93d1-d09f51bcff9f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1309 entries, 0 to 1308\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  1309 non-null   int64  \n",
      " 1   Survived     891 non-null    float64\n",
      " 2   Pclass       1309 non-null   int64  \n",
      " 3   Name         1309 non-null   object \n",
      " 4   Sex          1309 non-null   object \n",
      " 5   Age          1046 non-null   float64\n",
      " 6   SibSp        1309 non-null   int64  \n",
      " 7   Parch        1309 non-null   int64  \n",
      " 8   Ticket       1309 non-null   object \n",
      " 9   Fare         1308 non-null   float64\n",
      " 10  Cabin        295 non-null    object \n",
      " 11  Embarked     1307 non-null   object \n",
      "dtypes: float64(3), int64(4), object(5)\n",
      "memory usage: 122.8+ KB\n"
     ]
    }
   ],
   "source": [
    "full.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "d2bdcb74-4efe-4825-b241-bfb92e89ff07",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1309 entries, 0 to 1308\n",
      "Data columns (total 12 columns):\n",
      " #   Column       Non-Null Count  Dtype  \n",
      "---  ------       --------------  -----  \n",
      " 0   PassengerId  1309 non-null   int64  \n",
      " 1   Survived     891 non-null    float64\n",
      " 2   Pclass       1309 non-null   int64  \n",
      " 3   Name         1309 non-null   object \n",
      " 4   Sex          1309 non-null   object \n",
      " 5   Age          1046 non-null   float64\n",
      " 6   SibSp        1309 non-null   int64  \n",
      " 7   Parch        1309 non-null   int64  \n",
      " 8   Ticket       1309 non-null   object \n",
      " 9   Fare         1308 non-null   float64\n",
      " 10  Cabin        295 non-null    object \n",
      " 11  Embarked     1307 non-null   object \n",
      "dtypes: float64(3), int64(4), object(5)\n",
      "memory usage: 122.8+ KB\n"
     ]
    }
   ],
   "source": [
    "full.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "deda9d38-ac8f-4048-b7ff-06dc61217bda",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>1309.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>1309.000000</td>\n",
       "      <td>1046.000000</td>\n",
       "      <td>1309.000000</td>\n",
       "      <td>1309.000000</td>\n",
       "      <td>1308.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>655.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.294882</td>\n",
       "      <td>29.881138</td>\n",
       "      <td>0.498854</td>\n",
       "      <td>0.385027</td>\n",
       "      <td>33.295479</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>378.020061</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.837836</td>\n",
       "      <td>14.413493</td>\n",
       "      <td>1.041658</td>\n",
       "      <td>0.865560</td>\n",
       "      <td>51.758668</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.170000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>328.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>21.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.895800</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>655.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>982.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>39.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.275000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1309.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived       Pclass          Age        SibSp  \\\n",
       "count  1309.000000  891.000000  1309.000000  1046.000000  1309.000000   \n",
       "mean    655.000000    0.383838     2.294882    29.881138     0.498854   \n",
       "std     378.020061    0.486592     0.837836    14.413493     1.041658   \n",
       "min       1.000000    0.000000     1.000000     0.170000     0.000000   \n",
       "25%     328.000000    0.000000     2.000000    21.000000     0.000000   \n",
       "50%     655.000000    0.000000     3.000000    28.000000     0.000000   \n",
       "75%     982.000000    1.000000     3.000000    39.000000     1.000000   \n",
       "max    1309.000000    1.000000     3.000000    80.000000     8.000000   \n",
       "\n",
       "             Parch         Fare  \n",
       "count  1309.000000  1308.000000  \n",
       "mean      0.385027    33.295479  \n",
       "std       0.865560    51.758668  \n",
       "min       0.000000     0.000000  \n",
       "25%       0.000000     7.895800  \n",
       "50%       0.000000    14.454200  \n",
       "75%       0.000000    31.275000  \n",
       "max       9.000000   512.329200  "
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "full.describe() #describe dataset- to find missing value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "266263f0-cea4-412c-8660-b0b3c0b54a64",
   "metadata": {},
   "outputs": [],
   "source": [
    "full['Age']=full['Age'].fillna(full['Age'].mean())#Filling in missing values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "dd41759d-a207-473f-a683-841259fedea9",
   "metadata": {},
   "outputs": [],
   "source": [
    "full['Fare'] = full['Fare'].fillna( full['Fare'].mean() )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "dd31ea61-f712-45c8-8529-8d9f37c8982a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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AIDiBAnQS3Nzcw4++OAMHTp0h+3xxx/v03393d/9XZqbm3fbbLvz/n7yk5+kpqZmt9wX8H/nnVqBHdx66635/Oc/X3oM4ADiERKgT5qbm/OXf/mXGTNmTGpra3PGGWfk6aefzu/93u+lUqnkxBNPzJNPPtlz/P/8z/9k5syZOeKII3Lcccdl9uzZefPdBtavX5+LL744Rx11VIYOHZr3vOc9mT9/fs/X1tTU5JprrskRRxyR888/v9cc3d3dOffcc3PGGWeks7MzSXLHHXdk7NixqVQqaWlpyf33399z/Jo1a3LhhRdm+PDhOeGEE7Jo0aI9+ccE9JEgAfpswYIFeeihh7JmzZqsXbs2Z555Zr7yla+ko6MjH/zgB3Pttdf2HLty5cocdNBBWbFiRe65557cdNNNuf3225Mk06ZNy0EHHZTly5ens7Mzl19+eS677LK8+uqrPV//7LPPZuXKlT1fkySvvvpqLrjgglSr1fzwhz9MfX19HnjggXzhC1/IN7/5zXR0dORb3/pWpk6dmn/7t39Lknz2s59NtVrNCy+8kEcffTQ//vGP99KfFrArBAmwg8svvzzDhg3rtbW0tPSsT5kyJccee2wqlUpOPvnknHXWWTnttNMyYMCAnH322XnhhRd6jh0+fHhuvPHGDBkyJOPGjcv06dOzYMGCJG98uu83v/nNDB48OC+++GIqlUq2bduW9evX93z95z//+Rx00EEZNmxYkjfeSfP888/PmjVrcv/99+eQQw5J8sYHoF122WWZMGFCBgwYkDPOOCO///u/n/nz52fVqlV5+OGH81d/9VepVCoZOXLkTj9EDShHkAA7mDt3bjZv3txrW7p0ac/6kUce2fPPAwcOTGNjY8/tQYMGZfv27T23jz322Awa9L9PV2tubu75rJDnn38+5513XkaNGpXJkyfnX/7lX5K88WueNx122GG9ZnvppZcyePDg/OIXv+h59CNJXnjhhcydO7dXRN1+++1ZvXp1Vq1alSQ55phjeo4//vjj390fDrBHCBKgzwYM2PW/OtauXdvr9vPPP5/jjjsu3d3dOe+88/K5z30uL730UpYsWZJLL730He9v5MiReeihh3LVVVflc5/7XLZs2ZIkaWpqyle/+tVeEfXcc8/ltttu6wmRX//61z33s3Llyl0+B2DPEyTAHrVy5cr82Z/9WV5//fU88cQTaWtryxe+8IW8/vrree211zJgwIDU1NTkN7/5Ta655pokybZt237r/Q0ePDg1NTW58cYbM3DgwPzxH/9xkuTSSy/NzTffnCeeeCLVajXt7e350Ic+lO9///s56qij8slPfrLneS5r1qzJ9ddfv1fOH9g1ggTYwR/+4R/u9H1Ivv71r/f5vj7wgQ9k9erVOfTQQ3PxxRdnzpw5mThxYurq6vLd7343X/va1zJ06NB85jOfyVe+8pWMGDEiP//5z9/xfg8++ODMnz8/t912W/75n/85F154YW666aZMnz49DQ0NueCCCzJjxoxcfvnlSd54D5Nhw4blPe95T37nd34nn/jEJ/p8LsCe49N+AYDiPEICABQnSACA4gQJAFCcIAEAihMkAEBxggQAKE6QAADFCRIAoDhBAgAUJ0gAgOIECQBQ3P8DIPrHdImuY3wAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "a=full['Embarked'].value_counts()\n",
    "a.plot.bar(color='black')\n",
    "full['Embarked']=full['Embarked'].fillna('S')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "853f0b6d-841e-43e1-9448-4903f296b18b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0     NaN\n",
       "1     C85\n",
       "2     NaN\n",
       "3    C123\n",
       "4     NaN\n",
       "Name: Cabin, dtype: object"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "full['Cabin'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "08771d33-675b-4606-ae33-a49496c6818f",
   "metadata": {},
   "outputs": [],
   "source": [
    "full['Cabin'] = full['Cabin'].fillna('U') #Replace missing values in cabin with the string U"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "ea524a36-a224-41e7-8c9c-d46183dfb132",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 640x480 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig = plt.figure()\n",
    "Survived_cabin = train.Survived[pd.notnull(full.Cabin)].value_counts()\n",
    "Survived_nocabin = train.Survived[pd.isnull(full.Cabin)].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "fed2cfe9-2b1d-4f7d-9bd6-84b0c9824e22",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'Percentage of cabin')"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAcMAAAGsCAYAAAChJMxRAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAABCmElEQVR4nO3dd3hUZd7G8e+ZQBopJBCa9C6IIAgCiqCISAfLKiqKuq7isqJig1fFjmUVK+JaQF0VlFUQFFARLIiCIF1AIPQaCEkI6fO8fxwZHBMgk3am3J/rmivJ5MyZ30wmc89zzlMsY4xBREQkhLmcLkBERMRpCkMREQl5CkMREQl5CkMREQl5CkMREQl5CkMREQl5CkMREQl5CkMREQl5CkMREQl5CkMpkYcffhjLsgpdKleuTLVq1ejRowf//e9/nS6z3KSnpzNp0iSnyygX+fn53HPPPdSqVYvIyEjOPPPMCr3/Hj16YFkWhw8fPuW2x16HM2bMKPe6JLhVcroACWyDBg2iXbt2np8LCgo4cOAA06ZNY9iwYfz+++888sgjzhVYTlq0aEHNmjW59dZbnS6lzL311lv8+9//plmzZgwfPpwaNWo4XdIJ9ejRA4CWLVs6W4gEPIWhlMrgwYMZPnx4oetHjx7NWWedxZNPPslNN91E/fr1K764crR3715q1qzpdBnlYvny5QC8+uqr9OrVy+FqTq5Hjx6eQBQpDR0mlXLRrFkzBg8eTH5+PnPnznW6HPFBTk4OAElJSQ5XIlJxFIZSbk477TQAUlJSvK7/+OOP6dq1KzExMcTFxdGzZ08WLFjgtc3ChQuxLIuJEydyxRVXEBkZSe3atVm0aBEAR48eZdy4cbRo0YKoqCgaN27MyJEjOXDggNd+cnNzGT9+PK1atSIyMpIaNWpwzTXXsGXLFq/tpkyZgmVZzJ8/33OIMDIykiZNmvD4449TUFDgVRfAypUrsSyLhx9+2LOfH374gUsvvZTatWsTHh5OQkICvXr1Yv78+YWen82bNzN06FBq1qxJTEwMffv25bfffqNp06aFWjvFfRwns3jxYgYMGEBCQgKRkZG0adOG5557jvz8fAC2bt2KZVm88847AJx11llYlsXChQtPut/k5GT+/ve/U7duXaKjo2ndujXPP/88eXl5Xtv58twAbN++nSFDhhATE0O1atW4+uqrCz3eos4ZWpbF8OHD+fHHH+nRowcxMTEkJCRw5ZVXsnXr1mI/XxJijEgJjBs3zgBm8uTJJ9zm0ksvNYB55513PNc9+OCDBjCNGjUy//znP83IkSNNrVq1jMvlMu+9955nuwULFhjA1KhRwzRt2tTcc889pk+fPiYzM9NkZmaatm3bGsB06tTJ3HnnnWbgwIEGMK1btzbp6enGGGNyc3PNhRdeaADTuXNnM3r0aHPdddeZyMhIk5iYaFavXu25v8mTJxvAdOjQwVSpUsXccMMN5s477zS1atUygHn88ceNMcYkJyd7HnvNmjXNuHHjzIIFC4wxxsyYMcO4XC5Tu3ZtM2LECHPvvfeavn37GsuyTFhYmFm+fLnn/jZu3GiSkpKMy+UyQ4YMMXfffbdp0aKFSUxMNFWrVjXdu3f3bOvL4ziRadOmmbCwMBMVFWWGDh1q/vWvf5mWLVsawPTt29fk5+eb1NRUM27cOM9ze8stt5hx48aZ5OTkE+531apVJiEhwViWZQYMGGDuvvtu06FDBwOY6667zrOdL89N9+7dDWBq1aplWrRoYe6++24zYMAAz+th27Ztnm2P/S0+/fRTz3WAadOmjQkPDzcXXnihueeee8z5559vANO8eXPjdrtP+XxJ6FEYSomcKgyXLl1qKlWqZCIjI82+ffuMMcb8/PPPxrIsc+GFF5qjR496tj148KBp3ry5qVKlijlw4IAx5ngYRkdHmz179njt+4EHHjCAGT16tNcb22OPPWYAM2HCBGOMMc8884wBzJgxY7xuv2zZMhMeHm46derkue5YGMbHx5vff//dc31ycrKpXLmyqVevntc+ANO2bVuv646F2d69e72uf/bZZw1g7r//fs91ffv2NYCZNm2a57rs7Gxz3nnnGcArDH15HEVJTU018fHxpmrVqmbFihVe93fsQ8RLL73kuf766683gPn1119Pul9jjDnvvPOMZVlmxowZnusKCgpMr169vPbhy3NzLAy7dOlisrKyPNdPmjTJAObqq6/2XHeiMATMM88847nO7Xabiy++2ABm/vz5p3xcEnoUhlIix96EBg0aZMaNG+e5jB071lx++eUmMjKy0JvsLbfcYgDzyy+/FNrfG2+8YQDzyiuvGGOOh2GvXr0KbdukSRMTFxdnsrOzva5PT0839957r5k3b54xxn4Drlq1qsnLyyu0j2uuucYAZs2aNcaY42F44403Ftr2WEvpz2/Mfw3DgoIC88knn5jPP/+80O2XLVtmAHPTTTcZY4zZv3+/cblc5txzzy207aJFiwqFoS+PoyjvvPOOAcy4ceMK/W7r1q0mLCzM67EUNwx37NhhANO7d+9Cv/vll1/MuHHjzNq1a316bow5HoYLFy4stH2rVq1MRESE529/ojCMiooyOTk5XredMGGCAcxrr7120scloUm9SaVUZs6cycyZMz0/V65cmerVq3PxxRczYsQILrnkEs/vli1bBsD06dOZNWuW13527twJwIoVK7yub9SokdfPWVlZbN68mfPPP5+IiAiv38XGxvL0008DcOTIETZs2ECtWrV4/PHHC9W9d+9ez/21bt3ac33z5s0LbRsfHw/YHUsiIyOLeBbA5XIxZMgQALZt28aaNWvYtGkTa9eu5bvvvgPwnHdcvnw5brebLl26FNrPOeecQ6VKx/8tS/o4/mzlypUAdOvWrdDvGjRoQL169Vi9ejVutxuXq/jdCFatWuWp+a86dOhAhw4dPD8X97k5xuVynfD5WbduHevXr6dt27YnrK1BgwaEh4d7Xffnv6PIXykMpVQmT55c5NCKohwbRP3UU0+dcJtDhw55/RwVFVXk7+Pi4k56X2lpaYAdFicb5/jX+/trwAKeDjPGmJPe5+rVq7n99ts9HU7Cw8Np1aoVnTp1YsOGDZ7bH+tQVNTQjLCwMK9xfSV9HH+Wnp4OnPg5q1OnDlu3biUnJ6fQ830yqampJ93vnxX3uTkmMTGxUJiB/YEHIDMz86T3V5q/o4QmhaFUmJiYGMLCwsjKyqJy5col3gdARkZGkb/PzMykSpUqnu26devmaX2Up4yMDHr16kVaWhpPP/00ffr04fTTT6dSpUr88ssvvPfee55tj4XHsZAqal/HlMXjOBYgu3fvLvL3qampREVF+RSEf66tqL+F2+32hKsvz80xR44cKfI+jz2GhIQEn2oVORUNrZAK07ZtWwoKCgodCgW72//9999/yjf8+Ph46tWrx4oVK8jNzfX6XU5ODklJSVx88cXEx8fToEED1q5dS3Z2dqH9vPvuuzz88MMkJyeX6jEdM3/+fPbt28fIkSO59957adOmjedw59q1a4HjLZL27dtjWRY///xzof2sW7fOK1zK4nEcmyHo2LCUP9u7dy8bN2484SHWk2nTpg0AS5YsKfS7H374gSpVqvDEE0/49Nwck52dzbp167yuy8/PZ8mSJcTGxhZ5OFukNBSGUmGOHU696667vN7wMzIyGDFiBE8//bRnzNvJXHvttaSlpfHoo496Xf/iiy+SlZXFRRdd5Lm/Q4cOMXbsWK8323Xr1jFy5Eiee+45EhMTS/RYKlWq5DWO7lirat++fV7bbd++3TMO8dj2p512Gr169eLrr7/miy++8Gybk5PDvffeW+i+Svs4Bg0aRFxcHBMnTvScPzx2fyNHjqSgoIDrrrvOh0dva9y4MV26dGHevHnMmzfPc73b7eaZZ57BGEOvXr18em7+bOzYsV7nEp955hm2b9/OddddR1hYmM/1ipyMDpNKhenRowe33347L730Eq1bt6Zfv36Eh4fz6aefsmPHDm6++WYuvPDCU+5n7NixfP755zzxxBN8++23nHPOOfz222988cUXdOzYkTvuuAOA+++/n7lz5zJhwgS+/fZbunfvzuHDh/n444/JzMzknXfe8XSq8FXdunVZv349//znP7nkkku48MILadiwIe+99x4pKSm0bduWHTt2MHPmTCIjI7Esi4MHD3pu/9JLL9G5c2cGDhzIkCFDOO2005g3b57nfOKf3+xL+zji4+N58803GTp0KF26dGHw4MEkJSXx5Zdfsn79evr06cNtt91Woufh9ddf5/zzz6dfv34MHjyYhg0bMn/+fFasWMGoUaPo1KkTmZmZPj03ANHR0SxZsoROnTrRs2dPVqxYwVdffUXLli2L7EgkUmqO9WOVgFacQfcn8t5775kuXbqY6OhoExcXZzp27GjefPNNU1BQ4Nnm2NCKUaNGFbmPtLQ0c88995iGDRuaypUrmzp16ph//etf5vDhw17bHT161DzyyCPm9NNPNxEREaZGjRqmd+/enoHyxxwbWnFsjOKfHevqn5qa6rlu9uzZpnHjxiY8PNwzLOD33383l156qalZs6aJjo42LVu2NLfeeqvZvXu3adeunYmKijIZGRmefaxdu9YMHDjQxMfHm+joaNO/f3+zatUqA5j+/fuX6HGczKJFi0zfvn1NfHy8iYqKMu3atTMvvviiyc/P99rOl3GGxtgTCAwdOtRUr17dVK5c2bRo0cJMmDDB6+/py3PTvXt3U61aNbN+/XrTq1cvExUVZZKSkswtt9xiUlJSvO77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iaF27drccsstpKamen6/efNmrrjiChITE0lISKBv376sXbsWKPow6apVq+jVqxcxMTE0aNCA999/v1Bdbrebp556iqZNmxIREUHz5s15+eWXvbbp0aMHf//733n66aepX78+kZGRdO3alZ9//hmAKVOmcMMNNwDQqFGjUh2G7Vgd2mrAq4QwA7ygQKxwPTTWuUR8DsPrrruOt956izFjxvDll1/y3HPPsXr1aq666iqMMcyZM4dLLrmEatWqMW3aNJ599lk+++wzLr/8cgD27NlDp06d+O2335g4cSLvv/8+hw4domfPnqSkpBS6v127dnH++edz6NAh3n//fR5//HHuu+8+du3a5bXdiBEjeOihh7j22muZNWsWV1xxBXfccQePPfaY13bTp09nxowZvPTSS3z44Yfs27ePyy+/nIKCAvr168cDDzwAwCeffMKDDz7o69Pj5aZmpbq5SMDLdsPrG7TeZ0U5LcpeyFd859Mw2dzcXDIyMnj55Ze58sorAejevTsZGRmMHj2avXv38tBDD9G2bVs+/fRTz+2ioqIYM2YMu3fvZsKECWRlZfH1119Tq5a9RPxZZ51F586dWbx4MW3atPG6zxdeeIG8vDzmzJlDjRr2WeHmzZvTuXNnzzYbN27kjTfeYPz48dx3330AXHzxxbhcLp588kluu+02qlWrBkBeXh7z5s0jLi4OgIyMDK6//npWrFhBhw4daNKkiaemhg0b+vL0FHJ1I7hvGWQVlGo3IgEtNRembYFLG0JantPVBLfuahWWmE8tw/DwcObOncuVV17Jnj17+O677/jPf/7D7NmzATssly1bxpAhQ7xud9lll7Fx40bq1KnD999/T5cuXTxBCHanmW3btjFgwIBC93ls+2NBCHDOOedQv/7xqV6++eYbjDEMGDCA/Px8z2XgwIFkZ2fz/fffe7Zt3bq1JwgB6ta1V+XNzMz05akolqrhMKxxme9WJOBsz4RvdkN0mNOVBK8IF3Sp7nQVgcvnCZTmzZvHHXfcwfr164mNjeXMM88kNtZul+/cuRNjjFdw/dXBgwdp1KhRse/v0KFDRW5fu/bxke0HDx4E7KAryu7duz3fR0d7z0/kctmfB9zu8lmx9M5W8OYmcJd82KJIUFiVas9h2iIBcrVAcJk7p7rdk11Kxqcw3Lx5M4MHD2bQoEHMnj3bc0hx4sSJ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",
      "text/plain": [
       "<Figure size 500x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.rc('font',family='Arial',size=12)#font\n",
    "fig=plt.figure(figsize=(5,5))\n",
    "ax1=fig.add_subplot(1,1,1)\n",
    "labels=['accident','survive']\n",
    "colors=['#03A2FF','#64C8FF']\n",
    "ax1.pie(Survived_cabin,labels=labels,colors=colors,startangle=90,autopct='%1.1f%%')\n",
    "ax1.set_title('Percentage of cabin')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "1f938295-1419-4bd0-bbc0-00d35e49b8d0",
   "metadata": {},
   "outputs": [],
   "source": [
    "def getTitle(name):\n",
    "    str1=name.split(',')[1]#Mr.Owen Harris\n",
    "    str2=str1.split('.')[0]#Mr\n",
    "    #The strip() method is used to remove the beginning and end of the string (defaults to spaces)\n",
    "    str3=str2.strip()\n",
    "    return str3\n",
    "#Storing the extracted features\n",
    "titleDf=pd.DataFrame()\n",
    "#map function: applies a custom function to each of the data in the Series.\n",
    "titleDf['Title']=full['Name'].map(getTitle)\n",
    "titleDf.head()\n",
    "#Mapping of title strings in names to defined title categories\n",
    "title_mapDict={\n",
    "                'Capt':        'Officer',\n",
    "                'Col':         'Officer',\n",
    "                'Major':       'Officer',\n",
    "                'Jonkheer':    'Royalty',\n",
    "                'Don':         'Royalty',\n",
    "                'Sir':         'Royalty',\n",
    "                'Dr':          'Officer',\n",
    "                'Rev':         'Officer',\n",
    "                'the Countess': 'Royalty',\n",
    "                'Dona':         'Royalty',\n",
    "                'Mme':          'Mrs',\n",
    "                'Mlle':         'Miss',\n",
    "                'Ms':           'Mrs',\n",
    "                'Mr':           'Mr',\n",
    "                'Mrs':          'Mrs',\n",
    "                'Miss':         'Miss',\n",
    "                'Master':       'Master',\n",
    "                'Lady':         'Royalty'}\n",
    "titleDf['Title']=titleDf['Title'].map(title_mapDict)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "055bd496-93f3-4608-9a16-0202ad1f2dea",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Mr</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Mrs</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Miss</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Mrs</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Mr</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Title\n",
       "0    Mr\n",
       "1   Mrs\n",
       "2  Miss\n",
       "3   Mrs\n",
       "4    Mr"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titleDf.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "5a0aa22e-fecd-4132-9b65-03b979a64d5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "full=pd.concat([full,titleDf],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "6f6946d3-d238-4276-a1d9-777ab0953527",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Title</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>U</td>\n",
       "      <td>S</td>\n",
       "      <td>Mr</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>Mrs</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>U</td>\n",
       "      <td>S</td>\n",
       "      <td>Miss</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>Mrs</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>U</td>\n",
       "      <td>S</td>\n",
       "      <td>Mr</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1       0.0       3   \n",
       "1            2       1.0       1   \n",
       "2            3       1.0       3   \n",
       "3            4       1.0       1   \n",
       "4            5       0.0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked Title  \n",
       "0      0         A/5 21171   7.2500     U        S    Mr  \n",
       "1      0          PC 17599  71.2833   C85        C   Mrs  \n",
       "2      0  STON/O2. 3101282   7.9250     U        S  Miss  \n",
       "3      0            113803  53.1000  C123        S   Mrs  \n",
       "4      0            373450   8.0500     U        S    Mr  "
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "full.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "8851b6a0-5a24-4e59-b076-d6c18e07509e",
   "metadata": {},
   "outputs": [],
   "source": [
    "def survived_rate1(string):\n",
    "    survivedRate = full.groupby([string, 'Survived']).Survived.count().unstack()\n",
    "    survivedRate['Total'] = survivedRate[0].values + survivedRate[1].values\n",
    "    survivedRate['Rate Survived'] = survivedRate[1].values / survivedRate['Total'].values\n",
    "    \n",
    "    survivedRate = survivedRate.sort_values(by='Rate Survived', ascending=True)\n",
    "    \n",
    "    \n",
    "    survivedRate = survivedRate.fillna(0)\n",
    "    survivedRate.rename(columns={0: \"die\", 1: \"survived\"}, inplace=True)\n",
    "    \n",
    "    # create object of chart\n",
    "    fig = plt.figure(figsize=(6, 6))\n",
    "    \n",
    "    survivedRate[[\"die\", \"survived\"]].plot(kind=\"barh\", stacked=True, color=[\"k\", \"g\"])\n",
    "    plt.grid(axis=\"x\", ls=\"-\")\n",
    "    plt.ylabel(string, fontsize=15)\n",
    "    plt.xlabel(\"Number\", fontsize=15)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "a2267a50-c625-47fd-8932-259b7e8975e1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 600x600 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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0e3btxUeHp7th7AOHz5cH330kT766CN16NBBPj4+WrJkiU2fRYsWqWLFiqpZs6ZNe926dRUWFpap/8KFC+Xn56c2bdrIMAx16NBB//73vzVo0CCtXbtWdevW1fPPP68vv/zS+p3BgwdrzJgx6tOnj1avXq2AgAANHTo0myNjH04bAQDgID/99JOuXr2qN954Q/Xq1ZMkVapUSdOnT8/2Yz369++vzp07Wz936tRJS5cu1fjx4yVJd+7c0Zo1azRkyJD7fj8yMlKTJk1SUlKSvLy8JEmLFy9Wly5d5OHhoc2bN2vjxo1asmSJunXrJkkKDw/X7du3NXToUHXv3l2JiYmaPHmyBg4cqNGjR1v7XLhwQRs3bszW77FHnp55adSokQzDUKNGjZxdCgAA2fbUU08pMDBQbdq0Uf/+/bVu3TqVKlVKH3zwgUJCQrK1rSpVqth8joqK0unTp62noNatW6fExET16NHjvt+PiorS7du3tW7dOknS999/r1OnTikqKkqStHXrVlksFkVEROjevXvWpW3btrp06ZKOHTumvXv3KjU1Ve3atbPZdteuXbP1W+yVp8MLAABm5u3trZ07dyoiIkJLlixR27ZtFRgYqH79+ik5OTlb2ypZsqTN5+eee04hISHWU0GLFi1S/fr1M11Dk6FcuXKqV6+eTf/Q0FDVr19fkhQXFyfDMOTj4yM3NzfrkhFMLl68aL22JTAw0GbbpUqVytZvsRenjQAAcKCKFStq/vz5SktL0/fff6/58+friy++ULly5WSxWJSWlmbTPzExMUvbtVgs6tGjh+bPn69//etf2rBhgz7//PO//E5UVJTefPNNxcfHa9myZerTp48slj8eBOjv7y9vb299++239/1uWFiYvv/+e0nSlStXVLFiReu6uLi4LNWcU5h5AQDAQVasWKHAwEBdvnxZrq6uqlOnjqZNmyZ/f3/FxMTI19dX165ds5mF2bVrV5a3HxUVpQsXLmjUqFGyWCzq0qXLX/bPmEUZOXKkLl26pMjISOu6hg0bKjExUYZhqEaNGtbl2LFjGjNmjO7du6e6devK09PTendShoxTUbmFmRcAABykXr16SktLU/v27TV06FD5+vpq6dKlio+PV6dOnXTv3j1NnjxZffr00UsvvaRjx47pww8/lKura5a2X7lyZVWvXl3Tpk1Tx44d5efn95f9ixYtqtatW2vatGmqWbOmKlWqZF3XqlUrNWjQQO3atdPIkSP15JNP6vvvv9eoUaMUHh6ugIAASX8En3feeUdFihRR48aNFR0dTXgBACBDdt8plJ6eroSEBPn6+srFxfknF0qVKqVNmzbpnXfe0YsvvqikpCQ99dRTWrlypZ577jlJ0ocffqjJkydr1apVql69ulavXq26detmeR9RUVE6ePCgzSzKw/qvXLkyU38XFxdFR0dr5MiRev/99xUbG6vHHntMb731lv71r39Z+w0bNkze3t769NNP9emnn6pu3br66KOP1L9//yzXbC+LYRj57m1TCQkJ8vPzk4eHR7YviCqwRufcpjxdPLW46mK9cPQF3Um/k3MbRpbYO/68gM4+qampio6OVqtWreTm5ubsckwhOTlZp0+fVtmyZeXh4WHXtvJaeCmIHnQMsnKcM/5+x8fHy9fX94H74MgCAABTIbwAAABTIbwAAABTIbwAAABTIbwAAPKEfHj/CP5HTh5fwgsAwKky7spKSkpyciVwpIzjmxN34fGcFwCAU7m6usrf31+xsbGSJC8vL+sj67MrPT1dd+/eVXJyMrdKO8mfj4FhGEpKSlJsbKz8/f2z/AC+v0J4AQA4XVBQkCRZA8yjMgxDd+7ckaen5yMHINjnQcfA39/fepztRXgBADidxWJRqVKlVKJECaWmpj7ydlJTU7Vjxw41aNCAhwQ6yf2OgZubW47MuGQgvAAA8gxXV1e7/si5urrq3r178vDwILw4SW4cA04IAgAAUyG8AAAAUyG8AAAAUyG8AAAAUyG8AAAAUyG8AAAAUyG8AAAAU8nXz3k5f/68ihcv7uwyCpzU1FRFR0crfmg8z1lwAsYfQH7HzAsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsAADAVwgsA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0aRJE8PT09MoUaKE8fLLLxsJCQk2fa5fv2707t3b8Pf3N3x9fY2WLVsaP//8cy5Wbz6zZ882JD1wmTNnjmEYjL8jJScnG+PGjTMqVKhguLu7G48//rjxzjvvGMnJydY+jH/u+XN4MQzG35Hu3LljjB071njiiScMd3d3o1y5csb48eONe/fuWfvk9vhbDIMX/wAAAPPIN9e8AACAgoHwAgAATIXwAgAATIXwAgAATIXwAgAATIXwAgAATIXwAgAATIXwAgAATIXwAuCh5s6dK4vFotKlS+vGjRsP7Dd69GhZLBbNnTs394p7QA1/fhcRgPyD8AIgyy5duqQ33njD2WUAKOAILwCyZcGCBVq7dq2zywBQgBFeAGRZ9erVJUn9+vXT9evXnVwNgIKK8AIgy1q0aKGoqChdvnw5S6eP/ur6k969e8tisWjLli3Wtscff1x///vfdfr0aXXt2lVFixaVj4+PwsPDdeLECaWlpWnixIkqX768vLy8VLVqVS1atOi++05LS9PYsWMVGhoqDw8PValSRVOmTFF6enqmvufPn1e/fv0UEhKiwoULKzg4WC+//LIuXLhg0y/j2p+5c+eqW7du8vT0VIkSJbR69eqHjgWAnFPI2QUAMJfPPvtMW7Zs0cKFC9W1a1e1bds2R7d/7do11a5dW6GhoXrppZd08OBBff3112rRooWeffZZbd68WZ06dVJ6erq+/PJL9ejRQyEhIapfv77NdsaMGaObN2/qhRdekIeHh9auXas33nhDR44csQlTx48f13PPPadr164pIiJClStX1smTJzV79mytXbtWO3bsUIUKFWy2PXToUPn4+GjAgAH68ccfVadOnRwdAwB/jfACIFuKFi2q6dOnq23bturXr5+effZZFStWLMe2f+HCBXXu3FnLli2TxWKRJNWuXVv79u3Txo0bdezYMZUqVUqSVLNmTb300ktasGBBpvASFxenPXv26JlnnpEkjR07Vk2aNNHs2bMVFRWlhg0bSpKioqIUFxen6OhotWjRwvr9DRs2qFWrVvrnP/+pXbt22Ww7KSlJx44dU0BAQI79bgBZx2kjANnWpk0bRUZGZvn0UXa9/fbb1uAiSc8++6wkqVevXtbgIkn16tWTJJ05cybTNl588UVrcJGk4sWLa9y4cZKk+fPnS5L279+vH374QZ06dbIJLpLUsmVLNWvWTLt379aJEyds1jVu3JjgAjgRMy8AHsnkyZO1detWLVy4UF26dFG7du1ybNt/Pk3j7e0tSSpfvrxNu6enpyQpJSUl0zb+PBMjyXp659ChQ5L+CC+SdOXKFY0ePTpT/8TERGv/J5980tperly5LP0OAI5BeAHwSP739NErr7xy37DwqDLCyp+5u7tneRtBQUGZ2nx8fCT9XyjJeODejh07tGPHjgdu6893Vnl5eWW5DgA5j9NGAB7Zw04fZZz6ud8dPrdv33ZobTdv3szUlnH3UMY1OhlhZuLEiTIM44HLgAEDHForgOwhvACwy+TJkxUUFKSFCxdq/fr1NusKFy4sSbp161am7/32228OrSvjlND/2rlzp6Q/LvSVpKefflqStG/fvvtuY9q0aRozZsx9r6kB4DyEFwB2yTh9JEkHDhywWZdxncj69etlGIa1fdWqVTpy5IhD65o5c6ZOnz5t/Xzp0iWNGTNGLi4u6tu3r6Q/LvitVKmSVq9enelZLbt27dLAgQP1ySefcHEukMdwzQsAu7Vt21Y9evTQwoULbdojIiIUGhqqb7/9Vs8++6zq16+vn3/+WevXr1fDhg21fft2h9VUtGhRVa9eXc8//7zS09O1evVqxcbGavz48apWrZokycXFRQsWLFDTpk3VsWNHhYeHq2rVqjp37pxWr14twzA0Z86cB16DA8A5mHkBkCMyTh/9Lzc3N23btk3PP/+8fv75Z02ePFmxsbFau3atOnfu7NB6pkyZor59+2rVqlWaO3euypQpo+XLl2vo0KE2/apXr65Dhw6pb9++On78uD799FN99913ioiI0O7du9WhQweH1gkg+yzG/87lAgAA5HHMvAAAAFMhvAAAAFMhvAAAAFMhvAAAAFMhvAAAAFMhvAAAAFMhvAAAAFMhvAAAAFMhvAAAAFMhvAAAAFMhvAAAAFMhvAAAAFP5/6mLzZ0eROVZAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "survived_rate1(\"Pclass\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "6d3c9324-5a88-4273-9c24-751cfaf92055",
   "metadata": {},
   "outputs": [],
   "source": [
    "def survived_rate2(string):\n",
    "    \n",
    "    survivedRate = full.groupby([string, 'Survived']).Survived.count().unstack()\n",
    "    survivedRate['Total'] = survivedRate[0].values + survivedRate[1].values\n",
    "    survivedRate['Rate Survived'] = survivedRate[1].values / survivedRate['Total'].values\n",
    "    \n",
    "    \n",
    "    survivedRate = survivedRate.sort_values(by='Rate Survived', ascending=True)\n",
    "    \n",
    "    \n",
    "    survivedRate = survivedRate.fillna(0)\n",
    "    survivedRate.rename(columns={0: \"die\", 1: \"survived\"}, inplace=True)\n",
    "    \n",
    "    \n",
    "    fig = plt.figure(figsize=(6,6))\n",
    "    \n",
    "    \n",
    "    survivedRate[[\"die\", \"survived\"]].plot(kind=\"line\", color=[\"b\", \"r\"])\n",
    "    \n",
    "    plt.grid(axis=\"y\", ls=\"-\")\n",
    "    plt.ylabel(\"Number of Passengers\", fontsize=15)  \n",
    "    plt.xlabel(string, fontsize=15)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "65e24951-ab8a-417d-a215-97b74172ac1c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Figure size 600x600 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "survived_rate2(\"Embarked\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "4e66c0f8-069d-4ccb-b2a6-bcc8c0d5f849",
   "metadata": {},
   "outputs": [],
   "source": [
    "def survived_rate3(string):\n",
    "    \n",
    "    survivedRate=full.groupby([string,'Survived']).Survived.count().unstack()\n",
    "    survivedRate['Total']=survivedRate[0].values+survivedRate[1].values\n",
    "    survivedRate['Rate Survived']=survivedRate[1].values/survivedRate['Total'].values\n",
    "    \n",
    "    survivedRate=survivedRate.sort_values(by='Rate Survived',ascending=True)\n",
    "    \n",
    "    survivedRate=survivedRate.fillna(0)\n",
    "    survivedRate.rename(columns={0:\"die\",1:\"survive\"},inplace=True)\n",
    "    survivedRate[[\"die\",\"survive\"]].plot(kind=\"bar\",stacked=True,color=[\"k\",\"r\"])\n",
    "    plt.grid(axis=\"y\",ls=\"-\")\n",
    "    plt.ylabel(\"Total\",fontsize=15)\n",
    "    plt.xlabel(string,fontsize=15)\n",
    "    fig=plt.figure(figsize=(6,6))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "da0eed0a-f9cc-4358-aa9d-cdff897044cd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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vriHkc/ChQsdtq+SwhAiAMBdXbx4sdBtXTYILVq0SP3791dkZKQSExNVsWJFSVJgYGCBvTpZWVkKCwuTdG2IraAhsKysLAUGBt7wM8eMGaPhw4fbnmdkZKhGjRqKjY0tsIepuG5Wg6tIT093dgkAABTLzUZ//pdLBqE33nhDo0aNUtOmTbVixQq74BAZGalDhw7Ztc/Ly1NycrK6du1qa5OYmKi8vDy7HqBDhw6pbt26N/xcb29veXt759vu5eUlLy+v2z0sm0uXLjlsXyXFkccLAEBpKsp3mEtNlpak999/XyNHjlT37t21bt26fL0nsbGxSkpKUkpKim1bYmKiMjMzbVeJxcbGKjMzU4mJibY2KSkpSkpKsruSDAAAmJvFMAzD2UVcd+rUKdWqVUuVK1fWZ599lu+qr7vvvlsWi0X33nuvqlevrnHjxik1NVUjR45UgwYN7Oa1NG/eXLt379b06dNVqVIlxcfHKzU1VXv37tUdd9xRqHoyMjIUGBio9PR0hw6NFTR/ydW40I8FAABFUpTvb5caGlu9erUuXbqko0ePKioqKt/rH3/8sfr3769NmzZp2LBh6tOnj/z9/dW9e3fNmDHDrm1CQoKGDx+uESNGKC8vT40bN9aSJUsKHYIAAEDZ51I9Qq6GHiEAANxPUb6/XW6OEAAAQGkhCAEAANMiCAEAANMiCAEAANNyqavGAACQJLnBRSWSJC4scXv0CAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANMiCAEAANPydHYBAACgBFkszq7g1gzDaR9NjxAAADAtghAAADAtghAAADAtghAAADAtghAAADAtghAAADAtghAAADAtghAAADAtghAAADAtghAAADAtghAAADAtghAAADAtlw5Cx44dU1BQkDZv3my3/eDBg2rfvr0CAwNVqVIlDRo0SGlp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euedd9S0aVOFh4fLz89PCQkJ6tevn7Zv384VY7ewaNEiu6udWrVqVWavdipJ/7t23UsvvVRm166jRwimwhUlJSM4OFjz589X69atNWDAACUmJurEiROSpMWLF2vYsGE6efKkk6t0D3v27FHTpk314IMPatOmTRo9erT+/ve/67777tOePXv0zDPP6J133nF2mW6jLF/tVJLMtHYdQQimsnnzZj366KMqX768s0spUzp37qwLFy5o4MCBeuqppzRgwAC9++67+uGHHxQXF6f77rtPS5cudXaZbuPUqVPau3evYmJiZBiGpk6dqm+//VaPPvqoxowZU2aGJEpbWbvaqSSZae06ghBMJSgoSKtWrVLjxo3VokULzZo1S3Xq1HF2WW4vOTlZ7du313/+8x/VrVtXGzZsUGhoqEJDQ1W+fHmtX78+3xpDKNjWrVsVFRXl7DLKpN9//12ff/65Pv/8c/373//WHXfcwVphIAjBXAIDAzV8+HD1799f4eHhWr58+U3nDNx5552lV5ybMwxDZ86cUZUqVWzbduzYoQcffNDuFia4OavVqrvvvlsDBgxQXFycatSo4eyS3FpZvtoJjkEQgqkMGDBAc+fOlcVikWEYt+z6ZY7Q7btw4YK2bt2qNm3aOLsUt7Bjxw7NnTtXS5YsUXp6ulq0aKGBAweqS5cuBMpCMsvVTnAMghBMJTc3V2vXrtXZs2c1YMCAAm8L8d/69etXitW5r6NHj2rw4MG220IUhFBZNFeuXNHy5cs1d+5crV+/Xn5+furVq5cGDBigRx991Nnluaz/vdqpZ8+eZfZqJzgGQQimcv/99+vTTz9V/fr1FR4eriVLluiRRx5xdllur2vXrtqwYYMGDBigb7/9VuXLl1fDhg21bt067d27VwkJCerYsaOzy3Rbv/76qwYPHqzNmzfLYrHo/vvv16hRo/TEE084uzSXY6arneAYN79+GChjDh06pDNnzkiS2y8L70qSkpI0adIk/eMf/9CAAQPk7e2t119/Xd9//72io6O1YsUKZ5fodi5evKjPPvtMsbGxqlu3rn788UcNHjxYX375pRo3bqx+/frppZdecnaZLqdp06Zq0qSJypcvL8MwbvrgdiWQ6BGCydSvX1+///676tWrp6SkJNWvX18BAQEFtuWvxcLz9vbW+vXr1bRpU23atEk9evRQSkqKJCkh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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 600x600 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "survived_rate3(\"Title\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "924cc39c-373a-42d8-9a63-3c4eb5333d89",
   "metadata": {},
   "outputs": [],
   "source": [
    "familyDf = pd.DataFrame()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "ca94b9d1-9419-4278-b6aa-a8497bbbe76d",
   "metadata": {},
   "outputs": [],
   "source": [
    "familyDf['FamilySize']=full['Parch']+full['SibSp']+1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "b83d6380-c02b-4f7a-b1d2-2154669a6f55",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>FamilySize</th>\n",
       "      <th>Single</th>\n",
       "      <th>Small</th>\n",
       "      <th>Large</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   FamilySize  Single  Small  Large\n",
       "0           2       0      1      0\n",
       "1           2       0      1      0\n",
       "2           1       1      0      0\n",
       "3           2       0      1      0\n",
       "4           1       1      0      0"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "familyDf['Single']=familyDf['FamilySize'].map(lambda s : 1 if s==1 else 0)\n",
    "familyDf['Small']=familyDf['FamilySize'].map(lambda s :1 if 2<= s <= 4 else 0)\n",
    "familyDf['Large']=familyDf['FamilySize'].map(lambda s :1 if 5<= s else 0)\n",
    "familyDf.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "eba14d38-97d3-4f9f-a474-eebc38ff5a31",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>NaN</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1         0       3   \n",
       "1            2         1       1   \n",
       "2            3         1       3   \n",
       "3            4         1       1   \n",
       "4            5         0       3   \n",
       "\n",
       "                                                Name     Sex   Age  SibSp  \\\n",
       "0                            Braund, Mr. Owen Harris    male  22.0      1   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...  female  38.0      1   \n",
       "2                             Heikkinen, Miss. Laina  female  26.0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)  female  35.0      1   \n",
       "4                           Allen, Mr. William Henry    male  35.0      0   \n",
       "\n",
       "   Parch            Ticket     Fare Cabin Embarked  \n",
       "0      0         A/5 21171   7.2500   NaN        S  \n",
       "1      0          PC 17599  71.2833   C85        C  \n",
       "2      0  STON/O2. 3101282   7.9250   NaN        S  \n",
       "3      0            113803  53.1000  C123        S  \n",
       "4      0            373450   8.0500   NaN        S  "
      ]
     },
     "execution_count": 33,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "full = pd.concat([full,familyDf],axis=1)\n",
    "train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "76d77e77-b714-4f05-b05a-e9acfbda9233",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   FamilySize  Single  Small  Large\n",
      "0           2       0      1      0\n",
      "1           2       0      1      0\n",
      "2           1       1      0      0\n",
      "3           2       0      1      0\n",
      "4           1       1      0      0\n",
      "Index(['FamilySize', 'Single', 'Small', 'Large'], dtype='object')\n"
     ]
    }
   ],
   "source": [
    "print(familyDf.head())\n",
    "print(familyDf.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "75c7d504-5568-4827-b385-fd23d823f32b",
   "metadata": {},
   "outputs": [],
   "source": [
    "b = full.groupby('FamilySize')['Survived'].mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "002de413-c6ad-4290-a424-8d1ebc3f24ce",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: xlabel='FamilySize'>"
      ]
     },
     "execution_count": 36,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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SdL+/fv17LPP6p577tF5551nudK24fV6NWjQoID2iy66SEVFRRYqar36+nq/R6s7HA6dccYZAeulpaXpu+++68jS2swVV1yhUaNGad68eXrssce0ZMkS9evXT1dddZWGDh2qvn37qmvXrurSpYvtUlvs888/17Bhw3yvhwwZIkk6++yz/dY766yz9Pnnn3dobW3lzTffDGr9jIyMdqrkyI77QPL+++/r5z//ub799ls9++yzio6Otl1Sm+jTp4/Wrl2rkSNH+rVfdNFFKikp0W9+8xtde+21uvXWWy1V2Hrdu3fXu+++qxEjRkiSTjnlFC1ZssTvC0GSNm3aFHIPb/ztb3+riy++WDfccIP69eunv/71rxo7dmyzQTnUfPbZZxo4cKAiIyM1dOhQbd26VRdeeKHfOhs2bFDPnj0tVdg6gwcP1mOPPabMzExJUlZWlpYtW+b7jB6yePHigC+7UBIVFaW77rpLM2bM0PLly7VixQoVFhbqhx9+8K0TKn+0SVKvXr20cuVK3+/plVdekXTw/x0XXXSRb72333474P8voSKk/ui2O0FzbNixY4fp1q2b32GAULdw4ULTqVMnM3nyZFNZWRnw/oMPPmgcDofp3bt3yB2yKSwsNE6n0+Tl5Zndu3cHvP/111+bwsJC43K5zJw5cyxU2DYWLVpkYmNjzTXXXGN27dplHA6Heffdd22X1WoREREmIiLCREVFmfPOO88MHDjQuN1uU1VVZYwx5osvvjCzZs0yUVFR5r777rNcbXDeeust43K5zNChQ83SpUvNq6++arp3727Gjh1rFi1aZObPn28uvPBCExkZaVatWmW73KD89FDbT+3bt89s3rzZPPPMMyF37s8TTzxhHA6Hueiii8yYMWNMly5dTG5uromNjTVPPfWU+eijj8yiRYtMTExMSJ2P9p8+//xzk5qaak4++WSzevVqU1FRcdjFJgLJ/3nqqadMVFSU+eKLL2yX0maKiopM165dzW233dbk+48//rhxuVwhF0gOHDhg7rjjDhMdHW3ef//9gPefeeYZ43A4zMSJE0Puioaf2rFjh7n44ovNf/3Xf5mIiIiQDiTff/+92bBhg3n88cfNpEmTzM9+9jNzwgkn+M7NWrRokXE4HObaa681Xq/XcrXBe+edd8yIESN8wes/r9I4FP5LS0ttlxm0IwWSULdo0SLfFXtLliwxXq/XjBo1yndllMPhMNnZ2Wbfvn22S221UPmjm4fr/R9jjN5//30lJiYqLi7Odjltxhgjj8cjt9vd5Pu7du3S6tWrdd1113VwZUdv7969ioqKUmRkpF/7119/rfr6evXu3dtSZW1v/vz5WrFihZ544okmz00IVebgH0WKiIjQl19+Ka/XG/K/t++++07vv/++vv76azU0NCgmJkZnnHGG0tLSbJfWKm+88YYGDhyoE0880XYpHeqtt97SZ599pjPPPFMDBgywXc5RKykp0e9+9zt98sknx+zhJwIJAABhLhT+6CaQAAAA6477q2wAAAhXoXTZLzMkAACEqa5du4bMZb/MkAAAEKZC6V5bzJAAABDGDt2Q8LrrrlNxcbHtcppFIAEAIMxx2S8AALCOy34BAABaIMJ2AQAAAAQSAABgHYEEAABYRyABjhMTJ06Uw+FodlmyZEm77v+ee+7xuynT8OHDNXz48KC28e2332rq1KlKSkqS0+nUSSedpIsvvlgrVqw47L4AHPu4MRpwHDn11FP10ksvNflecnJyu+77hhtu0OjRo1vd/4cfftBFF12kffv2aebMmUpJSVFtba2WL1+u7OxsPfzww7r99tvbZF8AOh6BBDiOOJ1O/exnP7Oy7x49eqhHjx6t7l9aWqoPP/xQVVVVSklJ8bVfccUV+v777zVr1ixNnjxZkZGRR70vAB2PQzYAfA4cOKCioiKdddZZcrlcOuGEEzR06FCtXbvWt84999yjM888Uy+//LLOOussRUVFacCAAXrrrbe0YcMGnX/++XK5XDrrrLP0+uuv+/Vr7jDKr371KyUkJKixsdGvfdKkSerTp4+MMfrqq68kHbyfwk/ddddduvvuu+X1egP2VVFR0exhql69evm28fnnn+vXv/61TjrpJEVHR+viiy/WP//5z9b9IAEEjUACHGf2798fsBz6kr/zzjs1e/Zs3XzzzSovL9cTTzyhb775RtnZ2dq7d69vGzt37tTUqVOVn5+v5cuX67vvvlN2drZ+/etf68Ybb9Tzzz+vxsZGXX311frhhx+OWNP111+vf/3rX1q3bp2vzev16oUXXvCd+zJ69Gh16tRJI0eO1OzZs7Vhwwbt27dPkjRo0CBNnz69yed0nHfeeXrrrbf8ltmzZ0uSbrrpJknSN998o6FDh+rdd9/V/PnztWzZMjU2NiojI0Mffvhh63/YAFrOADguTJgwwUhqcvnTn/5kjDHmmmuuMQ8//LBfvxUrVhhJprKy0hhjzKxZs4wk87e//c23TmFhoZFknnzySV9baWmpkWT++c9/+vU7ZNiwYWbYsGHGGGMOHDhgevToYa699lrf+y+88IJxOBxmx44dfrWccsopvrpdLpcZNWqUef755/1q/um+/lNVVZWJi4szOTk5vra77rrLREVF+e3L6/WaPn36mOzs7GZ/pgDaDueQAMeR0047TStXrgxo7969uyRp6dKlkg7OGHz88ceqqqryrd/Q0ODXZ+jQob7/PvXUUyXJ7/yUbt26SZJqamqOWFdERIQmTpyoP//5z/rrX/+q6OhoPf300xoxYoQSExN9640bN06XX3651q5dqzVr1qiiokJr1qzRq6++qtLSUi1fvvywV9fU1NQoKytLvXv3VklJia/99ddf14ABA9S9e3ft37/fV9Oll17a7lcfATiIQAIcR7p06aL09PRm33/nnXf0u9/9Tps2bZLL5VK/fv18gcD85NyN2NjYgP5H82jz3Nxc3X///SorK1NmZqZee+01Pf300wHrde7cWaNGjdKoUaMkSf/+9781efJklZaWatWqVRozZkyT2z9w4ICuvvpq7dmzR6+99ppfrd9++622b9+uzp07N9n3+++/P6Yf2w6EAwIJAEmSx+PR6NGjdfbZZ+t//ud/lJaWpoiICK1evTrgPh/toXfv3ho+fLiWL1+umpoanXDCCRo3bpzv/aFDhyo1NdVvZkM6OOuzePFirVixQlu3bm02kEyfPl1r167V2rVr1bNnT7/34uLiNGzYMM2dO7fJvk6n8yhHB+BIOKkVgCTpo48+0rfffqspU6aoX79+iog4+L+Hv/3tb5IUcAVMe7j++uu1Zs0aLVmyRDk5OX6zEr1799aLL76o6urqgH5VVVWSpP79+ze53ZKSEv35z3/WX/7yF1144YUB7w8bNsx3OXF6erpvWbJkiRYvXqzIyMg2GiGA5jBDAkCSlJqaqtjYWN1///3q1KmTOnfurNLSUj355JOS5HeVTXv55S9/qd///vd6++23NW/ePL/37r//fq1bt06DBw/WlClTNGTIEEVGRmrTpk2aO3euLr300iZvhrZhwwZNmjRJV155pdLT0/X222/7HX4699xzNXXqVD333HPKzMzU9OnT1a1bN73wwgtatGiRHn744XYfNwACCYD/43a79d///d+aMWOGfvWrXykmJkbnnnuu3nzzTV166aVav369Lr/88natISoqShdffLE++OADv5NmJalXr17avHmzCgsLtXTpUj3wwAMyxuiMM87QjBkzNGXKlCZPaC0vL1dDQ4NeeumlJu9S++mnn6pXr16qrKxUXl6eJk2apB9//FEpKSl68sknlZub227jBfD/HOanZ6oBgCU//PCDEhISlJeXp2nTptkuB0AHIpAAsO6zzz7TM888ozVr1mjr1q2qrq6W2+22XRaADsQhGwDWRURE6JFHHtGJJ56o559/njACHIeYIQEAANZx2S8AALCOQAIAAKwjkAAAAOsIJAAAwDoCCQAAsI5AAgAArCOQAAAA6wgkAADAOgIJAACw7n8BLwsZrUMDdoAAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "b.plot.bar(color='black')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "d44274b3-b398-4343-b24f-bc5cfcfc3043",
   "metadata": {},
   "outputs": [],
   "source": [
    "full['Cabin'] = full['Cabin'].fillna( 'U' )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "a7bca3a3-71ea-485e-9885-a576d0529175",
   "metadata": {},
   "outputs": [],
   "source": [
    "sex_mapDict={'male':1,\n",
    "            'female':0}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "298ea43e-523b-4459-a86a-96ba03e99483",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Title</th>\n",
       "      <th>FamilySize</th>\n",
       "      <th>Single</th>\n",
       "      <th>Small</th>\n",
       "      <th>Large</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>1</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>U</td>\n",
       "      <td>S</td>\n",
       "      <td>Mr</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>0</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>U</td>\n",
       "      <td>S</td>\n",
       "      <td>Miss</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>U</td>\n",
       "      <td>S</td>\n",
       "      <td>Mr</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1       0.0       3   \n",
       "1            2       1.0       1   \n",
       "2            3       1.0       3   \n",
       "3            4       1.0       1   \n",
       "4            5       0.0       3   \n",
       "\n",
       "                                                Name  Sex   Age  SibSp  Parch  \\\n",
       "0                            Braund, Mr. Owen Harris    1  22.0      1      0   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...    0  38.0      1      0   \n",
       "2                             Heikkinen, Miss. Laina    0  26.0      0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)    0  35.0      1      0   \n",
       "4                           Allen, Mr. William Henry    1  35.0      0      0   \n",
       "\n",
       "             Ticket     Fare Cabin Embarked Title  FamilySize  Single  Small  \\\n",
       "0         A/5 21171   7.2500     U        S    Mr           2       0      1   \n",
       "1          PC 17599  71.2833   C85        C   Mrs           2       0      1   \n",
       "2  STON/O2. 3101282   7.9250     U        S  Miss           1       1      0   \n",
       "3            113803  53.1000  C123        S   Mrs           2       0      1   \n",
       "4            373450   8.0500     U        S    Mr           1       1      0   \n",
       "\n",
       "   Large  \n",
       "0      0  \n",
       "1      0  \n",
       "2      0  \n",
       "3      0  \n",
       "4      0  "
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "full['Sex']=full['Sex'].map(sex_mapDict)\n",
    "full.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "99a15c3a-9692-48f5-9ca6-177ad1336e7f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAjkAAAHtCAYAAAD7rG/JAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAAA/xUlEQVR4nO3dfXyP9f////trs5PX2ImTaU6HYYiEkb2T4S28pTlJUpGz+qISKQ0jJ2G8pxOF1FtUUiz2dpISyXJOOe2TECIi5GQbsrE9f3+47PV7rw17se21HbtdL5ddLr2ex/E8jsexvV697p7H8TwOmzHGCAAAwGLcXF0AAABAXiDkAAAASyLkAAAASyLkAAAASyLkAAAASyLkAAAASyLkAAAASyLkAAAASyLkAAAASyLkAC6wbNkydejQQWXLlpWXl5fKlSunjh07atmyZa4uTb1795bNZtOuXbvybB9HjhyRzWZTp06dbrluixYtZLPZsvy4ubnJz89PderU0csvv6wLFy7cUU2nTp3SRx99dEfb+LsPP/xQNptNb731Vq5u92YK8nsLyG/FXF0AUNQMGjRI06dPV3BwsCIjI1WmTBmdOHFCK1as0LJly9S/f3/NmjXLZfV16tRJVapUUVBQkMtqyM7gwYMVEBDgeG2M0bFjx/TVV1/p9ddf15o1a7R582Z5e3s7ve3Tp08rNDRULVq0UK9evXKx6vxV0N9bQH4j5AD5KCEhQdOnT1enTp0UFxcnDw8Px7LExES1bNlS7733ntq3b6/IyEiX1NipU6ccjbDktyFDhqhKlSpZ2hMTE/XAAw9o165d+uijj9S/f3+nt3358mUlJibmQpWuUxjeW0B+43QVkI+++OILSddHJf73S0iS/P39NXnyZEnS4sWL8722wsrf319Dhw6VJH377bcursZ1eG8BWRFygHx09epVSdJPP/2U7fIHHnhAcXFxevHFFx1tY8eOlc1m05IlS7KsX6VKlUyncDKuAYmLi1OrVq3k5eWl4OBgPfzww7LZbFq1alWWbWzZskU2m03PPfecpMzX5Jw+fVoeHh76xz/+kW297dq1U7FixfTHH384jm/atGlq2rSp/P395enpqeDgYA0YMECnTp3K0e/odgQGBkqSrly5kqn9yJEjGjBggEJCQuTt7a0SJUqoUaNGmjlzpmOdDz/8UFWrVpUkLV26VDabTR9++KFj+cGDB9WjRw/ddddd8vLyUu3atRUTE+P4W+ZEenq6XnvtNVWqVEl2u11NmjTRokWLHMsvX74sPz8/Va5cWcaYLP379u0rm82mw4cP33Aft/Pekq6f9ps1a5YaNmwou92ukiVLKjIyUjt37nSsk5qaqrp168pms2np0qWZ+o8fP142m03/7//9v1v/IoD8ZgDkm+XLlxtJxsvLywwePNhs2bLFXLt27aZ9xowZYySZ//73v1mWBQcHG39/f8fruXPnGkmmbNmypkGDBmbYsGHmkUceMVu2bDGSTN++fbNsY/DgwUaS2bhxozHGmF69ehlJZufOncYYY9q1a2dsNps5evRopn5nzpwxxYoVMw8++KCjrUuXLkaSadasmXn55ZfN888/b2rWrGkkmQYNGjjW+/XXX40k07Fjx1v8xoyJiIgwksyvv/56w3Wef/55I8mMGjUq0z5Kly5t7Ha7efLJJ83w4cNNz549jd1uN5LMtGnTjDHG7Ny50/E7CA0NNWPGjHEc+/bt242/v7/x9PQ03bt3N1FRUeb+++83kky7du1MWlraTWvP+HsEBQUZu91unn76adO/f39TunRpI8m8++67jnUzfu/r1q3LtI2//vrL+Pv7m/vvv/+m+7qd95YxxvTs2dNIMnXr1jVDhgwxzzzzjPH39zfe3t5mzZo1jvW2bdtm3N3dTeXKlc3FixeNMcbs3r3beHh4mGrVqpnk5ORb7gvIb4QcIJ8NHDjQSHL8+Pn5mfbt25s333zTHDt2LMv6txNyKlasaC5dupRp3Ro1apiSJUua1NRUR1taWpopX768qVq1qqPt7yHnk08+MZLM1KlTM23v3XffNZLMhx9+aIwxZvPmzUaSeeKJJzKtd/XqVXPvvfcaSWbfvn3GmNwJOampqebIkSNmwoQJxs3NzQQEBJhTp045lvfv399IMqtWrcrU7/vvvzeSTNOmTR1t2dWTnp5u6tata+x2u+N3keGll14ykszMmTNvWnvG36NYsWLmhx9+yLS/cuXKmeLFi5vz588bY4xZs2aNkWQGDBiQaRtxcXFGkpk1a9ZN92WM8++tjG336NEjUyDKCIgVK1bM9H6JiooyksywYcMcf1c3NzezYcOGW9YGuAIhB3CBpUuXmgcffNB4eHhk+lLy8PAwI0eOzDRCcDsh55lnnsmy7tixY40ks2LFCkdbQkKCkWSio6MdbX8POZcuXTIlSpQwjRs3zrS9Fi1aGLvdbpKSkowxxhw7dsx8+OGH5tChQ1n2PWjQICPJrF+/3hhzeyHnZj/16tUz27Zty9Rv/fr15oMPPsh2m76+viYkJMTxOrt6MkLb888/n6X/xYsXjaenpwkLC7tp7Rl/j969e2dZ9u9//9tIMnPnzjXGXA9VlStXNmXKlDFXr151rNexY0fj6elpzp07d9N9ZXDmvdW2bVsjyZw5cybLdqKjo40k88UXXzjarly5YmrXrm08PDxM3759jSQzfPjwHNUFuAKzqwAXiIyMVGRkpJKTk7V+/XqtWbNGy5Yt08GDBzVp0iRJ0sSJE297+xnXmPyvHj16aOzYsYqLi1P79u0lSQsWLHAsuxEfHx916tRJn3zyiY4cOaIqVaro5MmTWrdunR599FH5+vpKkipWrKhevXrp2rVr2rFjh/bv36+DBw9q586djguC09LSbvuYMqaQG2N06NAhxcXFydPTU7Nnz1b37t2zrN+sWTM1a9ZM586d065du3Tw4EHt27dPW7du1cWLF1W6dOmb7m/79u2Srl+TM3bs2CzLfX19tXv3bhljZLPZbrqt+++/P0vbfffdJ0navXu3JMlms+nJJ59UTEyMVq1apfbt2+v8+fP66quv1KFDB5UsWfKm+8jgzHtr+/bt8vb21vTp07NsZ9++fZKkXbt26aGHHpIkeXl5ac6cObr//vs1Z84c1a9fX+PGjctRXYBLuDplAbguPT3dfPDBB8bNzc3Y7XZz+fJlY8ztjeS8+eab2e6jadOmJiAgwKSkpJirV6+awMBA07Bhw0zr/H0kxxhjVq5caSSZKVOmGGOMeeutt4wks2zZskx9Z82aZcqXL+8YPShVqpRp166dCQ8PN5LM2rVrjTG5c7pq8+bNxsfHx/j4+DhGiP7XuXPnTK9evRwjGjabzYSEhJh+/fqZ4sWLm+DgYMe62dUzYcKEW44gSXKMZGUn4+/xv6MhGXbs2JFl1O3nn392nD7K+H1KMvHx8bf8Pd3Mjd5bxYoVu+XxDR06NNO2UlNTTbVq1Ywk069fvzuqC8hrzK4C8klSUpJq1KihDh06ZLvcZrOpb9++evDBB/XXX3/p2LFjjnZJ2c66uXz5slM19OjRQxcuXNCqVav07bff6syZM3ryySdv2a9169YKCgpSXFycJGnhwoUqXbq02rVr51jn888/14ABA1S6dGnFx8fr999/19mzZ/XVV1+pYcOGTtWZE02bNtWMGTN0+fJlde7cWSdPnsy0vEePHvroo4/Uu3dvbdy4UcnJyTp48KBmz56do+2XKFFCkvTBBx/IXD+1n+1PxkjWzVy8eDFL24kTJyQp0whNrVq11LhxYy1dulSpqamKi4tTqVKlHCMpN3K7760SJUqoUqVKNz2+119/PdO2Jk6cqMOHD6tUqVKaM2eO1q5de8vjB1yFkAPkEz8/PyUmJuqbb7656XRqY4zc3Nwcdxz29PSUlPWL8sKFC/rzzz+dqqF79+7y8PDQ0qVL9fnnn8vNzS3bUz1/5+7uru7du2v79u3asmWLtmzZokcffTTT/Vjmz58vSfrss8/UuXNnlS9f3rEsY1pzdkHtTvTu3VsdO3bUn3/+mWkK84ULF/Tll18qLCxM77//vv7xj3+oePHikqSjR4/q0qVLmWrJ7nRT/fr1Jf3/p63+19WrV/XSSy/pnXfeyVGd2W1j48aNkqRGjRplan/qqaeUnJyspUuXav369erWrZvjPXAjt/veql+/vo4fP55tny+++EKjRo1ynE6TpD179mjSpEmqW7euNmzYIC8vLz399NO6dOnSTesDXCafR46AIm3cuHGOKdYnTpzIsnzp0qXGzc3NPProo462+Ph4I8l069Yt07ovv/yykeTU6SpjjOnQoYMJCgoyd911l2ndunWW5dmdrjLGmB9++MFIcsyU+vuMmu7duxtJmaYdG2PMRx995Dj18fXXXxtjcncK+e+//278/PyMJLNw4UJjjDGXL1827u7uJiQkxKSkpDjWvXz5snnooYeMJFOuXDlH+/Hjx40k0759e0fbtWvXTLVq1Yynp6fZsmVLpn2OHz/eSDI9e/a8ae0Zf4+AgADzyy+/ONp//vln4+fnZ8qUKeM4dZThzz//NB4eHiY4ODjT1P5buZ33VkZ9jz76aKbf04kTJ0zFihWNzWZz1H316lXToEEDY7PZHDVl/B5eeOGFHNUI5DdCDpCPrl27Zrp27WokGR8fH9O5c2cTFRVlhg4d6rj/Su3atTPNdklJSXFc59KuXTvzyiuvmGbNmpmAgABTr149p0POggULHKEjY2bP/7pRyDHGmNq1axtJpkqVKiY9PT3Tsi+++MIxbbl///5m2LBhjoBStmxZI8l8+umnxpjcv0/O22+/baTr96PJmJL96KOPGkmmfv36ZtiwYWbgwIGmYsWKplixYqZkyZLGbrc7ZhqlpKQYLy8vY7fbzdChQx3X+Kxfv94UL17cFCtWzHTt2tW88sorplWrVkaSCQ4ONsePH79p7Rl/j+rVq5uSJUuaZ5991vTt29f4+fmZYsWKmeXLl2fbr2PHjkaSqVat2i1/Pxlu572Vnp5uOnXqZKTr9wgaNGiQGThwoOM+PhMmTHCsmxGi+vfv72hLSUkxoaGhxmazZXtdFOBqhBzABeLj402XLl1MxYoVjbe3t/Hz8zONGjUyMTExWf5lb4wxBw4cMJ07dzZ+fn7G19fXtG/f3vz000+mY8eOToecv/76y/j5+Rlvb2+TmJiYZfnNQs7EiRONJDNy5Mhst71gwQLTsGFDU7x4cRMYGGjuu+8+M2PGDMdFthkX1OZ2yElLSzNNmjTJdCFvYmKiGTJkiAkODjbe3t6matWqpnPnzmbbtm1myJAhRpL55ptvHNuYPXu2KV++vPHy8jJjx451tP/000/m8ccfN2XLljWenp4mJCTEDBo0yJw8efKWtWf8PRYvXmxeeeUVU7ZsWWO3201ERMRNQ8H8+fONJDNmzJhb7uPvnH1vXbt2zUybNs3ce++9xm63m1KlSpkHHnjALF682LHOnj17jKenZ6YQmWHt2rVGkqlRo0a22wdcyWZMLp8kBwDckaioKMXGxuqXX35RSEiIq8sBCi1CDgAUIMePH1ejRo1Ur149ffPNN64uByjUCtTsqmPHjikgIEAJCQm3XPeTTz7R3XffLbvdrtDQ0BxPCwWAgmj+/Plq0KCBQkNDdebMGb366quuLgko9ApMyDl69KgefPBBJSYm3nLdzz//XE899ZTatGmjJUuWqFWrVnrmmWccU1gBoLCpWLGifvvtN/n7++u9995T8+bNXV0SUOi5/HRVenq6PvroI7388suSpHPnzmnt2rVq0aLFDfuEhoaqfv36jhuTSdJjjz2m7du36+DBg3ldMgAAKARcPpKzZ88eDRw4UL169dK8efNuuf6RI0d04MABdenSJVN7165ddejQIR04cCCvSgUAAIWIy0NO5cqVdfDgQb3xxhvy8fG55fo///yzJKlmzZqZ2qtXry5JhBwAACBJcvlTyEuVKqVSpUrleP0LFy5Iun4b8/+V8fyYpKSkbPulpKQoJSXF8To9PV3nzp1T6dKlb/kEYQAAUDAYY5ScnKzy5cvLze3mYzUuDznOSk9Pl5T1WTMZlxbd6IBjYmI0bty4vC0OAADki2PHjqlixYo3XafQhZyAgABJWUdsMh5e6O/vn22/ESNGaOjQoY7XiYmJqly5so4dO5ZlVAgAABRMSUlJqlSpkuMMzs0UupATGhoqSTp48KAaNGjgaM+YVVWnTp1s+3l5ecnLyytLu5+fHyEHAIBCJieXmrj8wmNnVa9eXdWqVdOiRYsytS9atEg1a9ZUcHCwiyoDAAAFSYEfyUlKStLevXsVEhKiwMBASdLo0aPVp08flS5dWpGRkVq2bJni4uK0cOFCF1cLAAAKigI/krNjxw6Fh4drxYoVjrbevXtr1qxZWr16tTp16qSEhAR9/PHH6tatmwsrBQAABYnL73jsKklJSfL391diYiLX5AAAUEg48/1d4EdyAAAAbgchBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWBIhBwAAWFKBCDkrV65UWFiYfHx8FBwcrJiYGBljbrj+tWvXNHnyZNWoUUPFixfXvffeq4ULF+ZjxQAAoKBzecjZtGmTIiMjVbt2bcXHx6tnz56Kjo7WpEmTbthn7Nixio6OVo8ePbR06VKFh4ere/fuWrRoUT5WDgAACjKbudmQST5o27atzp8/r23btjnaoqKiNHPmTJ0+fVp2uz1Ln/Lly+uf//yn5s2b52hr2rSp7Ha71q5dm6P9JiUlyd/fX4mJifLz87vzAwEAAHnOme9vl47kpKSkKCEhQV26dMnU3rVrV128eFHr16+/Yb+/H1iZMmV09uzZPKsVAAAULi4NOYcPH1Zqaqpq1qyZqb169eqSpAMHDmTbb+jQofr444+1cuVKJSUlaf78+Vq5cqV69uyZ5zUDAIDCoZgrd37hwgVJyjIq4+vrK+n6kFR2Bg0apPXr1+tf//qXo61v374aNmzYDfeVkpKilJQUx+sbbRsAAFiDS0NOenq6JMlms2W73M0t60BTSkqKHnjgAf3xxx+aNWuWatWqpQ0bNmjixIkqUaKEpk2blu22YmJiNG7cuNwrHgAAFGguDTkBAQGSso6qJCcnS5L8/f2z9Fm8eLH27Nmj1atXq3Xr1pKkiIgIBQQE6Pnnn9fTTz+tevXqZek3YsQIDR061PE6KSlJlSpVyq1DAQAABYxLQ05ISIjc3d118ODBTO0Zr+vUqZOlz9GjRyVJ999/f6b2iIgISdLevXuzDTleXl7y8vLKlboBAEDB59ILj729vdW8eXPFx8dnuvnfokWLFBAQoCZNmmTpU6tWLUnKMvNq48aNkqSqVavmYcUAAKCwcOlIjiSNGjVKrVu3Vrdu3dS3b19t2rRJsbGxmjJliux2u5KSkrR3716FhIQoMDBQkZGRuu+++9SjRw+NGzdOtWrV0tatWzVhwgQ9/PDD2QYjAABQ9Lj8ZoCS9N///ldjxozR/v37VaFCBT333HN66aWXJEkJCQlq2bKl5s6dq969e0u6fj1NdHS0Fi9erHPnzqlatWp66qmnNHToUHl6euZon9wMEACAwseZ7+8CEXJcgZADAEDhU2jueAwAAJBXCDkAAMCSCDkAAMCSCDkAAMCSCDkAAMCSCDkAAMCSCDkAAMCSCDkAAMCSCDkAAMCSCDkAAMCSCDkAAMCSCDkAAMCSCDkAAMCSCDkAAMCSirm6AOQ/m83m6hKQj4wxri4BAFyCkRwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJhBwAAGBJxW6341dffaXVq1frxIkTiomJ0c6dO9WoUSMFBwfnZn0AAAC3xemRnMuXL6tNmzZ66KGHNGfOHH3++ec6f/683n33XTVq1Eg//fRTXtQJAADgFKdDzsiRI7V9+3atWbNGf/75p4wxkqR58+apQoUKGj16dK4XCQAA4CynQ87ChQsVExOjli1bymazOdqDgoI0atQobdiwIVcLBAAAuB1Oh5wLFy6oSpUq2S4rWbKkLl68eKc1AQAA3DGnQ07dunU1f/78bJctX75cdevWdbqIlStXKiwsTD4+PgoODlZMTIzjNNiNrFixQk2aNJHdblfFihU1ePBgXbp0yel9AwAAa3J6dtWoUaPUuXNnnT17Vg8//LBsNpu+++47zZ07V7NmzdJnn33m1PY2bdqkyMhIPfbYY5owYYI2bNig6OhopaenKzo6Ots+y5cvV6dOnfTUU09p8uTJ2rt3r0aOHKkzZ87o008/dfaQAACABdnMrYZMsvHpp59q+PDhOn78uKOtbNmymjhxovr16+fUttq2bavz589r27ZtjraoqCjNnDlTp0+flt1uz7S+MUbVq1dXo0aNFBcX52ifNm2a3n77bf3444/y8fG55X6TkpLk7++vxMRE+fn5OVVzYfe/11LB+m7jIw4ABZYz39+3FXIy7N+/X2fPnlVAQIBq1aolNzfnzn6lpKTIz89P48aN0/Dhwx3t33//vZo0aaKvv/5abdq0ydRn586datiwodatW6cHHnjgdksn5KDIIOQAsBJnvr+dvianVatW2rdvnyQpNDRU//jHP1SnTh25ublpz549uueee3K8rcOHDys1NVU1a9bM1F69enVJ0oEDB7L02bVrlyTJbrerQ4cOstvtKlmypAYNGqQrV644ezgAAMCicnRNzoYNG5Seni5JSkhI0HfffafTp09nWe+LL77QoUOHcrzzCxcuSFKWJObr6yvpelr7uzNnzkiSOnfurCeeeEIvvfSSvv/+e40ZM0anT5/WwoULs91XSkqKUlJSHK+z2zYAALCOHIWc2bNn6+OPP5bNZpPNZtOzzz6bZZ2MIfEnnngixzvPCE43On2S3emv1NRUSddDzpQpUyRJLVu2VHp6ukaMGKHx48crNDQ0S7+YmBiNGzcux7UBAIDCLUchZ9q0aerTp4+MMWrVqpVmzJihOnXqZFrH3d1dAQEBuvvuu3O884CAAElZR1WSk5MlSf7+/ln6ZIzydOjQIVN7u3btNGLECO3atSvbkDNixAgNHTrU8TopKUmVKlXKca0AAKBwyVHI8ff3V0REhCRp7dq1atSokUqUKHHHOw8JCZG7u7sOHjyYqT3j9d+DlCTVqFFDkjKdepKkq1evSlKW2VgZvLy85OXldcc1AwCAwsHp++RERETo+PHjWrFihVJTUx2nqdLT03Xp0iWtX79eCxYsyNG2vL291bx5c8XHx+vll192nLZatGiRAgIC1KRJkyx9mjdvruLFi+uzzz7Tww8/7GhftmyZihUrpvDwcGcPCQAAWJDTIefzzz9Xjx49dPXqVUcoMcY4/rtWrVpObW/UqFFq3bq1unXrpr59+2rTpk2KjY3VlClTZLfblZSUpL179yokJESBgYEqUaKExo8fr5deekklS5ZUly5dtGnTJk2ZMkWDBw9WYGCgs4cEAACsyDjp3nvvNffdd5/ZsWOH6devn3nqqafM3r17zdSpU42Xl5dZtWqVs5s08fHxpl69esbT09NUrVrVTJ061bFs7dq1RpKZO3dupj5z5swxd999t/H09DRVqlQxkyZNMmlpaTneZ2JiopFkEhMTna63sJPETxH6AQArceb72+mbAfr4+Gj+/Pnq3LmzPvvsM/373//Wzp07JV2/U/HWrVuVkJDgzCZdgpsBoqhw8iMOAAVant4M0M3NTaVLl5Yk1axZU/v27XNMBW/Xrp327t17GyUDAADkLqdDTu3atbVhwwZJ12c6paamOu5CfP78+SyzngAAAFzB6QuP+/fvrwEDBujixYuaNGmSWrZsqb59+6pfv36aPn26GjVqlBd1AgAAOMXpkZynn35a06ZNc9x5+P3339eVK1c0ePBgXb16VW+99VZu1wgAAOC0O3oKeQZjjP78889CNX2bC49RVHDhMQArydMLj7Njs9kUGBioo0ePqkuXLrmxSQAAgDuSo5CTlpam6OhoBQUFKSgoSFFRUUpLS3MsT01N1fjx41WnTh0tXbo0z4oFAADIqRyFnNdee00xMTGqVq2aGjRooKlTpzqeAL5hwwbdfffdGjt2rCpUqKDly5fnacEAAAA5kaPZVXFxcXryySc1b948SdIbb7yhd955R3Xr1tWjjz4qT09PTZ48WS+++KI8PDzytGAAAICcyNGFxyVKlFBcXJzat28vSfrjjz9Uvnx5+fv7q3HjxpozZ44qVqyY58XmJi48RlHBhccArMSZ7+8cjeRcvnxZZcqUcbzOuONxy5YttXjxYr40AQBAgXNbs6vc3K53Gzx4MAEHAAAUSHc0hdzX1ze36gAAAMhVOQ452Y3YMIoDAAAKqhw/u6pp06ZZ2sLCwrK02Ww2Xbt27c6qAgAAuEM5CjljxozJ6zoAAAByVa48u6owYgo5iooi+hEHYFH5/uwqAACAgoaQAwAALImQAwAALImQAwAALImQAwAALClHU8jHjx+f4w3abDaNHj36tgsCAADIDTmaQp7xrKocbdBmU1pa2h0VlR+YQo6iginkAKwk159Cnp6eniuFAQAA5JdcvyYnMTExtzcJAADgtBw/uypDSkqK3nzzTX333XdKTU11DIWnp6fr0qVL+umnn3T58uVcLxQAAMAZToecV155Re+8847q1aun06dPy263KzAwUD/++KNSU1M1duzYPCgTAADAOU6frlq8eLFefPFF7d69Wy+88ILCwsK0detW/fLLL6pSpQrX7wAAgALB6ZBz+vRpPfTQQ5Kke+65R9u2bZMkVahQQSNGjNCCBQtyt0IAAIDb4HTICQgIUEpKiiSpZs2aOnbsmJKTkyVJNWrU0G+//Za7FQIAANwGp0POAw88oLfffluXLl1S1apVVbx4ccXHx0uSNm/eLH9//1wvEgAAwFlOh5wxY8Zo8+bN6tChg4oVK6Znn31W/fv3V6NGjTRq1Cg98sgjeVEnAACAU5yeXXXPPfdo3759+vHHHyVJMTEx8vPz08aNGxUZGakRI0bkepEAAADOytFjHf7X9u3b1ahRo7yqJ9/wWAcUFTzWAYCVOPP97fTpqsaNG6tOnTqaPHkyFxkDAIACy+mQs2LFCjVq1EgxMTGqVq2aWrZsqblz5zpmWAEAABQEToecf/3rX5o3b55OnTqlTz/9VP7+/ho4cKDuuusuPf7441qxYkVe1AkAAOAUp6/Jyc6FCxc0ZswYzZw5U+np6UpLS8uN2vIU1+SgqOCaHABW4sz3t9Ozq/7X999/rwULFujzzz/X8ePH1ahRI/Xs2fNONgkAAJArnA45P/74oxYsWKCFCxfq119/VcWKFdWjRw/17NlTtWvXzosaAQAAnOZ0yKlfv758fX31yCOP6D//+Y9atmyZF3UBAADcEadDzvz589W5c2d5e3vnRT0AAAC5Ikch57ffflO5cuXk4eGh+++/X6dPn77p+pUrV86V4gAAAG5XjkJO1apVtXnzZjVp0kRVqlS55eycwjC7CgAAWFuOQs6cOXMUEhLi+G+mIAMAgILO6fvknD59WmXLls2revIN98lBUcF9cgBYSZ4+u6pChQp66KGHtGDBAl25cuW2iwQAAMhLToect99+W8nJyXryySd11113qU+fPlq7dm1e1AYAAHDbbvuxDseOHdNnn32mzz77TLt371aFChXUo0cPPfnkk6pbt25u15nrOF2FooLTVQCsxJnv71x5dtW+ffs0Y8YMzZo1i2dXFQKEnKKFkAPASvLt2VWnTp1SXFyc4uLitHnzZgUGBurxxx+/k00CAADkCqdDzrlz57R48WItWLBA69atk6enpzp27KiRI0eqbdu2cnNz+jIfAACAXOd0yAkKClJaWpoiIiL0/vvv69FHH1WJEiXyojYAAIDb5nTIGTt2rHr06MGjGwAAQIHm9LmlWbNmKSEhIQ9KAQAAyD1Oh5xr164pMDAwL2oBAADINU6frnrttdc0aNAgRUdHq27durrrrruyrMOpLAAA4GpO3yfHw8PDcR+cG91vhfvkFGzcJ6do4T45AKwkT++TM3v27NsuDAAAIL84HXJ69eqVF3UAAADkKqdDzrp16265TvPmzZ3a5sqVKzVq1Cjt3btXgYGBGjBggIYPH56j0yrXrl1TeHi4ihcvzqwvAADg4HTIadGihWw2W6bz/H8PI85ck7Np0yZFRkbqscce04QJE7RhwwZFR0crPT1d0dHRt+w/efJk/fDDD4qIiMj5QQAAAMtzOuSsXbs2S9vFixe1YcMGzZs3T4sWLXJqe+PGjdO9996refPmSZLatWunq1evavLkyRo6dKjsdvsN++7evVuTJk1SUFCQcwcBAAAsL1eeQp5h4sSJ2rx5s7744oscrZ+SkiI/Pz+NGzdOw4cPd7R///33atKkib7++mu1adMm275Xr15V48aN1a5dO23ZskWSnDpdxewqFBXMrgJgJc58f+fq0zSbNWuW7UjPjRw+fFip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B1WqVCmfKwKQm06dOqV7770322UVK1bU+fPn87cgFEiEHFhSv379NHHiRE2dOlW//PKLJOnixYtavHixJk2apN69e7u2QAB3pHr16vryyy+zXZaQkKDq1avnc0UoiJhdBUuKiorSr7/+qqioKEVFRUmSWrZsKUl68sknNWLECFeWB+AODRkyRP3791dqaqoefvhh2Ww2/fLLL1q7dq2mTp2qN954w9UlogDgKeSwtAMHDujbb7/VuXPnFBAQoIiICN19992uLgtALoiJidHEiRP1119/OS489vT01CuvvKLx48e7uDoUBIQcAEChlZSUpE2bNjn+IdO0aVOVKlXK1WWhgCDkwDJatWqV43VtNpvWrFmTh9UAAFyNa3JgGenp6Tl+XAPZHih8qlatmuPPuM1m06FDh/K4IhR0hBxYRkJCgqtLAJCHIiIieO4cnMLpKhQ5ly5d0vr169WuXTtXlwIAyEOM5MCSjh49qv79++u7775TampqtuukpaXlc1UActupU6eUmprqOAWdnp7u+IfMgAEDXFwdXI2RHFhSly5d9M0336hPnz7auHGjfHx8FB4erlWrVunHH39UfHy8IiMjXV0mgNu0e/duPf7449q/f3+2y202m65du5bPVaGg4Y7HsKTvvvtOEyZM0LRp09SnTx95eXlpypQp+uGHHxQREaGlS5e6ukQAd2DYsGE6f/68pk6dqhYtWqht27aaPn262rdvL5vNxjV6kETIgUVdvHjR8VybOnXqaNeuXZIkd3d3Pffcc/r2229dVxyAO7Z161ZNmDBBL774orp3766LFy9q4MCBWr58uTp16qS3337b1SWiACDkwJLKlSunP/74Q9L1Z9ycO3dOJ0+elCSVKlWKp5ADhVxKSopq1qwpSapVq5b27NnjWNanT58bPqAXRQshB5b00EMPafTo0dq0aZMqVaqkihUraurUqUpOTtacOXNUoUIFV5cI4A5UrlxZhw8fliTVqFFDSUlJOnLkiCTJy8tL586dc2F1KCi48BiWdPbsWbVv316+vr765ptvNH/+fPXq1csxA2PGjBnMvAAKsREjRmju3LmaPn26unbtqjp16qhBgwYaPny4XnnlFf3++++ZRndQNBFyYGknT55UuXLlJEkbNmzQ5s2b1aRJE0VERLi4MgB34sqVK+rZs6cuXbqkL7/8Ul9//bU6d+6sK1euqFixYlqwYIG6dOni6jLhYoQcAEChdfXqVXl4eEiSDh8+rO3bt6tRo0aqVq2aiytDQcDNAGFJqampeuedd7Rx40ZduHAhy3Ie0AkUbufPn9err7560884z64CIQeW9Nxzz+mDDz5Q3bp1Vbp06SzLGcAECrdnnnlGS5cu1b/+9S/Vr1/f1eWggOJ0FSypTJkyGjhwoF577TVXlwIgDwQEBGjcuHEaPHiwq0tBAcYUcliSm5ubmjdv7uoyAOQRX19fhYaGuroMFHCEHFhSr169NGfOHKWnp7u6FAB54Pnnn9frr7+uixcvuroUFGCcroIlXblyRQ0aNNDVq1cVFhYmHx+fTMttNps++OADF1UH4E5dunRJYWFhOnnypGrVqpXtZ5zJBeDCY1jS8OHDtX//fhUvXlxbt27Nstxms7mgKgC5pX///tq/f79CQ0Nlt9uzTCbg3++QGMmBRZUuXVq9evVSbGys3N3dXV0OgFzm5+en6OhoRUVFuboUFGBckwNLSktLU2RkJAEHsCgvLy81btzY1WWggCPkwJI6d+6suLg4V5cBII889dRTevfdd5lcgJvimhxYUtOmTTV8+HDt3r1b4eHh8vPzy7TcZrNp9OjRLqoOwJ3y9/fX6tWrVaVKFTVp0iTbzziTC8A1ObAkN7ebD1LabDalpaXlUzUAclvVqlVvutxms+nw4cP5VA0KKkIOAACwJK7JgeUlJiZq3759SklJYfQGAIoQQg4sKyEhQffdd59KlSqlunXr6qefftITTzyhl156ydWlAQDyASEHlvTtt9+qTZs2stvtmjJliuPGYA0aNNC0adP0xhtvuLhCAEBe45ocWFJ4eLgqVaqkuLg4Xbt2TZ6envrhhx/UsGFDjRw5UkuWLNHevXtdXSYAIA8xkgNL2rVrl/r27Ssp6yMc2rRpoyNHjrigKgBAfiLkwJL8/f118uTJbJf99ttv8vf3z+eKAAD5jZADS+rYsaOio6P1ww8/ONpsNpuOHz+uSZMmqUOHDi6sDgCQH7gmB5Zx5coVeXt7S5LOnz+vli1b6scff1RQUJBOnjypmjVr6tixY6pcubLWr1+vMmXKuLhiAEBeIuTAMoKCgvTf//5X4eHhGj9+vPr166cvv/xS3377rc6ePauAgABFRESoT58+8vHxcXW5AIA8RsiBZdjtdn3yySd65JFH5O7uri1btvCUYgAowgg5sIyIiAht2bJF5cuX19GjR1WuXDl5eXllu67NZtOhQ4fyuUIAQH7iKeSwjE8//VRvvfWWzp49q48++kgNGjRQYGCgq8sCALgIIzmwpKpVq2rJkiWqX7++q0sBALgIIQcAAFgS98kBAACWRMgBAACWRMgBAACWRMgBUGj83//9n7p3766goCB5enqqXLlyeuyxx7Rz505XlwagACLkACgUfvrpJ4WHh+vMmTN6++23tXr1ak2dOlVHjx5VeHi4tmzZ4uoSARQwzK4CUCj069dP33zzjQ4ePCgPDw9H+6VLl1SrVi3dc889WrFihQsrBFDQMJIDoFD4448/JEl//3dZ8eLF9eabb6pbt26OtqVLlyosLEze3t4KCgrS4MGDdenSJUlScnKyqlSpolq1aiklJcWxzTZt2qhs2bI6depUPh0RgLxGyAFQKHTo0EG//fabwsPDNWPGDP3888+OwNO1a1f16tVL0vU7X3fq1Em1atXSkiVLNHbsWM2bN08dO3aUMUa+vr6aM2eODhw4oEmTJkmSZs6cqdWrV+uDDz7QXXfd5bJjBJC7OF0FoNB49dVXFRsbqytXrkiSypQpo7Zt22rQoEG67777ZIxR5cqVVbduXX311VeOfmvWrFHr1q31xRdf6KGHHpIkPf/88/rPf/6jpUuXqmvXrurRo4dmzZrlkuMCkDcIOQAKlfPnz2vlypVas2aN1q5dq8OHD8tms+nNN99U27ZtVbt2bc2cOVPPPPNMpn6lS5dWnz599NZbb0m6fi1P/fr19euvv6p69erauXOnfHx8XHBEAPIKp6sAFColS5bU448/rtmzZ+vQoUPasWOH6tSpo6ioKJ09e1aS9Oyzz8rDwyPTT1JSkk6cOOHYTvHixdW1a1elp6erVatWBBzAghjJAVDg/f7772rcuLFee+019evXL8vyJUuWqHPnzkpISFCLFi0UGxurFi1aZFmvZMmSCgkJkSTt3btXDRs2VO3atbVnzx599913atasWV4fCoB8xEgOgAIvKChIxYoV04wZMxzX4/yv/fv3y9vbW3Xr1lXZsmX166+/KiwszPFTsWJFDR8+3HHTwGvXrqlXr16qUqWKNm7cqLCwMPXu3dsxAwuANRRzdQEAcCvu7u5699131alTJ4WFhen5559X7dq1dfnyZa1atUrTp0/XhAkTVLp0aU2cOFH9+/eXu7u7Hn74YV24cEGvvfaajh8/rkaNGkmSYmJitH37dq1bt04+Pj56//33FRYWpqioKE2fPt3FRwsgt3C6CkChsWPHDsXGxmrDhg06c+aMvLy81LBhQw0aNEhdunRxrBcXF6d///vf+r//+z+VKFFC999/vyZMmKB69epp9+7daty4sZ5++mnNnDnT0ScqKkqxsbH65ptv1KpVK1ccHoBcRsgBAACWxDU5AADAkgg5AADAkgg5AADAkgg5AADAkgg5AADAkgg5AADAkgg5AADAkgg5AADAkgg5AADAkgg5AADAkgg5AADAkgg5AADAkv4/umLvWAwdKDUAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "\n",
    "survival_rates = full.groupby('Sex')['Survived'].mean()\n",
    "\n",
    "\n",
    "survival_rates.index = ['female' if idx == 0 else 'male' for idx in survival_rates.index]\n",
    "\n",
    "\n",
    "survival_rates.plot(kind='bar', color='black')  \n",
    "\n",
    "\n",
    "plt.xlabel('Sex')\n",
    "plt.ylabel('Survival Rate')\n",
    "plt.title('Survival Rate by Sex')\n",
    "plt.ylim(0, 1)  \n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "42a1d0f4-c488-4fd0-91b5-daa6485202a7",
   "metadata": {},
   "outputs": [],
   "source": [
    "embarkedDf=pd.DataFrame()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "b3065f5f-a2b8-4cd8-83d9-855c35352749",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Embarked_C</th>\n",
       "      <th>Embarked_Q</th>\n",
       "      <th>Embarked_S</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Embarked_C  Embarked_Q  Embarked_S\n",
       "0           0           0           1\n",
       "1           1           0           0\n",
       "2           0           0           1\n",
       "3           0           0           1\n",
       "4           0           0           1"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "embarkedDf=pd.get_dummies(full['Embarked'],prefix='Embarked').astype(int)\n",
    "embarkedDf.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "cc042ce9-563b-40ac-9116-4962eca628cb",
   "metadata": {},
   "outputs": [],
   "source": [
    "full = pd.concat([full,embarkedDf],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "bde523ba-5c38-4da6-a465-95e79c011210",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>Embarked</th>\n",
       "      <th>Title</th>\n",
       "      <th>FamilySize</th>\n",
       "      <th>Single</th>\n",
       "      <th>Small</th>\n",
       "      <th>Large</th>\n",
       "      <th>Embarked_C</th>\n",
       "      <th>Embarked_Q</th>\n",
       "      <th>Embarked_S</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Braund, Mr. Owen Harris</td>\n",
       "      <td>1</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>U</td>\n",
       "      <td>S</td>\n",
       "      <td>Mr</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>\n",
       "      <td>0</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>C</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Heikkinen, Miss. Laina</td>\n",
       "      <td>0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>U</td>\n",
       "      <td>S</td>\n",
       "      <td>Miss</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>S</td>\n",
       "      <td>Mrs</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>Allen, Mr. William Henry</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>U</td>\n",
       "      <td>S</td>\n",
       "      <td>Mr</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  \\\n",
       "0            1       0.0       3   \n",
       "1            2       1.0       1   \n",
       "2            3       1.0       3   \n",
       "3            4       1.0       1   \n",
       "4            5       0.0       3   \n",
       "\n",
       "                                                Name  Sex   Age  SibSp  Parch  \\\n",
       "0                            Braund, Mr. Owen Harris    1  22.0      1      0   \n",
       "1  Cumings, Mrs. John Bradley (Florence Briggs Th...    0  38.0      1      0   \n",
       "2                             Heikkinen, Miss. Laina    0  26.0      0      0   \n",
       "3       Futrelle, Mrs. Jacques Heath (Lily May Peel)    0  35.0      1      0   \n",
       "4                           Allen, Mr. William Henry    1  35.0      0      0   \n",
       "\n",
       "             Ticket     Fare Cabin Embarked Title  FamilySize  Single  Small  \\\n",
       "0         A/5 21171   7.2500     U        S    Mr           2       0      1   \n",
       "1          PC 17599  71.2833   C85        C   Mrs           2       0      1   \n",
       "2  STON/O2. 3101282   7.9250     U        S  Miss           1       1      0   \n",
       "3            113803  53.1000  C123        S   Mrs           2       0      1   \n",
       "4            373450   8.0500     U        S    Mr           1       1      0   \n",
       "\n",
       "   Large  Embarked_C  Embarked_Q  Embarked_S  \n",
       "0      0           0           0           1  \n",
       "1      0           1           0           0  \n",
       "2      0           0           0           1  \n",
       "3      0           0           0           1  \n",
       "4      0           0           0           1  "
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "full.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "ed735b83-ee26-4186-96f2-bdfd19107be9",
   "metadata": {},
   "outputs": [],
   "source": [
    "pclassDf = pd.DataFrame()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "2b5271c4-8d07-48a6-ba8f-620138d61b52",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Pclass_1</th>\n",
       "      <th>Pclass_2</th>\n",
       "      <th>Pclass_3</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Pclass_1  Pclass_2  Pclass_3\n",
       "0         0         0         1\n",
       "1         1         0         0\n",
       "2         0         0         1\n",
       "3         1         0         0\n",
       "4         0         0         1"
      ]
     },
     "execution_count": 46,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pclassDf = pd.get_dummies(full['Pclass'],prefix='Pclass').astype(int)\n",
    "pclassDf.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "a22d14bf-36b4-40ea-9eba-21625b3127bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "full = pd.concat([full,pclassDf],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "7b40b2d7-1dac-452f-91dd-fed187eb4289",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Master</th>\n",
       "      <th>Miss</th>\n",
       "      <th>Mr</th>\n",
       "      <th>Mrs</th>\n",
       "      <th>Officer</th>\n",
       "      <th>Royalty</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Master  Miss  Mr  Mrs  Officer  Royalty\n",
       "0       0     0   1    0        0        0\n",
       "1       0     0   0    1        0        0\n",
       "2       0     1   0    0        0        0\n",
       "3       0     0   0    1        0        0\n",
       "4       0     0   1    0        0        0"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "titleDf=pd.get_dummies(titleDf['Title']).astype(int)\n",
    "titleDf.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "183c0129-ff74-462d-a273-22ed86a46974",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>...</th>\n",
       "      <th>Embarked_S</th>\n",
       "      <th>Pclass_1</th>\n",
       "      <th>Pclass_2</th>\n",
       "      <th>Pclass_3</th>\n",
       "      <th>Master</th>\n",
       "      <th>Miss</th>\n",
       "      <th>Mr</th>\n",
       "      <th>Mrs</th>\n",
       "      <th>Officer</th>\n",
       "      <th>Royalty</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>U</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C85</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>U</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C123</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>U</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 28 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  Sex   Age  SibSp  Parch            Ticket  \\\n",
       "0            1       0.0       3    1  22.0      1      0         A/5 21171   \n",
       "1            2       1.0       1    0  38.0      1      0          PC 17599   \n",
       "2            3       1.0       3    0  26.0      0      0  STON/O2. 3101282   \n",
       "3            4       1.0       1    0  35.0      1      0            113803   \n",
       "4            5       0.0       3    1  35.0      0      0            373450   \n",
       "\n",
       "      Fare Cabin  ... Embarked_S Pclass_1  Pclass_2  Pclass_3  Master  Miss  \\\n",
       "0   7.2500     U  ...          1        0         0         1       0     0   \n",
       "1  71.2833   C85  ...          0        1         0         0       0     0   \n",
       "2   7.9250     U  ...          1        0         0         1       0     1   \n",
       "3  53.1000  C123  ...          1        1         0         0       0     0   \n",
       "4   8.0500     U  ...          1        0         0         1       0     0   \n",
       "\n",
       "   Mr  Mrs  Officer  Royalty  \n",
       "0   1    0        0        0  \n",
       "1   0    1        0        0  \n",
       "2   0    0        0        0  \n",
       "3   0    1        0        0  \n",
       "4   1    0        0        0  \n",
       "\n",
       "[5 rows x 28 columns]"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "full = pd.concat([full,titleDf],axis = 1)\n",
    "\n",
    "full.drop('Name',axis = 1,inplace = True)\n",
    "full.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "e3f88848-afbf-4c1d-99f8-7360097ecb19",
   "metadata": {},
   "outputs": [],
   "source": [
    "cabinDf = pd.DataFrame()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "771c573e-f5d1-4667-b55d-aab443a0901c",
   "metadata": {},
   "outputs": [],
   "source": [
    "full['Cabin'] = full['Cabin'].map(lambda c: c[0])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "683a2b03-9145-416f-a700-7b4abd04445f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    U\n",
       "1    C\n",
       "2    U\n",
       "3    C\n",
       "4    U\n",
       "Name: Cabin, dtype: object"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "full['Cabin'].head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "be251b29-d3c8-4fc5-9240-6c4a4fbd486e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Cabin_A</th>\n",
       "      <th>Cabin_B</th>\n",
       "      <th>Cabin_C</th>\n",
       "      <th>Cabin_D</th>\n",
       "      <th>Cabin_E</th>\n",
       "      <th>Cabin_F</th>\n",
       "      <th>Cabin_G</th>\n",
       "      <th>Cabin_T</th>\n",
       "      <th>Cabin_U</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Cabin_A  Cabin_B  Cabin_C  Cabin_D  Cabin_E  Cabin_F  Cabin_G  Cabin_T  \\\n",
       "0        0        0        0        0        0        0        0        0   \n",
       "1        0        0        1        0        0        0        0        0   \n",
       "2        0        0        0        0        0        0        0        0   \n",
       "3        0        0        1        0        0        0        0        0   \n",
       "4        0        0        0        0        0        0        0        0   \n",
       "\n",
       "   Cabin_U  \n",
       "0        1  \n",
       "1        0  \n",
       "2        1  \n",
       "3        0  \n",
       "4        1  "
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "cabinDf = pd.get_dummies(full['Cabin'],prefix='Cabin').astype(int)\n",
    "cabinDf.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "fa903cc7-4b59-4ebb-a158-2798cc77b6b6",
   "metadata": {},
   "outputs": [],
   "source": [
    "full = pd.concat([full,cabinDf],axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "1b7a969b-dc7e-4797-a197-7c28b43bfb83",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>PassengerId</th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Ticket</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Cabin</th>\n",
       "      <th>...</th>\n",
       "      <th>Royalty</th>\n",
       "      <th>Cabin_A</th>\n",
       "      <th>Cabin_B</th>\n",
       "      <th>Cabin_C</th>\n",
       "      <th>Cabin_D</th>\n",
       "      <th>Cabin_E</th>\n",
       "      <th>Cabin_F</th>\n",
       "      <th>Cabin_G</th>\n",
       "      <th>Cabin_T</th>\n",
       "      <th>Cabin_U</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>A/5 21171</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>U</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>PC 17599</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3</td>\n",
       "      <td>0</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>STON/O2. 3101282</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>U</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>113803</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>C</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>0.0</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>373450</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>U</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>5 rows × 37 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "   PassengerId  Survived  Pclass  Sex   Age  SibSp  Parch            Ticket  \\\n",
       "0            1       0.0       3    1  22.0      1      0         A/5 21171   \n",
       "1            2       1.0       1    0  38.0      1      0          PC 17599   \n",
       "2            3       1.0       3    0  26.0      0      0  STON/O2. 3101282   \n",
       "3            4       1.0       1    0  35.0      1      0            113803   \n",
       "4            5       0.0       3    1  35.0      0      0            373450   \n",
       "\n",
       "      Fare Cabin  ... Royalty Cabin_A  Cabin_B  Cabin_C  Cabin_D  Cabin_E  \\\n",
       "0   7.2500     U  ...       0       0        0        0        0        0   \n",
       "1  71.2833     C  ...       0       0        0        1        0        0   \n",
       "2   7.9250     U  ...       0       0        0        0        0        0   \n",
       "3  53.1000     C  ...       0       0        0        1        0        0   \n",
       "4   8.0500     U  ...       0       0        0        0        0        0   \n",
       "\n",
       "   Cabin_F  Cabin_G  Cabin_T  Cabin_U  \n",
       "0        0        0        0        1  \n",
       "1        0        0        0        0  \n",
       "2        0        0        0        1  \n",
       "3        0        0        0        0  \n",
       "4        0        0        0        1  \n",
       "\n",
       "[5 rows x 37 columns]"
      ]
     },
     "execution_count": 55,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "full.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "c94122f3-ea0f-4fe1-b1e9-28b30f2e24bc",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "PassengerId      int64\n",
      "Survived       float64\n",
      "Pclass           int64\n",
      "Sex              int64\n",
      "Age            float64\n",
      "SibSp            int64\n",
      "Parch            int64\n",
      "Ticket          object\n",
      "Fare           float64\n",
      "Cabin           object\n",
      "Embarked        object\n",
      "Title           object\n",
      "FamilySize       int64\n",
      "Single           int64\n",
      "Small            int64\n",
      "Large            int64\n",
      "Embarked_C       int64\n",
      "Embarked_Q       int64\n",
      "Embarked_S       int64\n",
      "Pclass_1         int64\n",
      "Pclass_2         int64\n",
      "Pclass_3         int64\n",
      "Master           int64\n",
      "Miss             int64\n",
      "Mr               int64\n",
      "Mrs              int64\n",
      "Officer          int64\n",
      "Royalty          int64\n",
      "Cabin_A          int64\n",
      "Cabin_B          int64\n",
      "Cabin_C          int64\n",
      "Cabin_D          int64\n",
      "Cabin_E          int64\n",
      "Cabin_F          int64\n",
      "Cabin_G          int64\n",
      "Cabin_T          int64\n",
      "Cabin_U          int64\n",
      "dtype: object\n"
     ]
    }
   ],
   "source": [
    "print(full.dtypes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "774a82fc-10c0-4153-895c-c7ebbd1c5021",
   "metadata": {},
   "outputs": [],
   "source": [
    "full.drop('Embarked',axis = 1,inplace = True)\n",
    "full.drop('Cabin',axis = 1,inplace = True)\n",
    "full.drop('Title',axis = 1,inplace = True)\n",
    "full.drop('Ticket',axis = 1,inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "70d49fca-3072-45b1-8dd6-e78c96bb3b04",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "PassengerId      int64\n",
      "Survived       float64\n",
      "Pclass           int64\n",
      "Sex              int64\n",
      "Age            float64\n",
      "SibSp            int64\n",
      "Parch            int64\n",
      "Fare           float64\n",
      "FamilySize       int64\n",
      "Single           int64\n",
      "Small            int64\n",
      "Large            int64\n",
      "Embarked_C       int64\n",
      "Embarked_Q       int64\n",
      "Embarked_S       int64\n",
      "Pclass_1         int64\n",
      "Pclass_2         int64\n",
      "Pclass_3         int64\n",
      "Master           int64\n",
      "Miss             int64\n",
      "Mr               int64\n",
      "Mrs              int64\n",
      "Officer          int64\n",
      "Royalty          int64\n",
      "Cabin_A          int64\n",
      "Cabin_B          int64\n",
      "Cabin_C          int64\n",
      "Cabin_D          int64\n",
      "Cabin_E          int64\n",
      "Cabin_F          int64\n",
      "Cabin_G          int64\n",
      "Cabin_T          int64\n",
      "Cabin_U          int64\n",
      "dtype: object\n"
     ]
    }
   ],
   "source": [
    "print(full.dtypes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "dab4fe83-ea30-48e1-b68d-2a9bfcb44e50",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Survived       1.000000\n",
       "Mrs            0.344935\n",
       "Miss           0.332795\n",
       "Pclass_1       0.285904\n",
       "Small          0.279855\n",
       "Fare           0.257307\n",
       "Cabin_B        0.175095\n",
       "Embarked_C     0.168240\n",
       "Cabin_D        0.150716\n",
       "Cabin_E        0.145321\n",
       "Cabin_C        0.114652\n",
       "Pclass_2       0.093349\n",
       "Master         0.085221\n",
       "Parch          0.081629\n",
       "Cabin_F        0.057935\n",
       "Royalty        0.033391\n",
       "Cabin_A        0.022287\n",
       "FamilySize     0.016639\n",
       "Cabin_G        0.016040\n",
       "Embarked_Q     0.003650\n",
       "PassengerId   -0.005007\n",
       "Cabin_T       -0.026456\n",
       "Officer       -0.031316\n",
       "SibSp         -0.035322\n",
       "Age           -0.070323\n",
       "Large         -0.125147\n",
       "Embarked_S    -0.149683\n",
       "Single        -0.203367\n",
       "Cabin_U       -0.316912\n",
       "Pclass_3      -0.322308\n",
       "Pclass        -0.338481\n",
       "Sex           -0.543351\n",
       "Mr            -0.549199\n",
       "Name: Survived, dtype: float64"
      ]
     },
     "execution_count": 59,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "corrDf = full.corr()\n",
    "corrDf\n",
    "corrDf['Survived'].sort_values(ascending =False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "1e58ec9a-38b8-423c-8427-87820bc8293e",
   "metadata": {},
   "outputs": [],
   "source": [
    "train = pd.read_csv('train.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "ca955a8d-bc4d-4609-a996-bc13e109bbc9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',\n",
       "       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "8c37a15b-1a06-4c4b-a5a1-d9a33bfc4f03",
   "metadata": {},
   "outputs": [],
   "source": [
    "features=['Pclass',  'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "f256f06c-ef55-4312-ab8d-80ac8b03384c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1,\n",
       "       1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1,\n",
       "       1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1,\n",
       "       1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0,\n",
       "       1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0,\n",
       "       0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,\n",
       "       0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0,\n",
       "       1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0,\n",
       "       1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,\n",
       "       0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0,\n",
       "       0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,\n",
       "       1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1,\n",
       "       0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1,\n",
       "       1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,\n",
       "       0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,\n",
       "       0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0,\n",
       "       0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0,\n",
       "       1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,\n",
       "       1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,\n",
       "       1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0,\n",
       "       0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1,\n",
       "       1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1,\n",
       "       1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,\n",
       "       0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1,\n",
       "       0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,\n",
       "       0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,\n",
       "       1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1,\n",
       "       0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0,\n",
       "       0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0,\n",
       "       1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1,\n",
       "       0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,\n",
       "       0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1,\n",
       "       0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1,\n",
       "       1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1,\n",
       "       1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0])"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data=train[features].values.copy()\n",
    "data\n",
    "#target\n",
    "target=train.Survived.values.copy()\n",
    "target"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "a5656ae5-b4c6-4e6b-aae8-95226541bd1b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.model_selection import cross_val_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "8febb28a-1e18-4b15-8ec3-62824d158e8c",
   "metadata": {},
   "outputs": [],
   "source": [
    "features = ['Pclass', 'Age', 'Fare'] + [col for col in full.columns if 'Sex_' in col or 'Embarked_' in col or 'Cabin_' in col]\n",
    "data = full[features].iloc[:len(train)].values\n",
    "target = train['Survived'].values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "9d3e8c44-8f01-44b3-8248-1b5c10529cd9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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IGIUQtih776Dbt3gzBnw8YFFiyAkRO+ohBDCUr6LpVw1a9akZs2aFu0XL1687Q0rhRAPLqUgLg4iIyEnB+rWhYQEqFJF78iEEMK6fBdLGRkZvPvuu2zevJnMzEyUUgDk5ORw5coVfv31V9LT020eqBCi6EtLgz59tOII4MUXYfZs7co3IYQorPJdLI0ePZr333+f2rVrc/r0aYxGI6VLl+bnn38mMzNT1iwJIazav1/btuS338DZGWbOhAED5G7cQojCL9/LKJcvX87w4cPZu3cvQ4cOpX79+uzcuZPff/+dypUrk5OTUxBxCiGKsGXLICBAK5Qeegi+/x5efVUKJSFE0ZDvYun06dM8++yzANSpU8d0z6WHHnqIyMhIlixZYtsIhRBF1vXrMHo0dOkCly9DYKB248kGDfSOTAgh7ly+iyVvb28yMjIAbZuT48ePc+nSJQCqVavGn3/+adsIhRBF0unTEBysLeYGGDkSEhOhbFl94xJCiPzKd7HUpEkTZs6cyZUrV6hSpQru7u4k/Ltac/v27XIlnBCCnTu1bUs2bgR3d/jf/7Siyemur78VQgj95LtYioqKYvv27bRr1w4nJycGDRrEgAEDqFevHuPHj6dTp04FEacQohDKzobNmw18//1DbN5s4Pp1+OgjaNoUTpwAf3/thpNduugdqRBC3L18/39enTp1+O233/j5558BmDp1Kl5eXmzbto0OHToQGRlp8yCFEIVPQgKEh8OJE05Afd55B9zcIPfOIaGh8Nln4OWlZ5RCCHHv8l0sDRo0iOeff56goCAADAYDY8eOtXlgQojCKyFB26bk39usmeQWSr17w+efy9VuQgj7kO9puIULF8pNJ4V4gGVnayNK/y2UbrZ5s3Z3biGEsAf5Lpaeeuopvv3224KIRQhRBGzZoq1Hysvx41o/IYSwB/kulurUqcP7779PlSpVePbZZ+nbt6/Z4+WXX853EGvWrKF+/fq4ublRqVIlpk6datpG5VZWr15NQEAARqORChUqEB4ezpUrV8z6/Pbbb3To0AEvLy9KlixJaGgoR44cMevTvXt3DAaDxUPuFyWEdadO2bafEEIUdvles/Tll19Svnx5AFJSUkhJSTF735DPRQpJSUl06NCBbt26MXnyZLZu3cq4cePIyclh3LhxVo9ZtWoVHTt25IUXXiAmJoaUlBTGjh1L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SnH27FmdIrKdgwcPkp2dTfv27c3amzVrRk5ODt9++61Okd29ffv2MXDgQF588UUWLFhg8f7atWtp0qQJpUuXNrW1bt0aT09PvvnmmwKNTYqlQsbb25v333+fRo0ambUnJCQAUKtWLT3CsiknJycef/xxypYti1KKv//+m6lTp7J+/XoGDx6sd3g288knn/Djjz/ywQcf6B1KgUhOTsbBwYGgoCDc3d0pUaIEAwYM4NKlS3qHds/27duHUooyZcrQq1cvPD098fDwoHfv3pw/f17v8Gzu3Xff5a+//mLGjBl6h2JTw4cPZ926dXzxxRdcvHiRtWvX8n//9388//zzeod2z3ILhmPHjpm15/4P9dGjR+93SPfs4Ycf5tChQ7zzzju4ublZvL9//37TwEEuBwcHqlSpwsGDBws2uAIdtxI2sW3bNuXi4qI6duyodyg298UXXyhAAapt27bq0qVLeodkE8eOHVOenp5q2bJlSimlXnzxRbuahsvOzlZubm7K09NTzZo1S23evFlNnz5deXp6qsaNG+e5qXVREB8frwBVvnx59fLLL6v169er2bNnK29vb9WgQYMin9/NMjIyVNmyZVWvXr30DsXmrl27pvr06WP6OwZQrVu3tpup40aNGqkSJUqohIQEdeHCBbVnzx5Vv3595eLiovr27at3ePdk48aNFtNwxYoVU+PGjbPo26hRowJft+RUsKWYuFdbtmyhffv2+Pn58emnn+odjs09/fTTbN68mQMHDvDmm2/yzDPP8MMPPxTpK6qUUvTt25e2bdvSqVMnvcMpEEopVq9eja+vL48++igATZs2xdfXl969e7N27VratGmjc5R3LzMzE4B69erxySefANCyZUu8vb3p0aMHiYmJtG7dWs8QbWbp0qX8888/jBo1Su9QbO65555j27ZtTJs2jYCAAPbt28eECRPo0qULX375JQaDQe8Q78ny5csZMGAAYWFhgDYzMW3aNCZNmoS7u7vO0dmeUsrqz0wphYNDwU6USbFUiC1ZsoSXXnoJf39/1q5dS4kSJfQOyeaqVq1K1apVadq0KX5+frRs2ZLly5fTq1cvvUO7ax9++CH79u3j559/5vr16wCmS3mvX7+Og4NDgf/BLmiOjo4EBgZatD/77LMA7N27t0gXS56engC0a9fOrD0kJATQpiDtpVhatmwZNWvW5PHH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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def para_tune(para,data,target):\n",
    "    clf =KNeighborsClassifier(n_neighbors = para)  # n_estimators 设置为parameter\n",
    "    score = np.mean(cross_val_score(clf,data,target,scoring=\"accuracy\"))\n",
    "    return score\n",
    "\n",
    "def accurate_curve(para_range,data,target,title):\n",
    "    score = []\n",
    "    for para in para_range:\n",
    "        score.append(para_tune(para,data,target))\n",
    "    plt.figure()\n",
    "    plt.title(title)\n",
    "    plt.xlabel(\"Number of Neighbors (k)\")\n",
    "    plt.ylabel(\"Accuracy\")\n",
    "    plt.grid()\n",
    "    plt.plot(para_range,score,\"o-\",color='b')\n",
    "    return plt\n",
    "g = accurate_curve([2,3,5,8,10],data,target,\"n_neighbors\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "7b0d31bd-0e48-459a-b406-c45257fc71a4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.model_selection import learning_curve\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "def plot_learning_curve(estimator, title, data, target, cv=5,\n",
    "                        train_sizes=np.linspace(.1, 1.0, 5)):\n",
    "    plt.figure()\n",
    "    plt.title(title) \n",
    "    plt.xlabel('train')\n",
    "    plt.ylabel('Accuracy') \n",
    "    train_sizes, train_scores, test_scores = learning_curve(estimator,data, target, cv=cv,\n",
    "                                                            train_sizes=train_sizes) \n",
    "    train_scores_mean = np.mean(train_scores, axis=1) \n",
    "    test_scores_mean = np.mean(test_scores, axis=1)\n",
    "    plt.grid() \n",
    "\n",
    "    plt.plot(train_sizes, train_scores_mean, 'o-', color='b',\n",
    "             label='traning score') \n",
    "    plt.plot(train_sizes, test_scores_mean, 'o-', color='r',\n",
    "             label='testing score') \n",
    "    plt.legend(loc='best')\n",
    "    return plt\n",
    "\n",
    "g = plot_learning_curve(KNeighborsClassifier(n_neighbors = 3), 'KNN',data, target) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "898fcfe0-73a8-48f7-9648-40b546e30ab9",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "5fc13846-d9af-4c0f-be11-c98457bfe4a5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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UybthvdNmzY1WycvLw8bN25Ehw4d0LBhQ9SpUwcA8OuvvwIAHn30UWRlZSEpKQnt2rUzGUMIgcLCQtSuXdvqnJRKJZRKpVm7QqGoMT/gNWmuVYl1sYx1sY61sYx1sYx1sc4WtSnr59n0ZGlHR0d06tQJSUlJJjdQTExMhFqtRps2bczWcXBwwJgxY/Dxxx8b24qLi7Fs2TIEBQXh4YcfhoODA0aNGoV58+aZrLt582bk5eUhnNdcEhEREarBfYSmTp2Krl27YuDAgRgxYgT27duHBQsWYP78+XBycoJGo8GxY8cQGBgILy8v2NnZYdSoUVi8eDF8fX3x0EMPYfny5fjhhx+wefNmyOVyqFQqTJgwAbNmzULdunXRo0cPHD16FLGxsejdu7fx3CIiIiKSNpsHoS5dumDjxo2YMWMG+vfvD19fXyxYsABvvfUWAODgwYPo3Lkz4uPjMXz4cAD6mzDK5XK89957uHHjBlq2bIlt27ahW7duxnFjY2NRt25dfPjhh1i+fDk8PT0RHR1t8URoIiIikiabByEAGDBgAAYMGGBxWXh4uNlzxxQKBWbPno3Zs2dbHVMul2P06NEYPXp0hc6ViIiIHhzV4qGrRERERLbAIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSVS2C0Pbt29G6dWuoVCoEBARg7ty5EEJY7V9QUIBJkybB398fTk5OCA0Nxdq1a836/fTTTwgLC4OLiwu8vb0xfvx4FBQUVOamEBERUQ1i8yC0b98+9OvXDw899BCSkpLw/PPPY8qUKXj33XetrjN48GD83//9H4YOHYpvvvkGQ4YMQXR0NJYsWWLsc/r0aUREREClUmH9+vV4++23sXz5cowZM6YqNouIiIhqAHtbTyAuLg4tW7bEF198AQDo0aMHtFot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x8fD0dERoaGhAICOHTuiWbNmeOKJJ5CQkIDXXnvN4phz585FXFycWXtKSgpUKlWJ86kuUlNTbT2Faol1sYx1sY61sYx1sYx1sc4Wtbl9+3aZ+tk0CBn24ty95ycnJwcA4O7uXuL6iYmJ6Natm9nJ0wDMrhQDgA4dOsDd3R1HjhyxOuakSZMQExNjfK/RaODv749u3bqZ7bmqbrRaLVJTUxEREQGFQmHr6VQbrItlrIt1rI1lrItlrIt1tqxNaUeVDGwahAIDA2FnZ4dTp06ZtBveN23a1Oq6Fy9exOHDhzFu3DizZVlZWUhKSkK7du1MxhBCoLCwELVr17Y6rlKphFKpNGtXKBQ15ge8Js21KrEulrEu1rE2lrEulrEu1tmiNmX9PJueLO3o6IhOnTohKSnJ5AaKiYmJUKvVaNOmjdV1f/rpJwD6vTx3c3BwwKhRozBv3jyT9s2bNyMvLw/h4eEVswFERERUo9n8PkJTp05F165dMXDgQIwYMQL79u3DggULMH/+fDg5OUGj0eDYsWMIDAw0OQT222+/QalUIjAw0GxMlUqFCRMmYNasWahbty569OiBo0ePIjY2Fr1790bXrl2rchOJiIiomrL5Iza6dOmCjRs34sSJE+jfvz/Wrl2LBQsW4O233wYAHDx4EO3bt8fWrVtN1svIyDC7UuxOsbGxWL58Ob777jv06dMHCxcuRHR0NDZs2FCZm0NEREQ1iM33CAHAgAEDMGDAAIvLwsPDLT53bMWKFVixYoXVMeVyOUaPHo3Ro0dX2DyJiIjowWLzPUJEREREtsIgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSxSBEREREksUgRERERJLFIERERESSVS2C0Pbt29G6dWuoVCoEBARg7ty5EEJY7Lt69WrIZDKrr88//9zY96effkJYWBhcXFzg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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.model_selection import learning_curve\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "\n",
    "def plot_learning_curve(estimator, title, data, target, cv=5,\n",
    "                        train_sizes=np.linspace(.1, 1.0, 5)):\n",
    "    plt.figure()\n",
    "    plt.title(title) \n",
    "    plt.xlabel('train') \n",
    "    plt.ylabel(' Accuracy') \n",
    "    train_sizes, train_scores, test_scores = learning_curve(estimator,data, target, cv=cv,\n",
    "                                                            train_sizes=train_sizes) \n",
    "    train_scores_mean = np.mean(train_scores, axis=1) \n",
    "    train_scores_std = np.std(train_scores, axis=1) \n",
    "    test_scores_mean = np.mean(test_scores, axis=1)\n",
    "    test_scores_std = np.std(test_scores, axis=1)\n",
    "    plt.grid()\n",
    "\n",
    "    plt.plot(train_sizes, train_scores_mean, 'o-', color='b',\n",
    "             label='traning score') \n",
    "    plt.plot(train_sizes, test_scores_mean, 'o-', color='r',\n",
    "             label='testing score') \n",
    "    plt.legend(loc='best')\n",
    "    return plt\n",
    "\n",
    "g = plot_learning_curve(RandomForestClassifier(), ' Random forest',data,target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "4228df02-390e-46d0-881d-97ed377d0d2a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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wqhUqRIjR44k1nbq098++eQTLly4QPv27QF44403eOKJJ+ytPYDff/+dLVu22NcpS3If4H2Za3YK8P33MHOm9fEbb1gvFyAiIlJULr/O0uTJk+ncuTMDBgzgkUceISkpifj4eOLi4jCbzaSmppKSkkK9evWoXr06AQEBTJs2jXHjxtnPdktKSiIuLo7IyEiq//0bcfz48UyfPp3rrruObt26sXPnTqZOnco999xD586dAXj++efp2rUrERERPPnkk5w6dYqpU6cSHBzM008/7cqPJU+2sJSTY20v5TG5JX/LzLx49lv//tYvERGR4nB5T6Jjx46sWLGC3bt307t3bxYvXkx8fDzPPPMMAN9//z1hYWGsXr3avk1UVBQLFizgyy+/pEePHixYsICYmBhmz55tX2fq1KnMnTuXzz77jJ49e/LSSy8xYsQIPvroI/s6nTt3Zu3atZw9e5aBAwcyatQobr/9drZs2WI/nqossYUl0OUDLueFF+CHH6ztN118UkREroTLZ5bAemZbn3xu+96+ffs87xM3dOhQhg4dmu97VqhQgVGjRjndxuRSXbp0cTpmqazy9rZ+ZWVZw9I117h6RGXT9u2O7bdrr3XteERExL25fGZJikYXpiyYrf2WnQ39+qn9JiIiV05hyc0oLBVs5kzYudPafnv9dVePRkREPIHCkpvRzXTzl5xsPVYJrEFJ7TcRESkJCktuRjNLecvdfuvbV+03EREpOQpLbiYgwPqnwpKjWbNgxw6oWtU6q5TPHXRERESKTGHJzWhmyVlyMsyYYX38+utw3XUuHY6IiHgYhSU3o7DkKCvrYvstIgIGDHD1iERExNMoLLkZhSVHM2debL+98YbabyIiUvIUltyMwtJFO3ZcbL/Nnav2m4iIlA6FJTejsGSVu/3Wpw8MHOjqEYmIiKdSWHIzCktWs2ZZD+y+5hp4802130REpPQoLLkZhSVr+236dOtjtd9ERKS0KSy5mfIelrKyYOhQa/utd2+4/35Xj0hERDydwpKbKe9hKTYWtm9X+01ERK4ehSU3U57D0s6dF9tv//oX1Kjh2vGIiEj5oLDkZsprWLKd/ZaVBffdBw884OoRiYhIeaGw5GZsYSktzbXjuNri4qztt+BgmDdP7TcREbl6FJbcTHm8ke4PP8C0adbHar+JiMjVprDkZspbG+7S9tugQa4e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9tu6+KSIiLgPlx/w0rFjR1asWMHu3bvp3bs3ixcvJj4+nmeeeQaA77//nrCwMFavXm3fJioqigULFvDll1/So0cPFixYQExMDLNnz7avM3XqVObOnctnn31Gz549eemllxgxYgQfffSRfR0fHx8SExNp0aIFw4cPZ9SoUYSFhbF27VoqVnR5jsyXu4Wln3+G556zPn7lFQgJce14REREiqJMJII+ffrQp0+fPF9r3759nveJGzp0KEOHDs33PStUqMCoUaMYNWpUgfuuVasWCQkJRRuwi7nTjXRzt9+6dbM+FhERcScun1mSorOFpaws61dZ9sorsHWr2m8iIuK+FJbckC0sQdluxf38M0yebH388stQq5ZrxyMiIlIcCktuyMcHvLysj8tqWMrJgUcesbbfuna1PhYREXFHCktuyGQq+wd5v/oqfPMNVK4M77yj9puIiLgvhSU3VZbD0u7dar+JiIjnUFhyU2U1LNnOfvvrLwgPh0cfdfWIRERErozCkpsqq2HptdfUfhMREc+isOSmymJY2rMHJk2yPn7pJbjxRteOR0REpCQoLLmpshaWcrffunSBxx5z9YhERERKhsKSmyprYWnOHEhKsrbf5s9X+01ERDyHwpKbKkth6ZdfYOJE6+MXX1T7TUREPIvCkpsqK2Epd/utc2cYNsy14xERESlpCktuqqyEpX/9C7ZsUftNREQ8l8KSm7KFpbQ0143h0vZb7dquG4uIiEhpUVhyU66eWbK13y5cUPtNREQ8m8KSmwoIsP7pqrA0d661/RYQoPabiIh4NoUlN+XKmaVffoHoaOtjtd9ERMTTKSy5KVeFJYsFHnnE2n7r2BGGD7+6+xcREbnaFJbclKvC0ty58PXX1vbbu++q/SYiIp5PYclNuSIs7d0LEyZYH8fHQ506V2/fIiIirqKw5KaudlhS+01ERMorhSU3dbXD0ty5sHmzdb/vvgsV9JMjIiLlhH7luamrGZbUfhMRkfJMYclN2cJSRob1ApGlxWKBRx+1tt86dIARI0pvXyIiImWRwpKbsoUlKN3Zpddfh6++UvtNRETKL/3qc1O+vhdP2y+tsLRv38X22+zZULdu6exHRESkLFNYclMmU+neTNfWfktPh/bt4fHHS34fIiIi7kBhyY2V5kHeb7wBX35p3ceCBWq/iYhI+aVfgW6stMLS/v3w7LPWx3Fxar+JiEj5VibC0tq1a2nZsiV+fn7Url2bWbNmYRhGgdusXr2aVq1aYTabCQkJITIykvMFpIaxY8diyuPeHPfffz8mk8npa+nSpVdcV2kLCLD+WZJh6dL22xNPlNx7i4iIuKOKrh5AUlISvXr1YuDAgcyYMYOvv/6aSZMmYbFYmDRpUp7brFq1it69e/PQQw8RGxtLSkoKEydO5MSJE3z44YdO63/11VfMmTMnz/dKTk5m8ODBjBo1ymF5gwYNrry4UlYaM0tvvgmbNoGfn85+ExERgTIQlmJiYmjevDmLFi0CoFu3bmRlZREbG0tUVBRms9lhfcMwGDNmDH379mXhwoUAdOzYkZycHObMmUN6ejp+fn729c+fP8/QoUO5/vrrOXz4sMN7paen88svvxAdHU3r1q1LudKSV9Jh6dL22003lcz7ioiIuDOXzhtkZGSwadMmIiIiHJb369ePtLQ0Nm/e7LRNcnIy+/fvZ/To0Q7LIyMj2bdvn0NQAnj66aepUaMGQ4cOdXqvnTt3YrFYaN68+ZUX4wIlGZZs7bfz5+Huu2HkyCt/TxEREU/g0rC0f/9+MjMzadiwocPy+vXrA7Bnzx6nbZKTkwEwm8307NkTs9lMcHAwo0eP5q+//nJYNzExkffff5+FCxdSIY9+ku295s2bR40aNahUqRJt27bl22+/LYHqSl9JhqV589R+ExERyYtL23BnzpwBIDAw0GF55cqVAUhNTXXa5sSJEwD06dOHQYMGMW7cOLZt28aUKVM4fvw4y5YtA+Ds2bM8+uijTJs2zSmM2djC0oULF1i6dCknT54kNjaWDh06sHXrVpo1a5bndhkZGWRkZNif28aZlZVFVlZWIau/yLZNUbc1m72ACqSm5pCVZSnyfm0OHIDx4ysCJl54IYcbb7RQjDKKpLg1uzPVXD6o5vJBNXuGwtbi0rBksVh/wed1lhqQ52xQZmYmYA1LcXFxAHTo0AGLxUJ0dDTTpk0jNDSUMWPGEBISwtixY/Pd/9ixY+nfvz+dOnWyL+vUqRMNGjTghRdesAevS82aNYuYmBin5evXr3dqAxZFYmJikdY/fvwWoB4//LCfNWtSirVPiwWmTLmT8+er06TJn9SuvYU1a4r1VsVS1Jo9gWouH1Rz+aCa3Vt6enqh1nNpWAoKCgKcZ5DOnTsHQJUqVZy2sc069ezZ02F5t27diI6OJjk5mV9++YWlS5fy3XffYbFY7F8A2dnZVKhQgQoVKhAaGkpoaKjTmNq0acOOHTvyHXd0dDRRUVH256mpqdSqVYvw8HCnWbLCyMrKIjExkS5duuDt7V3o7bZurcCqVVCjxk306FGnyPsFeOutCvzwgxdms8Hy5VWoV69Hsd6nqIpbsztTzarZU6lm1eyu8upg5cWlYalevXp4eXmxd+9eh+W2540bN3baxnZKf+42GFycSjObzSxfvpy//vqLW265xWl7b29vhgwZwnvvvcfSpUupWrUqXbp0cVjnwoULVKtWLd9x+/j44OPjk+d7X8kPUFG3t+WyCxe88Pb2KvL+Dh68eO+32FgTN9989X/4r/Qzc0equXxQzeWDanZvha3DpYfx+vr60q5dOxISEhwuQrl8+XKCgoJo1aqV0zbt2rXD39+fJUuWOCxfuXIlFStWJCwsjKlTp7Jt2zaHr2HDhgGwbds2pk6dCsAbb7zBE088YW/tAfz+++9s2bKF9u3bl3zBJexKDvDOffZb27bw5JMlOzYRERFP4fLrLE2ePJnOnTszYMAAHnnkEZKSkoiPjycuLg6z2UxqaiopKSnUq1eP6tWrExAQwLRp0xg3bhzBwcFERESQlJREXFwckZGRVK9enerVq1OnTh2H/Xz66acAtGzZ0r7s+eefp2vXrkRERPDkk09y6tQppk6dSnBwME8//fTV/BiK5UpupPv22/DFF2A2695vIiIiBXH5r8iOHTuyYsUKdu/eTe/evVm8eDHx8fE888wzAHz//feEhYWxevVq+zZRUVEsWLCAL7/8kh49erBgwQJiYmKYPXt2kfbduXNn1q5dy9mzZxk4cCCjRo3i9ttvZ8uWLfbjqcqy4s4sHTwIf3+8xMbC31dqEBERkTy4fGYJrGe29enTJ8/X2rdvn+d94oYOHZrnhSbzM3XqVHv7LbcuXbo4HbPkLooTlgwDHnvMOhul9puIiMjluXxmSYqvODfSfftt2LBB7TcREZHC0q9KN1bUmaWDB8F2KNasWWq/iYiIFIbCkhsrSljK3X676y645NZ6IiIikg+FJTdWlLD0zjtqv4mIiBSHfmW6MVtY+usvyMnJf71ff4Vx46yPZ86Ev6/rKSIiIoWgsOTGbGEJIL/b2+Ruv7Vpo/abiIhIUSksuTGzGWz3IM6vFTd/Pnz+Ofj6wsKF4FX0u6KIiIiUawpLbsxkAj8/6+O8wtJvv6n9JiIicqUUltxcfgd529pv585Z229PPXX1xyYiIuIJFJbcXH5haf58SEy0tt8WLFD7TUREpLgUltxcXjfTzd1+e+EFaNjw6o9LRETEUygsublLZ5YMA4YNs7bf7rwTIiNdNzYRERFPoLDk5i4NS+++C+vXq/0mIiJSUhSW3FzusPTbbxAVZX0+YwaEhrpuXCIiIp5CYcnN2S4d8NVX0K+ftf0WFgZjxrh0WCIiIh6joqsHIMWXkACffmp9vHjxxeUPPKD2m4iISEnRzJKbSkiwziTldTHKyEjr6yIiInLlFJbcUE6ONRAZRv7rjBlT8M11RUREpHAUltzQ5s1w+HD+rxsGHDpkXU9ERESujMKSGzp6tGTXExERkfwpLLmhmjVLdj0RERHJn8KSG2rbFkJCwGTK+3WTCWrVsq4nIiIiV0ZhyQ15ecFrr1kfXxqYbM9ffVWXDxARESkJCktuKiICli+HG25wXB4SYl0eEeGacYmIiHgaXZTSjUVEwH33Wc96O3rUeoxS27aaURIRESlJCktuzssL2rd39ShEREQ8l9pwIiIiIgVQWBIREREpgMKSiIiISAEUlkREREQKoLAkIiIiUgCFJREREZECKCyJiIiIFEBhSURERKQACksiIiIiBdAVvEuAYRgApKamFmv7rKws0tPTSU1NxdvbuySHVmapZtXsqVSzavZUnliz7fe27fd4fhSWSsC5c+cAqFWrlotHIiIiIkV17tw5qlSpku/rJuNycUouy2KxcOTIESpXrozJZCry9qmpqdSqVYtDhw4RGBhYCiMse1SzavZUqlk1eypPrNkwDM6dO8f1119PhQr5H5mkmaUSUKFCBUJCQq74fQIDAz3mB7CwVHP5oJrLB9VcPnhazQXNKNnoAG8RERGRAigsiYiIiBRAYakM8PHxYcqUKfj4+Lh6KFeNai4fVHP5oJrLh/JYs40O8BYREREpgGaWRERERAqgsCQiIiJSAIUlERERkQIoLLnY2rVradmyJX5+ftSuXZtZs2Zd9rLr7sQwDN5++22aNWtGQEAAN910E2PGjHG4Nczu3bu55557qFKlClWrVuXRRx/lzJkzrht0CYqIiKBOnToOyzy13q1bt9KhQwf8/f257rrrGDJkCMePH7e/7ml1v/POOzRp0gR/f38aNWrE66+/7vB315PqPXToEEFBQWzatMlheWFqPHfuHI8//jg1atTA39+fLl26kJKScvUGX0z51bxp0ybuvvtugoODqVGjBhEREezdu9dhHU+rObfXXnsNk8nEwYMHHZa7a82FZojLbNmyxfD29jYGDx5sfPbZZ8akSZMMk8lkzJgxw9VDKzFxcXGGl5eXMWHCBCMxMdF48803jWrVqhmdOnUyLBaLcfr0aeOGG24w7rjjDuOTTz4x3n77bSMoKMjo0qWLq4d+xRYtWmQARu3ate3LPLXe7777zvD19TXuueceY926dcbChQuNGjVqGGFhYYZheF7d77zzjgEYo0ePNj7//HPjueeeM0wmkxEfH28YhmfVe/DgQSM0NNQAjI0bN9qXF7bGe+65x6hevbqxcOFCY8WKFUazZs2M6667zjh58uRVrqTw8qs5KSnJqFixohEREWGsXr3a+Oijj4xbb73VuO6664wTJ07Y1/OkmnPbs2ePYTabDcA4cOCAw2vuWHNRKCy5UHh4uHHHHXc4LBs/frwREBBgpKenu2hUJScnJ8cICgoyRo4c6bD8P//5jwEY27ZtM2bOnGn4+fkZx48ft7++Zs0aAzA2b958tYdcYn7//XcjODjYCAkJcQhLnlpvhw4djNatWxvZ2dn2ZStWrDBCQkKM/fv3e1zdYWFhRps2bRyWDRw40KhTp45hGJ7xfc7JyTEWLFhgXHPNNcY111zj9Eu0MDUmJSUZgLF69Wr7OsePHzf8/f2N6dOnX7VaCutyNd97771G06ZNjZycHPuyI0eOGF5eXvag7Gk122RnZxthYWFGSEiIU1hyt5qLQ204F8nIyGDTpk1EREQ4LO/Xrx9paWls3rzZRSMrOampqQwePJhBgwY5LG/YsCEA+/btY926dbRt25bq1avbX+/atSuVK1dmzZo1V3W8Jemxxx4jPDycTp06OSz3xHpPnjzJpk2bGDlyJF5eXvblERERHDp0iLp163pc3RkZGU63SKhWrRonT54EPOP7vHPnTp544gmGDBnCokWLnF4vTI3r1q3D39+f8PBw+zrVq1fn7rvvLpOfw+VqbtWqFWPGjHG4h1jNmjUJDAxk3759gOfVbPPiiy9y7NgxJkyY4PSau9VcHApLLrJ//34yMzPtwcGmfv36AOzZs8cVwypRQUFB/Otf/6JNmzYOyxMSEgC45ZZb2LVrl9NnUKFCBerWreu2n8H8+fP53//+x9y5c51e88R6d+7ciWEYXHvttfzzn/+kcuXKBAQEMHjwYE6fPg14Xt1jx45l/fr1fPDBB5w9e5Z169bx73//mwcffBDwjHpvvPFG9u7dy8svv4yfn5/T64WpcdeuXdx0001UrOh4G9L69euXyc/hcjVPnjyZRx55xGHZxo0bOX36NLfccgvgeTUD/PTTT0ydOpUFCxbg7+/v9Lq71VwcupGui9gOgrz0ZoSVK1cGcDgA2pMkJSURFxdH7969adKkCWfOnMnzhoyVK1d2y8/g119/JSoqioULF1KtWjWn1z2tXoATJ04A8Mgjj9C9e3c+/vhjfvnlF6Kjo9m3bx9btmzxuLr79+/PF198YQ9HYJ1VefXVVwHP+D5fc801XHPNNfm+Xpga3e1zuFzNlzpx4gTDhg0jJCSEIUOGAJ5Xc3Z2NkOGDOGxxx7j7rvv5sCBA07ruFvNxaGw5CIWiwUAk8mU5+u5p3k9xebNm7n33nupV68e7777LmA9Wy6vz8AwDLf7DAzD4JFHHqFHjx707ds333U8pV6bzMxMAFq0aMH8+fMB6NSpE0FBQTzwwAMkJiZ6XN333XcfW7ZsYfbs2bRq1YqdO3cydepU+vfvz3//+1+PqzcvhanRYrF47Odw5MgRunbtyvHjx9mwYQMBAQGA59X8wgsvcPr0aWJjY/Ndx9NqzovCkosEBQUBzjNI586dA3A6HsLdLV26lIcffpjQ0FDWrVtn/59MlSpV8vyfR1paGiEhIVd7mFfk9ddfZ+fOnfzwww9kZ2cD2E8lz87OpkKFCh5Vr41tNrRnz54Oy7t16wZAcnKyR9WdlJTEunXreOedd3jssccAuPvuu7npppvo2bMnq1ev9qh681OYGoOCgvJsw6Slpbn1v3E//PAD99xzD2lpaaxdu5Y77rjD/pon1bx9+3ZmzpzJmjVr8PHxITs72/4f/ZycHHJycvDy8vKomvPjGZHPDdWrVw8vLy+n63PYnjdu3NgVwyoV8fHxDBo0iNatW/PVV19Ro0YN+2uhoaFOn4HFYuHAgQNu9xksX76cP//8k5o1a+Lt7Y23tzfvv/8+v/76K97e3kybNs2j6rVp0KABYD3oObesrCwAzGazR9X966+/Ajgdi3f33XcD1uM7PKne/BSmxtDQUA4cOGD/BWuzd+9et/0cvvjiC9q0aYNhGHz11VfceeedDq97Us2ffPIJmZmZdO7c2f5v2qOPPgpYj0eyncDiSTXnR2HJRXx9fWnXrh0JCQkOF7Jbvnw5QUFBtGrVyoWjKzlvvfUW48ePp3///qxfv97pfxnh4eF8+eWX9uNewHpmxblz5xzOrHAHb731Ftu2bXP46tmzJzVr1mTbtm0MHz7co+q1adSoEXXq1GHp0qUOy1euXAlA27ZtParum2++GcDpjNUtW7YAULduXY+qNz+FqTE8PJxz586xbt06+zonTpzgyy+/dMvPYfv27dx7773ceOONbN261X5Qd26eVPPw4cOd/k2bMmUKYP37/dZbbwGeVXO+rva1CuSiDRs2GCaTyejXr5+xZs0aY/LkyYbJZDJmz57t6qGViKNHjxpms9moXbu2sXnzZuObb75x+Dp+/Lhx4sQJo1q1asatt95qJCQkGO+8844RHBxsdO/e3dXDLxFDhgxxuM6Sp9b70UcfGSaTyRgwYICxfv16Y86cOUZAQIDRt29fwzA8r+6+ffsa/v7+RmxsrLFx40Zj7ty5RrVq1Yzbb7/dyMzM9Lh6N27c6HT9ncLW2L59eyM4ONh45513jISEBKNZs2bGDTfcYJw6deoqV1E0edV82223Gd7e3sZHH33k9O/Z3r177et5Us2XWrhwYZ4XpXTXmgtLYcnFEhISjKZNmxqVKlUy6tata7z44ouuHlKJeffddw0g36+FCxcahmEYP/zwg9GpUyfDbDYb1157rTF8+HAjNTXVtYMvIZeGJcPw3HpXrVpl3HHHHYaPj49Rs2ZN4+mnnzb++usv++ueVHdGRobx3HPPGXXq1DEqVapk1K9f33jmmWeMc+fO2dfxpHrz+yVamBpPnTplPPzww0ZQUJARGBhodO/e3fj555+v4uiL59Ka9+3bV+C/Z0OGDLFv6yk15yW/sOSuNReWyTA86EZkIiIiIiVMxyyJiIiIFEBhSURERKQACksiIiIiBVBYEhERESmAwpKIiIhIARSWRERERAqgsCQiIiJSAIUlEXG5OnXq8PDDD7t6GAXatGkTJpOJTZs2XfF7TZ06Nc+7tItI2aSwJCJylT322GN88803rh6GiBRSRVcPQESkvAkJCSEkJMTVwxCRQtLMkoiUCVlZWTz11FMEBwcTHBzMkCFDHO5on5iYSNu2balSpQpVq1Zl0KBBHDp0yP76e++9h8lk4uDBgw7ve2mLz2Qy8cYbb/DYY49xzTXXEBAQQL9+/Th27JjDdm+99RYNGzbEbDZz99138+uvvzqN+auvvqJr164EBwdTqVIl6taty9SpU7FYLAAcPHgQk8nEyy+/TKNGjbjmmmt477338mzDffLJJ7Rs2RJfX19q1KhBZGQk58+ft7/+119/MWrUKEJCQvDx8eHmm2/mpZdeKvLnLCJFp7AkImXCsmXL+O677/j3v/9NfHw8q1evpnfv3gB88MEHhIeHc8MNN7BkyRJeeeUVvvnmG8LCwjh+/HiR9zVx4kRycnJYunQpL774IqtXr2bMmDH21+fOncvjjz9Ot27d+OSTT2jdujXDhw93eI8dO3bQqVMnqlatyrJly1i1ahVt2rQhJiaGpUuXOqw7adIknnnmGebPn0/Hjh2dxvPhhx/Su3dvbr75Zj7++GOmTp3KokWLuO+++7DdvjMyMpI1a9bw4osvsm7dOu677z6efvpp3nvvvSLXLyJF5OIb+YqIGLVr1zaqVavmcMf6jz/+2ACMdevWGTVq1DA6d+7ssM3evXuNSpUqGePHjzcMI/+7odeuXdvhjvCAcddddzmsM3ToUCMgIMAwDMOwWCzGtddea/Tr189hnccff9zhjuzvv/++0b17dyMnJ8e+Tk5OjlGlShVj+PDhhmEYxoEDBwzA+Oc//+nwXlOmTDFs//xaLBYjJCTE6Natm8M6n3/+uQEYn376qWEYhhEaGmo89thjDutMmzbNWLVqlSEipUszSyJSJvTo0YPKlSvbn9977714e3vzzjvv8Mcff/DPf/7TYf169eoRFhbGxo0bi7yvsLAwh+chISH2ltfu3bs5fvw49913n8M6AwYMcHj+4IMPsmbNGjIzM/npp5/sM0LZ2dlkZmY6rNu0adN8x7J7924OHz5Mr169yM7Otn/dfffdBAYGkpiYCECHDh2YP38+PXr04M033+TXX3/lueeeo2fPnkWuX0SKRmFJRMqEGjVqODyvUKECVatW5cyZM3m+bltme70o/Pz8nPZl/N3uOnXqFADVq1d3WKdmzZoOzy9cuMBjjz1GlSpVaNq0KePGjePAgQN4e3vb38vmuuuuy3csJ0+eBGDkyJF4e3s7fKWmpnLkyBEAXn31VWbMmMGBAwcYOXIkderU4c4772T79u1Frl9EikZnw4lImXD69GmH5zk5Ofz5558EBgYC8Mcffzhtc/ToUapVqwZgP2A6JyfHYZ20tLQijcP2fpce8G0LNTaRkZEsX76cZcuW0aVLF/z9/QG49tpri7S/oKAgAOLj42nfvr3T68HBwQD4+PgwadIkJk2axG+//caqVauYPn06gwYNYteuXUXap4gUjWaWRKRM+Pzzz8nOzrY/X758OdnZ2YwYMYIaNWqwePFih/X379/PN998w1133QVgD1W5z5DbvXu3U8i5nAYNGlCrVi0++ugjh+WrVq1yeP7111/ToUMHevfubQ9K//vf/zhx4oT9bLjCuPnmm7n22ms5cOAALVu2tH+FhIQwYcIEtm/fzoULF2jYsKH97Lcbb7yRUaNG8cADDzjUKyKlQzNLIlIm/PHHH/Tt25fRo0fzyy+/EB0dTZcuXejSpQuzZs1i6NCh3H///QwZMoQ///yTqVOncs011xAVFQVAx44d8fPzIyoqihdeeIFz587Z1ykKk8lEXFwcgwYNYtiwYfTv35+tW7fy5ptvOqzXqlUr/vOf/zBv3jwaNWrEjh07mDFjBiaTyeGU/8vx8vLihRdeYMSIEXh5eXHvvfdy5swZpk+fzuHDh2nRogVms5kWLVoQExNDpUqVaNasGbt37+a9996jX79+RapPRIrB1UeYi4jUrl3biIyMNIYNG2YEBAQY11xzjTFy5EgjLS3Nvs7y5cuNFi1aGJUqVTKqVatmDB482Pjtt98c3uezzz4zbr31VqNSpUpGw4YNjcWLFxtdu3Z1OhtuypQpDtvlPjvNZunSpUaTJk0MHx8fo2XLlsaSJUsczoY7efKkMWjQIKNq1apGQECA0bRpU+O1114zhg8fbtSsWdPIzs62nw23cOHCy+5v2bJlRosWLQwfHx+jatWqRq9evYydO3faX09NTTWeeuop48YbbzQqVapkhISEGE8//bSRnp5exE9bRIrKZBiXHIkoIiIiInY6ZklERESkAApLIiIiIgVQWBIREREpgMKSiIiISAEUlkREREQKoLAkIiIiUgCFJREREZECKCyJiIiIFEBhSURERKQACksiIiIiBVBYEhERESmAwpKIiIhIAf4fEzJmvxov4QwAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def para_tune(para,data,target):\n",
    "    clf = RandomForestClassifier(n_estimators=para)  \n",
    "    score = np.mean(cross_val_score(clf,data,target,scoring=\"accuracy\"))\n",
    "    return score\n",
    "\n",
    "def accurate_curve(para_range,data,target,title):\n",
    "    score = []\n",
    "    for para in para_range:\n",
    "        score.append(para_tune(para,data,target))\n",
    "    plt.figure()\n",
    "    plt.title(title)\n",
    "    plt.xlabel(\"boundaries\")\n",
    "    plt.ylabel(\"Accuracy\")\n",
    "    plt.grid()\n",
    "    plt.plot(para_range,score,\"o-\",color='b')\n",
    "    return plt\n",
    "g = accurate_curve([2,5,70,90,150],data,target,\"n_estimator\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "9b185d99-7407-4bc0-a2e8-26fd15035e9d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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Z9r/3XffZfLkyZnmEe3cuTPTvgkJCcCVx3Ei4jhKlkSkWLM+Xvrtt9/s2r///ntmz54NQFpamq195cqVrFixgvbt2/Pwww8zc+ZMgoKCGDVqFL/88kuez1++fHk6dOjAZ599xsqVK23t586d47nnnsu0/+OPP84ff/zB2LFj7ZK4PXv28Mwzz/DSSy9lenz33//+l59++sn2+eDBg8yePZsKFSoQHh5uay9RooRdrCJSNDRnSUSKtc6dOxMQEMALL7zATz/9RLVq1Thw4ABr166ldOnSAJw5c8b258CBA/Hx8eGNN94AzHlFL774In369KFfv35s3Lgxz2N47bXXuOeee+jevTtdu3alYsWKrFmzBnd390z7jh49mg0bNjBr1iw+//xzWrduzdmzZ1mxYgV//fUX77zzjm3cVu7u7tx999089NBDGIbBypUruXTpEqtWrcLX19e2X6VKlfjpp58YNGgQHTp04N///neeYxGRvNOdJREp1ipWrMgnn3xCu3bt+PTTT3n99dfZv38/Q4YM4aeffqJs2bJs2LABwzB45plnSEpKYuLEiXZv2D322GOEhoayadMm3n777TyP4Y477mDHjh08/PDDfPHFFyxYsIC77rqLjz76KNO+3t7ebN68mZiYGC5dusTcuXNZt24dLVq04LPPPqN3796Zjhk7dixDhgxhzZo1xMXF0axZMz7//HPuv/9+u/1ee+01qlSpwltvvcWHH36Y5zhEJH8shl6VEBFxiIULF9K3b19mzZrF0KFDHT0cEcmG7iyJiIiI5EBzlkTkppKYmMgHH3yQ6/0ff/xxqlSpUmjjEZHiT8mSiNxUEhMTc1ww8lqhoaFKlkRucpqzJCIiIpIDzVkSERERyYGSJREREZEcaM5SAcjIyODXX3+lVKlSWCwWRw9HREREcsEwDM6fP89tt91mK6OUFSVLBeDXX38lODjY0cMQERGRfDh27BiVKlXKdruSpQJQqlQpwPxh+/v7F1i/aWlpbNq0ifDwcDw8PAqs3+LE1WN09fjA9WNUfM7P1WNUfPmXnJxMcHCw7d/x7ChZKgDWR2/+/v4Fniz5+vri7+/vkn8BwPVjdPX4wPVjVHzOz9VjVHw37npTaDTBW0RERCQHSpZEREREcqBkSURERCQHSpZEREREcqBkSURERCQHSpZEREREcqBkSURERCQHSpZE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S57vLK1cuZJhw4axa9cuhgwZQuPGjfnqq684cOAAVapUsb22L1JYHngAatQwC+y+/bajRyMiIq4uz8lSUlIS999/PwD169e3TbSuWLEiY8aMYdmyZQU7QpFrqMCuiIgUpTwnSwEBAbY3zGrWrMmxY8ds6yLVqFGDX375pWBHKJKFxx6DW26BX36B99939GhERMSV5TlZatmyJa+88gp//fUXVatWpWTJkrY5P9u3b7/uQpIiBUEFdkVEpKjkOVmaNGkS27dvp3PnzpQoUYKBAwcyYMAAGjVqxPjx43nwwQcLY5wimTz9NJQsCd9/D5s2OXo0IiLiqvL8Nlz9+vX56aef2L17N2DWSfP392fbtm106dKFMWPGFPggRbISGAj9+8OsWebdpfbtHT0iERFxRXlOlgYOHEjv3r0JCwsDzPImY8eOLfCBieTG0KHmekuffQb/+x80buzoEYmIiKvJ82O4xYsX57pKr0hhu/12eOQR8/vYWMeORUREXFOek6UmTZrw8ccfF8ZYRPLln8XeiYuDQ4ccOxYREXE9+Zqz9Oqrr7Jy5Upq167NrbfearfdYrHwtlYKlCJUrx507Agff2yuu/T6644ekYiIuJI8J0urV6/mtttuA2DPnj3s2bPHbrvFYimYkYnkwbPPmsnS/PkwaZK5BpOIiEhByHOydOTIkcIYh8gNadUKmjaFnTvhtddgyhRHj0hERFxFnucsiRRH1gK7YCZLFy44djwiIuI68nxnqW3bttfd57PPPsvXYERuRNeuUL06HDxoPo4bMsTRIxIREVeQ5ztLGRkZGIZh93X+/Hm++uorfvzxR+68887CGKfIdV1dYPell1RgV0RECkae7yxt2bIly/Y///yT+++/X8mSONRjj8HEiWaB3RUroGdPR49IREScXYHNWQoMDGT06NHMmjWroLoUyTMfnyuP31RgV0RECkKBTvDOyMjgt99+K8guRfLMWmB31y6Ij3f0aERExNnl+THcF198kaktPT2dY8eOERMTQ6NGjQpkYCL5VaYM9OsHs2ebd5fCwx09IhERcWZ5TpZCQ0OxWCwYhmFbgNL451lHcHAws2fPLtABiuTHsGFmgd1PP4VvvgHl8CIikl95TpY2b96cqc1iseDv70/9+vVxc9PSTeJ41gK7771nFthdtszRIxIREWeV58ymdevW3Hvvvfj7+9O6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U8clruHVro3Pla7fkiVLqFChAmPHjqVjx4706NEDcI2/h5B9fM5+DY8ePUp0dDRz586lXLlymbY7+/W7XnwOv34Oefgn2bp8+bKxefNmY+/evXbt7733ngEY69evd9DICsbmzZszzefx8PAwxo8fn2nfe+65x2jfvn0Rjq5gZBXjd999Z3z55Zd2+x06dMjw9PTMNM+iuPviiy+M0qVLG7Vr1zbOnDljGIZrXcOs4nOl63fgwAHj888/N/773/8a5cuXN+rVq2dcunTJZa5hdvE58zXMyMgw2rZta/To0cPW1qdPH7s5Pc58/XITn6Ovn96GK2bc3d0JDQ3N1H7//fcDsGvXLjp27FjEoypcpUuXzvL/fC5cuJDj3Rpn0rBhw0xtd9xxB7Vq1WLXrl1FP6B8WrZsGY8//jghISFs3LiRMmXKAK5zDbOLz1WuH0D16tWpXr06rVq1olq1arRr146VK1e6zDXMLr5HH300077Ocg1ff/11vv/+e3bv3s3ly5cBc7kLgMuXL+Pm5ubU1y838Tn676AewxUzJ06c4M033+T48eN27ZcuXQLI8vakswsJCeHgwYN2bRkZGRw5coTatWs7aFQFJy0tjYULF2a5FsilS5ec5prGxsbSs2dPmjVrxhdffEH58uVt21zhGmYXnytcv6SkJN555x2SkpLs2ps0aQKYa4M58zW8XnyHDx926msYFxfH77//ToUKFfDw8MDDw4N3332Xo0eP4uHhwZQpU5z6+l0vvgkTJjj++hX6vSvJk0OHDhmAMWHCBLv2WbNmGW5ubsZPP/3koJEVjKweUcXExBglS5bM8pXXhIQEB4zyxmQV4+233260bNnSbr9vvvnGcHNzM958880iHmHezZs3zwCM7t27GykpKZm2O/s1vF58zn79rP9def755+3arcsjrF+/3qmvYW7ic+Zr+NNPPxlff/213Vfnzp2NChUqGF9//bVx4sQJp75+uYnP0ddPyVIx1Lt3b8PT09OYOnWq8cknnxiTJ082PD09jWeeecbRQ7thWSUSp0+fNsqVK2c0aNDAWLVqlfHmm28agYGBRseOHR030BuQVYxvv/22ARh9+vQxNm3aZJtP0bBhQyMtLc1xg82FkydPGj4+PkblypWNrVu3Gtu3b7f7SkpKcuprmJv4nPn6WT322GOGl5eXMX36dOPTTz81ZsyYYZQqVcpo3769kZGR4dTX0DCuH58rXMOrXTunx9mv37Wujc/R10/JUjF06dIlY8qUKUaNGjUMLy8v44477jCmTZtmXL582dFDu2FZJRKGYRi7d+822rVrZ/j4+Bi33HKL0b9/fyM5Odkxg7xB2cW4dOlS46677jJ8fX2NoKAgo3///rYJxMWZ9T9S2X0tWLDAMAznvYa5jc9Zr5/V33//bUydOtWoWbOm4eXlZVSpUsUYP3688ffff9v2cdZraBi5i8/Zr+HVrk0mDMO5r9+1sorPkdfPYhj/zKISERERkUw0wVtEREQkB0qWRERERHKgZElEREQkB0qWRERERHKgZElEREQkB0qWRERERHKgZElEREQkB0qWRMThqlSpwuOPP+7oYeRoy5YtWCwWtmzZcsN9TZ48GYvFcuODEpEioWRJRKSIPfXUU2zfvt3RwxCRXCrh6AGIiNxsKlWqRKVKlRw9DBHJJd1ZEpFiIS0tjSFDhhAYGEhgYCB9+vTh9OnTtu3x8fG0bNmS0qVLU7ZsWXr27MmxY8ds2xcuXIjFYuHnn3+26/faR3wWi4W5c+fy1FNPUaZMGfz8/OjWrRu//fab3XH/93//R82aNfHx8aF169YcPXo005i/+OIL2rdvT2BgIJ6enlStWpXJkyeTkZEBwM8//4zFYuHll1+mVq1alClThoULF2b5GO7DDz+kcePGeHt7U758eaKiovjrr79s2//++28GDRpEpUqV8PLy4s477+Sll17K889ZRPJOyZKIFAvLly/nf//7H++88w6xsbGsW7eOrl27AvDee+8RHh5OxYoVWbp0KbNmzWL79u00b96cpKSkPJ9r7NixpKens2zZMl588UXWrVvH0KFDbdtfe+01/vOf/9ChQwc+/PBDmjVrRv/+/e362LVrF+3ataNs2bIsX76cNWvW0KJFC2JiYli2bJndvuPGjWPkyJG89dZbtG3bNtN4lixZQteuXbnzzjv54IMPmDx5MosWLeKBBx7AWr4zKiqK9evX8+KLL7Jx40YeeOABRowYwcKFC/Mcv4jkUZGU6xURyUHlypWNcuXK2VVI/+CDDwzA2Lhxo1G+fHnjvvvuszvm4MGDhqenpzFq1CjDMAxjwYIFBmAcOXIkU999+vSxfQaMe++9126fvn37Gn5+foZhGEZGRoZxyy23GN26dbPb5z//+Y8BGJs3bzYMwzDeffddo2PHjkZ6erptn/T0dKN06dJG//79DcMwjCNHjhiA8eijj9r1NWnSJMP6n9+MjAyjUqVKRocOHez2+eSTTwzAWLt2rWEYhhESEmI89dRTdvtMmTLFWLNmjSEihUt3lkSkWOjUqROlSpWyff73v/+Nh4cHb775JqdOneLRRx+1279atWo0b96czZs35/lczZs3t/tcqVIl2yOvffv2kZSUxAMPPGC3T/fu3e0+9+7dm/Xr15OamsqPP/5ouyN0+fJlUlNT7fatV69etmPZt28fx48fp0uXLly+fNn21bp1a/z9/YmPjwegTZs2vPXWW3Tq1Ik33niDo0ePMmHCBDp37pzn+EUkb5QsiUixUL58ebvPbm5ulC1blrNnz2a53dpm3Z4Xvr6+mc5l/PO4648//gAgKCjIbp8KFSrYfb506RJPPfUUpUuXpl69egwfPpwjR47g4eFh68vq1ltvzXYsZ86cAWDgwIF4eHjYfSUnJ/Prr78CMHv2bKZOncqRI0cYOHAgVapU4Z577uG7777Lc/wikjd6G05EioU///zT7nN6ejq///47/v7+AJw6dSrTMSdPnqRcuXIAtgnT6enpdvtcuHAhT+Ow9nfthG9rUmMVFRVFXFwcy5cvJywsjJIlSwJwyy235Ol8AQEBAMTGxhIaGpppe2BgIABeXl6MGzeOcePG8csvv7BmzRqee+45evbsyd69e/N0ThHJG91ZEpFi4ZNPPuHy5cu2z3FxcVy+fJkBAwZQvnx5Fi9ebLf/4cOH2b59O/feey+ALam6+g25ffv2ZUpyrqdGjRoEBwezYsUKu/Y1a9bYff7yyy9p06YNXbt2tSVK33zzDadPn7a9DZcbd955J7fccgtHjhyhcePGtq9KlSoxevRovvvuOy5dukTNmjVtb7/dfvvtDBo0iEceecQuXhEpHLqzJCLFwqlTp3jwwQcZPHgwBw4cYMyYMYSFhREWFsa0adPo27cvDz/8MH369OH3339n8uTJlClThujoaADatm2Lr68v0dHRPP/885w/f962T15YLBZmzJhBz5496devHw899BA7duzgjTfesNuvadOmvP/++8ybN49atWqxa9cupk6disVisXvl/3rc3d15/vnnGTBgAO7u7vz73//m7NmzPPfccxw/fpxGjRrh4+NDo0aNiImJwdPTk/r167Nv3z4WLlxIt27d8hSfiOSDo2eYi4hUrlzZiIqKMvr162f4+fkZZcqUMQYOHGhcuHDBtk9cXJzRqFEjw9PT0yhXrpzRq1cv45dffrHr5+OPPzYaNGhgeHp6GjVr1jQWL15stG/fPtPbcJMmTbI77uq306yWLVtm1KlTx/Dy8jIaN25sLF261O5tuDNnzhg9e/Y0ypYta/j5+Rn16tUz5syZY/Tv39+oUKGCcfnyZdvbcAsWLLju+ZYvX240atTI8PLyMsqWLWt06dLF+P77723bk5OTjSFDhhi333674enpaVSqVMkYMWKEcfHixTz+tEUkryyGcc1MRBERERGx0ZwlERERkRwoWRIRERHJgZIlERERkRwoWRIRERHJgZIlERERkRwoWRIRERHJgZIlERERkRwoWRIRERHJgZIlERERkRwoWRIRERHJgZIlERERkRwoWRIRERHJwf8D2sTrkO7cnSIAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "def para_tune(para, data,target):\n",
    "    clf = RandomForestClassifier(n_estimators=300, max_depth=para)\n",
    "    score = np.mean(cross_val_score(clf, data,target, scoring='accuracy'))\n",
    "    return score\n",
    "\n",
    "def accurate_curve(para_range, data,target, title):\n",
    "    score = []\n",
    "    for para in para_range:\n",
    "        score.append(para_tune(para, data,target))\n",
    "    plt.figure()\n",
    "    plt.title(title)\n",
    "    plt.xlabel('boundaries')\n",
    "    plt.ylabel('Accurary')\n",
    "    plt.grid()\n",
    "    plt.plot(para_range, score, 'o-',color='b')\n",
    "    return plt\n",
    "\n",
    "g = accurate_curve([5, 15, 25, 35, 45], data,target, 'max_depth')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "bfc8f9a5-5c91-4505-b5ec-cdf07f3150c4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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nTWezB3zYhjzYhxzYxjzYhjzYpycucntrlI2WQshPz8/KJVKXL58Wac9+3X16tWNnnvr1i2cPn0ao0aN0nvv8ePHiIiIQJMmTXSuIYRARkYG3NzcjF5XpVJBpVLptVtZWRWbH/DiFGthYl4MY16MY24MY14MY16MkyM3ef08WQdL29jYoGXLloiIiNBZQDE8PBwuLi5o1KiR0XN//fVXAFIvz/Osra0xbNgwzJ49W6d927ZtSE1NRUBAgGm+ABERERVrsq8jNHHiRLRt2xa9evXCwIEDcfToUcydOxdz5syBra0tEhMTce7cOfj5+encAvv777+hUqkM7mtmZ2eHsWPHYvr06fDw8ED79u3x119/YerUqejUqRPatm1bmF+RiIiIiijZt9ho06YNtmzZggsXLqB79+5Yu3Yt5s6dizFjxgAA/vjjDzRt2hS7du3SOS82NlZvptizpk6diiVLlmDPnj3o3Lkz5s2bh8GDB2Pz5s0F+XWIiIioGJG9RwgAgoKCEBQUZPC9gIAAg/uOLV26FEuXLjV6TQsLC3zwwQf44IMPTBYnERERlSyy9wgRERERyYWFEBEREZktFkJERERktlgIERERkdliIURERERmi4UQERERmS0WQkRERGS2WAgRERGR2WIhRERERGaLhRARERGZLRZCREREZLZYCBEREZHZYiFEREREZouFEBEREZktFkJERERktlgIERERkdliIURERERmi4UQERERmS0WQkRERGS2WAgRERGR2WIhRERERGaLhRARERGZLRZCREREZLZYCBEREZHZYiFEREREZouFEBEREZktFkJERERktizlDoCIiIjMUFYWcPgwcPcu4OUFtGgBKJWFHgYLISIiIipcERHAyJHArVtP23x8gIULgeDgQg2Ft8aIiIio8EREACEhukUQANy+LbVHRBRqOCyEiIiIqHBkZUk9QULov5fd9tFH0nGFhLfGiIiIqGDEx8P56lUofvpJ6gE6ckS/J+hZQgA3b0pjhwICCiVEFkJERESUP4mJwPXr0uPaNd0/r1+HVUICAvJz3bt3TRhkzlgIERERkWFPnuRY6ODRo1wvkebsDOtq1WBRsSJgYQGsX5/753p5vVzcL4CFEBERkblKTQVu3DBe6Ny/n/s1SpcGKlYEKlR4+ud/z9VlyyIyJgYdO3aEhZXV0ynzt28bHiek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xeHDh/UW9oqLizPLvAghxNtvvy1UKpWYPXu22Ldvn5gzZ45wdHQU//vf/4RGozHbvDzv+enz5pyXt956S1hbW4sZM2aIn3/+WUydOlVYW1uL4cOHCyHMNzebN28WCoVC9OrVS0RFRYlFixYJBwcH0aNHDyGE+eYl2+rVqwUAcefOHb33iltuWAgVI4YKISGE+Pvvv8Xrr78ubG1tRZkyZcSgQYNEYmKizjGPHj0SAwYMEC4uLsLJyUl06NBB/PPPP4UYvemsWrVKADD6CAsLE0KYX16EECItLU3MmDFDVK1aVahUKlGhQgUxceJEkZaWpj3GHPPyvOcLISHMNy+pqanis88+E1WqVBEqlUpUqlRJzJo1S2chQXPNzY4dO0TDhg2FSqUSXl5eYvTo0fxv6T9z5swRAERqaqrB94tTbhRC/DdqkIiIiMjMcIwQERERmS0WQkRERGS2WAgRERGR2WIhRERERGaLhRARERGZLRZCREREZLZYCBFRicCVQIgoP1gIEVGxt337dvTv3/+lr7N69WooFApcv3795YMiomKBCyoSUbEXEBAAAIiJiXmp69y/fx9XrlxB3bp1oVKpXj4wIiryLOUOgIioqHB3d4e7u7vcYRBRIeKtMSIq1gICAnDw4EEcPHgQCoUCMTExUCgU+Oabb+Dr6wsPDw9ERUUBAFauXIkGDRrA3t4etra2qFOnDjZt2qS91vO3xgYMGIC2bdsiLCwMVatWhUqlQu3atbF79245vioRFQAWQkRUrC1duhR169ZF3bp1cezYMSQmJgIAPv30U8ybNw/z5s1D06ZN8fXXX2Pw4MHo1q0bdu3ahR9//BHW1tZ488038e+//xq9/u+//465c+fis88+w9atW2FlZYWQkBDEx8cX1lckogLEW2NEVKxVr14dTk5OAIAmTZpoxwkNHToUISEh2uOuXr2K0aNHY9KkSdq2ihUron79+vjll19Qvnx5g9dPSEjAyZMn4efnBwCwt7dHq1atsH//fvTo0aOAvhURFRYWQkRUItWsWVPn9bx58wBIhc2lS5dw8eJF7Nu3DwCQkZFh9Dru7u7aIggAfHx8AABPnjwxdchEJAMWQkRUInl4eOi8vnLlCgYPHoz9+/fDysoK/v7+qFWrFoCc1yCys7PTeW1hIY0o0Gg0Jo6YiOTAQoiISjyNRoNOnTrB2toaJ06cQN26dWFpaYlz587hxx9/lDs8IpIRB0sTUbGnVCpzfP/Bgwe4cOEC3n33XTRs2BCWltK/Affs2QOAvTtE5ow9QkRU7Lm4uODYsWPYv38/EhIS9N4vU6YMKlSogCVLlsDHxweurq6IjIzEggULAHC8D5E5Y48QERV7w4cPh5WVFTp06IDU1FSDx2zduhXe3t4YMGAAevXqhWPHjmH79u3w9/fH4cOHCzliIioquMUGERERmS32CBEREZHZYiFEREREZouFEBEREZktFkJERERktlgIERERkdliIURERERmi4UQERERmS0WQkRERGS2WAgRERGR2WIhRERERGaLhRARERGZLRZCREREZLb+DwLlgNnXvvxcAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "def plot_learning_curve(estimator, title, data,target, cv=5,\n",
    "                        train_sizes=np.linspace(.1, 1.0, 5)):\n",
    "    plt.figure()\n",
    "    plt.title(title) \n",
    "    plt.xlabel('train') \n",
    "    plt.ylabel('accurary')\n",
    "    train_sizes, train_scores, test_scores = learning_curve(estimator,data,target, cv=cv,\n",
    "                                                            train_sizes=train_sizes) \n",
    "    train_scores_mean = np.mean(train_scores, axis=1) \n",
    "    test_scores_mean = np.mean(test_scores, axis=1)\n",
    "    plt.grid() \n",
    "\n",
    "    plt.plot(train_sizes, train_scores_mean, 'o-', color='b',\n",
    "             label='traning score') \n",
    "    plt.plot(train_sizes, test_scores_mean, 'o-', color='r',\n",
    "             label='testing score') \n",
    "    plt.legend(loc='best')\n",
    "    return plt\n",
    "clf = RandomForestClassifier()\n",
    "para_grid = {'max_depth': [5], 'n_estimators': [90], 'max_features': [1, 5, 10], 'criterion': ['gini', 'entropy'],\n",
    "             'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 5, 10]}\n",
    "gs = GridSearchCV(clf, param_grid=para_grid, cv=3, scoring='accuracy')\n",
    "gs.fit(data,target)\n",
    "gs_best = gs.best_estimator_ \n",
    "gs.best_score_ \n",
    "g = plot_learning_curve(gs_best, 'ramdon forest',data,target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "e30710c3-e29f-4595-b31f-b2be4fd589da",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import StratifiedKFold"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "18298d41-076a-4fea-87cf-3b2a7d8d1f4d",
   "metadata": {},
   "outputs": [],
   "source": [
    "skf= StratifiedKFold(n_splits=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "b0834d50-8ef7-477f-be18-52b62fa10a3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "for train,test in skf.split(data,target):\n",
    "    x_train=data[train]\n",
    "    x_test=data[test]\n",
    "    y_train=target[train]\n",
    "    y_test=target[test]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "2170fdec-8441-4913-8f03-f6b8e4b03a46",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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zrxc6vjx4y5t//vf/+rVV1/N0v56eXl5qWPHjvr99981b948l3lTpkyRJN13333y9PTUI488ou3bt+u1115zaRcTE6Nhw4Zp/vz5Cg4OznENN9tLL72ksmXLasuWLXrllVec03O6/1+vXr16yeFwaMSIES5n/hYuXKjffvvNpW1u7TN5qVy5cmratKk2bdqUZZ+ZPn26tm/frrZt217zLF9O/u5PnDihWbNmacaMGS7tjh8/rnPnzrn8RjK7vx8A+QO/oQJQIIwYMUJffvmlZs6cqe+++05NmzZVQkKCPvroI505c0aLFi1yfhPfs2dPTZs2TfXq1VP79u2Vnp6ur776Sjt37lRoaKhOnDihkydPqlSpUpKkiIgI7dixQ4MGDXL+IPzxxx/XihUr1LlzZ/Xo0UMeHh5atmyZKlWq5HLDzasJDAzUvHnz1K1bN9WuXVsdOnRQeHi4YmJitGnTJt1xxx0aNmzYTXvPrkdwcLC6d++ujz76SBEREVq1apX27NmjUaNG6Y477pAkVapUSVOnTtXQoUNVp04dtWvXTj4+Plq5cqXi4uL0+OOPX9clcaGhofL09NS3336roUOHqkOHDnrwwQcVFBSkl156STt27FDFihW1a9curVy50rk9rZ4dmj59un744Qc9/vjj+vjjj1W9enX95z//0ffff6/27ds7B8OYNm2afvzxRw0ZMkSffPKJ6tevr0OHDunjjz9WkSJFNH/+fOdZs/zE19dXc+bMUdu2bTV+/Hh16tRJFSpUyPH+f70aNGigYcOGadq0abr99tv14IMPKi4uTp999pkqVqyoPXv2ONvm1j6T1+bOnavGjRvr8ccf17Jly1S9enX9+uuviomJUWRkpGbNmnXNZeTk775du3Zq1KiR3nzzTW3btk2NGjVSUlKSli1bJofDoQkTJjiXGxERIUkaNmyYWrRooTFjxtycNwFAzt28AQQB4MpyemNfYy7dBHP8+PHmtttuM56enqZEiRKmVatWWe7Tk5qaasaPH2+ioqKMl5eXKVOmjGnVqpX56quvzKuvvmokmXnz5jnbr1y50lSoUMEULVrU9O3b1zl98eLFpnbt2sbT09NERESYESNGmKNHj15x2PTL72dzufXr15u2bduakJAQ4+XlZapWrWpeeOEFk5CQcF39vtKw6dkN636lWv465PLlQ5gvXbrUREVFGW9vb3PHHXeY9957L9s6Pv/8c+fNTP38/EzDhg3NokWLXNpkNzT65ebNm2fCw8ONp6enGTdunDHGmJ9//tncf//9JiQkxPj7+5uaNWua4cOHm1OnTplixYqZsmXLOoePvlL/rrTeo0ePmv79+5vw8HBTpEgRExkZaUaPHu1yHyVjjDl58qQZOnSocz8oXbq0+fvf/25+/fXXbPvxV7rCsOmZ97q63JW23V9dz419jTGmY8eORrp0I1pjcrb/Z3cbgL+u/6/DdM+bN8/UrFnTeHl5mUqVKpn58+c7b9p8+T2njLm+fcaY7G9dYEzO9/O/yumNfY25NHx6nz59TKlSpUzRokVN+fLlzdChQ53DpV9vDdf7d3/q1Cnz3HPPmSpVqhhvb28THBxsHnjggSzDsZ84ccK0bNnSeTPz5OTk6+4TgJvLYYyFC6kBAAXa/v37Vb58ebVr106ffvqp3eUAAFBg5b9rGAAAAACggCBQAQAAAIBFBCoAAAAAsIjfUAEAAACARZyhAgAAAACLCFQAAAAAYBE39v1/GRkZOnLkiPz9/eVwOOwuBwAAAIBNjDFKTk5WeHj4NW/uTqD6f0eOHFGZMmXsLgMAAABAPnHw4EFFRERctQ2B6v/5+/tLuvSmBQQE2FwNAAAAALskJSWpTJkyzoxwNQSq/5d5mV9AQACBCgAAAMB1/RSIQSkAAAAAwCICFQAAAABYRKACAAAAAIsIVAAAAABgEYNSWHTx4kWlpaXZXQbyOQ8PDxUuXNjuMgAAAHCTEKhyyBijY8eOKTExUcYYu8tBPudwOBQYGKiSJUtyw2gAAAA3RKDKocTERCUkJCg0NFS+vr58SMYVGWN05swZnThxQt7e3goKCrK7JAAAAOQyAlUOGGMUHx+vgIAAFS9e3O5yUAB4e3srNTVV8fHxCgwMJIADAAC4GQalyIGLFy/q4sWL3PgXORIQEODcdwAAAOBeCFQ5kJ6eLkkqUoQTe7h+mftL5v4DAAAA90GgsoDLtpAT7C8AAADui0CFAoWRFQEAAJCfcO1aLnGMzx9nIcxYa4EjMjJSzZo10z//+c/cLSgXzZ8/X7GxsZoxY4bdpQAAAACSCFT4f5988km+H2xj4sSJatasmd1lAAAAAE4EKkiS6tSpY3cJAAAAQIHDb6gg6dIlf71799b+/fvlcDi0fPlytW/fXr6+vgoLC9OLL76opKQk9e3bV4GBgQoLC9Nzzz3n/E1T5us++OADtW3bVj4+PipTpozGjRunjIwM53ouXryoN954QzVr1pS3t7fKli2rESNG6Pz58842vXv3VosWLTRgwAAFBQXpjjvuUEREhA4cOKBFixbJ4XBo//79kqR169apVatWCg4OVtGiRVW+fHmXdWbW9dFHH6lTp07y9/dXcHCw+vXrp5SUFOc6jTGaM2eOqlevLm9vb1WqVElTp051+c3W999/r6ZNm8rHx0chISHq1auXTpw4cTM3CwAAAPI5AhWy1bdvX9WsWVOff/657r33Xo0ePVr169eXt7e3PvroIz300EOaOnWqli1b5vK6AQMGKDAwUB9//LF69eqliRMnavjw4c75/fv31+DBg9WuXTutWLFCTz75pGbNmqV27dq5hJd169Zp165d+vjjjzVmzBh9/vnnKlmypB544AH99NNPKlWqlLZu3aoWLVqoWLFiWrp0qT7//HPdfffdGj9+vD744AOXuvr376/IyEh9+umnevbZZ7VgwQJNmjTJOf/555/X4MGD1aZNG61YsUKPP/64nn/+eb344ovOelq0aCEfHx99+OGHevXVVxUTE6PmzZvr3LlzN2MTAAAAoADgkj9kq3Xr1po4caIkqVq1avrggw9UokQJzZ49W5LUsmVLffjhh/rxxx/18MMPO193xx136P3333cuIyUlRbNmzdLo0aN15MgRzZ8/Xy+++KJGjRrlXE54eLh69uypL7/8Un/7298kXbpn01tvvaVKlSo5l+3p6anQ0FA1bNhQkvTf//5XLVu21Pvvv69ChQo5l7dy5Up999136tatm/O1bdq00fTp0yVJLVq00Ndff62VK1dq8uTJSkhI0IwZM/T0009r6tSpzuXEx8fr+++/lySNHDlSVapU0cqVK1W4cGFJUsOGDVWtWjUtWLBAgwYNys23HwAAAAUEZ6iQrbvuusv5/yVLlpQkZ5CRLt1bKTg4WAkJCS6v69Gjh8vzjh07Ki0tTRs2bNB3330nSerevbtLm0ceeUSFCxfWt99+65zm7e2tihUrXrXGnj17avXq1bpw4YK2b9+uTz/9VOPGjVN6erouXLjg0rZRo0YuzyMiInTmzBlJ0oYNG5SWlqYOHTq4tJkxY4bWrFmjs2fPasOGDWrTpo2MMUpPT1d6eroqVKig2267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5OnpqbFjx2a53LCgoj/5lzv1RaI/+Zk79UWiP/mZO/VFoj/5mTv1RaI/N5vDcOdbAAAAALDE9kv+AAAAAKCgIlABAAAAgEUEKgAAAACwiEAFAAAAABYRqAAAQL73559/2l0CAGSLQJVH1q5dq+eff179+/fX6NGjtW7dOrtLsuzgwYNatWqV83lGRoYGDBig/fv321cUbjkrV660uwS4obS0NB0/flwXL160uxRISk9P16hRoxQYGKhy5cpp7969uvPOO3X06FG7S7vlxcXFZXnEx8crIyPD7tKQjdOnT+uXX35RRkaGLly4YHc5bodAdZOdP39ebdq0UatWrfTxxx9r+/btWrp0qZo1a6aHH364wB149u7dq3r16unjjz92Tjt9+rQ2btyoRo0aae/evTZWl3PJycn65ZdflJ6eLkl644039OCDD2ratGk2V2ZNRkaGTp486Xz+7bff6pVXXtHOnTttrCr3XLhwQfPmzdNtt92mdu3a2V1OjhljtGfPHpdpS5cu5cN7PpCSkqJevXopMDBQ4eHhCgoK0pNPPlngP3hkZGTo+PHjSktLc077888/lZqaamNV12/cuHH65ptv9NFHH6lo0aIKCwtTRESEBg8ebHdpN+Q///mPlixZonfffdflUZBERkaqfPnyLo9SpUopMDBQgwYNKrDHtcvr/uKLL7Rp0yYbq7lxKSkp6tatm4oVK6YmTZpo165dqlixYoH9XPDOO++oVq1aKl68uOLi4tSpU6cs9621hcFNNWLECFOzZk2zc+dOl+mxsbGmVq1aZsqUKTZVZk3Pnj3NU089le283r17m0cffTSPK7Luu+++M4GBgcbhcJioqCgzZ84cExQUZDp27GhCQkLMxIkT7S4xRw4dOmSqVatmHnvsMWOMMYsXLzaFCxc2devWNYGBgWbTpk02V2jdn3/+aSZMmGDCwsJMcHCwefrpp8327dvtLitHUlJSzF133WU6dOjgnHb8+HHj5eVlGjdubFJSUmyszpo1a9aYtm3bmjvuuMMcPXrUDB061KSlpdldliU9e/Y0DRs2NGvWrDE7duwwq1evNnfeeaeJjo62uzRLjh49arp06WI8PT1NoUKFjKenp+nYsaM5cOCA6devn5k3b57dJV6XyMhIc+jQIWOMMcHBwcYYY06fPm2KFStmZ1k3ZNSoUaZQoUImPDzcREZG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HQvckMnuFN7//e9/a/bs2fryyy+Vnp6uBQsWqFu3blluUFzQuNMIme5yjJbc74siSRowYIDmzZun8PBwl7OjDoejwPVpwYIFGjRokC5cuOCcZozJF2d3CVR5pESJEjp69GiB/0cgkzv1x536Il368ez+/ftdfnB6/vx5lS5d2nnX94LE3bZPQQ+8rVu31pdfful8fu7cOXl7eys9PV0XLlyQj4+PjdXlXPPmza86vyB+y+5u4V2SDhw4oDfeeEMLFixQoUKF1KNHD82YMcPusiy52sAHR44csakq69zpGO0uXxRdLjg4WGvXri3QQ79nCg8P18iRI/Xggw9muXTW7htjE6jySHR0tEqWLKkRI0bYXUqucKf+uFNfpEuX8Pz8888uB5cdO3aoadOmOn78uI2VWeNu26egB96//uYmJCREp06dsrEiXElBD+/ZSU1N1eLFi/XGG2/o559/trscSypVqqRBgwZlO0Lms88+a3d5OeZux+jLFdQvii4XGRmpnTt3usVgVCEhIfrzzz+v+DtEOxGo8sg999yjH3/8UT4+PipRooTLvIJ2ylVyr/64U18k6ZlnntHq1av17LPPqkyZMtq7d6+mTp2qHj16aPz48XaXl2Putn0KeuD9a6AKDg7W6dOnbazoxvzrX/9S165d9e6772Y73+FwqGfPnnlcVe4o6OHdXbnbwAfucIy+0t//5Qri740kad68efruu+80fPhwBQUFucyz+3dHOfX000+ratWqGjhwoN2lZEGgyiOLFi264rxevXrlYSW5w5364059kS59ozZhwgS9//77Onz4sMqUKaN+/frpueeec7kEqKBwt+1T0AOvu52hqlGjhn777TeVL18+2/kF8XcGmQp6eHdX7jbwgTsco6/095+pIB8HZs+erSFDhrgM4pBffneUU99++63uv/9++fv7ZwmHdm8fAhUA5KGCHnjdLVBJl24ge+rUKRUvXlyS9M0332jLli1q06ZNgb6pb0EP7+6qS5cu8vb2dquBD9zV+fPn5eXlZXcZNyQsLEzjx4/X/fffn+V3bnb/7iinKleurDvvvFMtWrTI0he7wzuB6iZr06aNVq1apebNm1/xw1JB+sGzO/XHnfryV19//bVmz56tQ4cOadWqVZo+fbqmTJmiIkWK2F3adXPn7VOQeXt7a+7cucr8p2PgwIF68803dfk/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",
      "text/plain": [
       "<Figure size 1000x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor \n",
    "from sklearn.feature_selection import SelectFromModel \n",
    "clf=RandomForestRegressor(n_estimators=50) \n",
    "clf.fit(x_train,y_train)\n",
    " \n",
    "features = pd.DataFrame() \n",
    "features['feature'] = ['Pclass','Sex_male','Sex_female','Age','Sibsp','Parch','Fare','Embarked_C','Embarked_Q','Embarked_S','Cabin_A','Cabin_B','Cabin_C','Cabin_D','Cabin_U'] \n",
    "features['importance'] = clf.feature_importances_ \n",
    "features.sort_values( by=['importance'],ascending=True,inplace=True) \n",
    "features.set_index('feature',inplace=True)\n",
    "features.plot(kind='bar', figsize=(10, 6), fontsize=10, color='g')\n",
    "plt.xlabel(\"Feature\")\n",
    "plt.ylabel(\"Importance\")\n",
    "plt.title(\"Feature Importance in Random Forest\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "58cdc623-1d71-4b84-9a9c-b1513180b047",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of features: 15\n"
     ]
    }
   ],
   "source": [
    "print(f\"Number of features: {len(features)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "ec95a8a5-9a87-4428-8695-b380a1f8b839",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Feature importances length: 15\n"
     ]
    }
   ],
   "source": [
    "print(f\"Feature importances length: {len(clf.feature_importances_)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "65eee4c7-8c27-49a2-bafa-41bdce06bc4b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "e3398be0-ddbf-41fc-9a27-9e8b8cc673ae",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = LogisticRegression()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "cc2ad082-2f48-4599-9247-815a584c13f9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    },
    {
     "data": {
      "image/png": "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dDSa6yruS4qHxMWSxCVlEhvrHoxLYmIiSilMJpPFX/137oC3d8b/4amUbinKnfvxzouJMZEr1+OdU7ZsWfPvpZo1awKY77Ffv34WvRgnT54kODiYESNG4OXlhcFgwN/fnxo1arBnzx66du1qPjcpVkopbt68yf79+wkMDAT0asmNGzdm69atvPTSS+k657333sPb25uNGzfi4eEBQOnSpalXr575Z2XNrl27aNq0KZ07dwYwtyjlzZsXk8nEggULOHXqFL/88guVKlUCoG7dulSrVo0dO3Zw8uRJtm7dyuLFi3n11VcBaNq0Ke7u7owbN47evXtTo0YNABwdHfnyyy/J9d8PZcuWLWzatImlS5fSpUsXAIKCgrh9+zajRo3ilVdesbpOUFJMjEZjihuwpvXfr00ToaQfSkqzuR7VijNnzhxMJhODBw9O9tqQIUPw8/Nj6NChj1WnqVOnMnHixGTlW7ZsMX+w0svVFUaOLMTcuRX499/7Y5/y5btL795/4Op6iYdaOalaVT/u3nXizz/zcfCgL7//7svff3vzyy8GfvkFQkPBxSWBcuVuU6lSSU6d+pGAgBhkpXdLERERtq5CtiRxSZnExrqIiAicnJwoWLAgt2/fJj4+3vzanTsAPraqWjIxMTHpmo2bkJBgPh8gNjYWgMDAQIs/0seNGwfoP75//fVXTpw4we7du83nxsTEcO/ePQBu375NTEwMRqOR/Pnz4+vra75W0h/z//77b7rP2b59O0FBQSQkJJiPKV++PMWKFcNoNKbYuPD8888zb948zp49S4sWLQgKCuLtt98238P27dvx9/enePHiFtdImrg0fvx4HB0dadmypcXrL774IuPGjWPv3r2ULVuWhIQE/P39SUxMNB/3/fffYzAYqF+/Ptce6Opo1qwZS5Ys4ccff6RChQrJ6hwfH8/du3fZvXu3+Wf1sKSf2aPYNBFK+iE+/MO5desWALkfkfqHhYXRvHnzZAPK1q9fz/Lly/n555/N2XRS0pWQkICDg0OKSdbo0aMJDg42P4+JiaFo0aI0b948WctVerRqBRMmwN69CeYZEfXqOePoWAWokuq5L710//uLF41s325g2zYHtm83cOmSEwcPPsPBg88A5fH1VTRurGja1ETTpopixZ646jmW0WgkIiKCoKAgWQX3ARKXlElsrHswLomJiZw7dw5P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nMTEROXl5aVGjx6tlFJq/PjxKleuXMpoNFoc16pVK/NfNznJ9evX1cCBA9XevXstyg8ePKgAtXz5cruMi9FoVAcPHlT//POPRXlQUJCqUqWKUso+Py8PatmyperSpYvq3r27RSJkz3F58803VeXKlVN83R5jc/XqVWUwGNTixYtTPMYe42JNbGysKlq0qGrdurW5LKfFRsYIZXPFihXjxIkTfPzxx3h4eCR7/ciRI5QuXdqizMHBgeLFi3Ps2DHzMSVKlMDJyXJruZIlS5qPyUl8fHz49NNPqVu3rkV5eHg4AM8995xdxsXJyYlKlSpRoEABlFL8888/TJ06la1btzJgwADAPj8vSebOncsvv/zC//73v2Sv2XNcDh48iIODA0FBQeTKlYu8efPy5ptvcuvWLcA+Y/P777+jlOKZZ57h1VdfxcvLC09PT1577TWuX78O2GdcrPnkk0+4ePEiM2bMMJfltNhIIpTN5c2bN9VNX2/cuGF1IzsvLy/zwOG0HJPTRUZGMn36dNq3b0/58uXtPi5Lly6lUKFCjBkzhpYtW9KlSxfAfj8vZ8+eJTg4mNmzZ5M/f/5kr9trXEwmE4cOHeL48eN07NiR77//nrFjx7Js2TJatWqFyWSyy9hcuXIFgF69euHu7s6aNWv48MMP2bBhg13H5WHx8fHMmjWLV155hZIlS5rLc1psZPf5HE4phcFgsFru4KDzXJPJ9MhjcrI9e/bQtm1bAgMDmTdvHiBxqVWrFrt27eLo0aOMGzeO559/np9++sku46KUolevXrRq1YqXXnopxWPsLS6g675hwwYKFixI2bJlAWjQoAEFCxbktddeY/PmzXYZm/j4eACqVavG3LlzAWjatCk+Pj507dqViIgIu4zLw1atWsXly5cZMWKERXlOi03O/0nYudy5c1vNnm/fvk3u3LkB3ZX0qGNyquXLlxMUFIS/vz/btm0jb968gMSlZMmSNGjQgD59+rBkyRIOHTrE6tWr7TIun332Gb///jszZswgISGBhIQE1H87CyUkJGAymewyLgCOjo40atTInAQlad26NQC//fabXcbGy8sLgDZt2liUt2jRAtDdifYYl4eFhYVRvnx5KlWqZFGe02IjiVAOV6ZMGU6cOGFRZjKZOH36NOXKlTMfc/r0aUwmk8VxJ06cMB+TE4WGhtKtWzdq167N7t27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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "def plot_learning_curve(estimator, title, data,target, ylim=None, cv=None, n_jobs=1, \n",
    "                        train_sizes=np.linspace(.1, 1.0, 5), verbose=0, plot=True):\n",
    "    train_sizes, train_scores, test_scores = learning_curve(\n",
    "        estimator, data,target, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)\n",
    "    train_scores_mean = np.mean(train_scores, axis=1)\n",
    "    test_scores_mean = np.mean(test_scores, axis=1)\n",
    "    if plot:\n",
    "        plt.figure() \n",
    "        plt.title(title)\n",
    "        if ylim is not None:\n",
    "            plt.ylim(*ylim)\n",
    "        plt.xlabel(u\"train\")\n",
    "        plt.ylabel(u\"accurary\")\n",
    "        plt.gca().invert_yaxis()\n",
    "        plt.grid()\n",
    "        plt.plot(train_sizes, train_scores_mean, 'o-', color=\"b\", label=\"training score\")\n",
    "        plt.plot(train_sizes, test_scores_mean, 'o-', color=\"r\", label=\"testing score\")\n",
    "        plt.legend(loc=\"best\")\n",
    "        plt.draw()\n",
    "        plt.gca().invert_yaxis()\n",
    "        plt.show()\n",
    "plot_learning_curve(model, \"logicistRegression\",data,target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "c8de0e40-dd66-49c9-984f-6c7bbf9c5f0c",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import SVC, LinearSVC\n",
    "from sklearn.model_selection import cross_val_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "e8f6d9c5-4865-4f67-b1c5-3d12b520b198",
   "metadata": {},
   "outputs": [],
   "source": [
    "model1= SVC()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "5fbd4c42-2f91-4dd9-bcbc-3f548dfbf983",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "def plot_learning_curve(estimator, title, data,target, ylim=None, cv=None, n_jobs=1, \n",
    "                        train_sizes=np.linspace(.1, 1.0, 5), verbose=0, plot=True):\n",
    "    train_sizes, train_scores, test_scores = learning_curve(\n",
    "        estimator, data,target, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)\n",
    "    train_scores_mean = np.mean(train_scores, axis=1)\n",
    "    test_scores_mean = np.mean(test_scores, axis=1)\n",
    "   \n",
    "    if plot:\n",
    "        plt.figure()\n",
    "        plt.title(title)\n",
    "        if ylim is not None:\n",
    "            plt.ylim(*ylim)\n",
    "        plt.xlabel(u\"train\")\n",
    "        plt.ylabel(u\"accurary\")\n",
    "        plt.gca().invert_yaxis()\n",
    "        plt.grid()\n",
    "        plt.plot(train_sizes, train_scores_mean, 'o-', color=\"g\", label=\"training score\")\n",
    "        plt.plot(train_sizes, test_scores_mean, 'o-', color=\"r\", label=\"testing score\")\n",
    "        plt.legend(loc=\"best\")\n",
    "        plt.draw()\n",
    "        plt.gca().invert_yaxis()\n",
    "        plt.show()\n",
    "plot_learning_curve(model1, \"SVC\",data,target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "id": "4b5c0ab0-f0aa-4881-addb-23e78063948d",
   "metadata": {},
   "outputs": [],
   "source": [
    "skf= StratifiedKFold(n_splits=5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "cb40044a-b6ee-4edf-b9dd-c11b729d6272",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression #logistic regression\n",
    "from sklearn import svm #support vector Machine\n",
    "from sklearn.ensemble import RandomForestClassifier #Random Forest\n",
    "from sklearn.neighbors import KNeighborsClassifier #KNN\n",
    "from sklearn.naive_bayes import GaussianNB #Naive bayes\n",
    "from sklearn.tree import DecisionTreeClassifier #Decision Tree\n",
    "from sklearn.model_selection import train_test_split \n",
    "from sklearn import metrics \n",
    "from sklearn.metrics import confusion_matrix "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "deaf5f13-342e-4e25-80b2-1777eb69634d",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import KFold #for K-fold cross validation\n",
    "from sklearn.model_selection import cross_val_score #score evaluation\n",
    "from sklearn.model_selection import cross_val_predict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "f1af7385-bb2d-4a07-a9c9-f2b4d1795d28",
   "metadata": {},
   "outputs": [],
   "source": [
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "27d0f99b-cdb7-4298-a0d7-d428e775dec9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n",
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Text(0.5, 1.0, 'confusion logisticregression')"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA8YAAANGCAYAAAA/IZcZAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy81sbWrAAAACXBIWXMAAA9hAAAPYQGoP6dpAAC74klEQVR4nOzdeVzU1f7H8feMCijCoEC5UAqSqJl6y0pTFBewsNIwuylWXM0y2xSTMi20DZd2y9KrYal1r7fQFrxhllqUZWkuhWmYmWUWLgi4kDrn94c/5jYODmAjy/B63sf38cjzPfP9nu/A5cxnPmexGGOMAAAAAACopaxV3QAAAAAAAKoSgTEAAAAAoFYjMAYAAAAA1GoExgAAAACAWo3AGAAAAABQqxEYAwAAAABqNQJjAAAAAECtRmAMAAAAAKjVCIwBAAAAALUagTFqlGXLlqlDhw7y8/NTSEiIVq5ceVbuExMTI4vFovz8/LNy/fJ65513dPXVV+ucc86Rr6+vmjZtqgEDBuidd95xqpeWliaLxaKxY8eWec2EhARZLBZ9+umnTuXHjh3T66+/rn79+um8886Tj4+PmjVrphtuuEGfffaZR58LAIDTqS19/eTJk2WxWDR//vxSz+/atUstW7aUxWLRuHHjJEk//vijLBaL/Pz89N1335322tdff70sFot+/PFHR1nJ86ampp72de+9954sFosmT558Jo8E1GgExqgxDhw4oMGDB2vr1q1KSkrS8OHD1bZt27Nyr6SkJKWmpsrPz++sXL887r77bg0YMEDffPONrr32Wo0dO1axsbHKzs7WgAEDNGrUKEfdm266SVarVf/5z39kjDntNQ8ePKhly5apVatW6tatm6P8559/Vo8ePZSYmKht27apb9++Gjt2rK644gq9/fbb6t69u1588cWz+rwAANS2vv509uzZoz59+mjnzp2699579dRTTzmdLy4u1siRI932+aczdepU5eTkeKqpgNeoW9UNAMrru+++0+HDhzV06FC9/PLLZ/VeSUlJZ/X6ZVm1apVeeOEFDRw4UIsXL1a9evUc5w4ePKhevXpp9uzZio+P17XXXquwsDD16tVLH374obKzsxUdHV3qdd98800VFxfrlltucZQdPXpUV155pb799ltNmTJFEyZMcLrf9u3b1bNnT919990KDw9XfHz82XtwAECtVpv6+tPZt2+f+vbtq++//1533XWXnn322VLrZWdna86cObr99tsrdP0//vhDI0eOVHZ2tiwWiwdaDHgHMsaoMYqLiyVJoaGhVdySs++9996TJN17771OQaok2Ww2TZ06VZL01ltvOcpvvvlmSdK//vWv01530aJFslgsjrrSyWHY3377rUaOHKmHH37Y5X6tWrXS/PnzZYzRI4888tceDAAAN2pTX1+agwcPKi4uTt9++63uuOMOzZw5s9R67dq1k4+Pj+6//37t3r27Qvf429/+ps8++0yzZs3yRJMBr0FgjArZu3evxo4dq/DwcDVo0ECtW7fWQw89pKKiIqd6u3bt0siRI9W8eXP5+PioRYsWuvfee7V3716neklJSbJYLDpw4IDuuOMONWnSRH5+furcubNT0BcTE6NevXpJkp577jlZLBbHN70Wi0WdOnVyaev8+fNlsVicvmktLCzU2LFj1aZNG/n5+emcc85RQkKCvvrqK6fXljbv6MSJE3r++efVsWNH+fn5KSgoSFdddZU++eQTp9euWrXKMWfolVde0UUXXSQ/Pz+FhYXpvvvu0+HDh8t8n48dOyZJ+vbbb0s9Hx0drcWLFzvNKR40aJD8/f315ptv6sSJEy6v+eWXX7R69WrFxMSoRYsWkiRjjNLT0yVJEydOPG17+vbtq6lTp2ry5MlnNGwLAFBz0NdXTl9/qkOHDik+Pl7r16/Xbbfd5nYK0wUXXKAHH3xQBw8e1F133VWh+7z00kvy9fXVhAkT9PPPP1e4nYDXMkA57d6925x//vlGkundu7cZN26c6dWrl+Pfx44dM8YY891335mQkBAjycTFxZnk5GTTs2dPI8m0bNnS7N6923HNW265xUgyl1xyiWnRooW5++67zfDhw42vr6+xWCzmk08+McYYk56e7qh7+eWXm9TUVLNkyRJjjDGSTMeOHV3am56ebiSZZ555xlEWFxdnJJmrr77a3H///eaWW24xfn5+pn79+iYnJ8dRr6S9Bw4cMMYYc+LECTNgwAAjyURGRpo777zTJCYmmoYNG5o6deqYBQsWOF67cuVKxzPVq1fP3HjjjWb8+PGmVatWRpK59dZby3yv3333XSPJ+Pr6mnvvvdd8/vnn5vjx42W+7uabbzaSzIoVK1zOzZgxw0gyr776qqNs8+bNRpJp06ZNmdcGAHg/+vrK6+tTU1ONJJOenm6OHDlievfubSSZESNGGLvdXuprduzYYSSZAQMGmOLiYtOuXTsjyWRkZDjVGzRokJFkduzYUerzPvroo0aSufbaa51eV/L5IzU1tcz2A96GwBjlNmzYMCPJPP/8807lI0aMMJIcnVfJH9709HSnemlpaUaSSUhIcJSVdICXXXaZKSoqcpQvWrTISDI33XSTo6ykE7r33nudrlveznLTpk1Gkrn55pud6v3nP/8xksy4ceMcZad2liXXio+PN4cOHXLUy8nJMUFBQaZ+/fpmz549Tu2sU6eO+eyzzxx18/PzTWhoqKlfv77Ts57OHXfcYSQ5jsDAQBMfH2+eeeYZs2vXrlJfs2LFitN2yJ06dTINGzZ0uveyZctK7RgBALUTfX3l9fUlgfGcOXNMfHy8o79/4403TvuaPwfGxhiTnZ1tLBaLadasmcnPz3fUKysw/uOPP0z79u2NJLN48WJHHQJj1GYMpUa5FBcXa8mSJYqKitLdd9/tdG7SpEl68MEH1bRpU/30009avXq1evbs6bKoRUpKiqKiorRkyRLt37/f6dxdd90lf39/x79LFnjatm2bx57B/P8Q4G+//dbp/gMHDtQPP/ygadOmnfa1r732miTphRdeUIMGDRzlbdu21X333acjR47ojTfecHpNz5491bVrV8e/bTabrrjiCh05ckS7du0qs72zZs3S22+/rdjYWNWrV08FBQVatmyZxo4dq4iICE2cOFF2u93pNb169dJ5552njIwMx3BsScrJydGGDRt0/fXXO73PJcPHAgICymwPAMC70ddXfl8vSQ899JCWLVumK6+8UlarVaNHjy73EOdu3bpp1KhR2r17t+6///5yvUaS6tWrp3/+85+yWq265557qnx7SqA6IDBGuWzfvl2HDh3S5Zdf7nKuZcuWevzxx3X55Zdr48aNklTqqshWq1Vdu3aVMUabN292Ote6dWunf9tsNkn/W4TDEzp06KBu3bpp3bp1at68ufr166dnnnlGP/74o8LDw1WnTp3Tvnbjxo0KCwtTeHi4y7nu3bs76vzZqc8kVfy5rr32Wi1fvlz79u1TZmamkpOTFRkZqWPHjumJJ57QQw895FTfarVq2LBh2r9/v1asWOEoX7RokSQ5rUYtScHBwZJObo8BAKjd6Ourpq//7bffdNNNNykzM1NjxozR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rG7duumOO+7QE088oU6dOumSSy5Rv3791L9/f11++eWyWCxu2wsAZwP9c9X0z3++v9VqdekrS7t/yXzp0n5W3bp10wcffOD2Xp7sg44cOaLt27erR48e8vX1dToXEBCgadOmubzm1Pd43bp1kk7OMS6tDw0ICNDGjRtljHE8e5cuXVzqdevW7bTtPN09v/rqq1LvWadOHcdnmtjYWEVGRmrZsmU699xz1bt3b8fv1Z8/F5W33qkKCwu1Y8cOde/eXQ0bNnQ53717d82dO1cbN27UwIEDHeVl/f7Vhq3IUDEExtXMgQMHJEmBgYFu6xUUFLit16xZM0lyWeDi1D/KJX/gzSmLKPxVWVlZmjZtmhYuXKjly5dr+fLlSk5OVkxMjNLT09WyZctSX1dQUHDavf/K+0xSxZ+rfv36LmUZGRl64IEH9P3330s62fFccskl6tSpk5YvX+5y7fK04+DBg45rnapx48ZO/z7Tn3FJIHyq0tpXHmfyMzn1/SxZvOW7777TlClTTnuv/fv3S5Iee+wxtWrVSi+//LK++uorffnll46y2bNnq0+fPmf0LABwpuifq6Z//vP9/fz85OPjU+b99+7dK39//1KDqJK67niyDyrp18r6vfmz0/Wh77//vt5///3Tvq6oqMixuNu5557rct7ds5/unv/6179O+5qSZ6tfv74+++wzPf7441q8eLHefvttvf322xo9erQGDhyouXPnqnHjxuWudypP/X9KOnv/v4J3YFXqaqbkj3hhYWGp5w8dOiTpf4HV7t27S61X0oEHBwd7tH2l/SEpbXVJf39/PfLII/rhhx+0detWzZw5U126dNGqVav097///bTXDwgIqPRnKs0XX3yhwYMH6+jRo1q4cKF++OEHHTx4UCtXrvxL+/s2atRI0v8C5D87dWXmqvoZn8oTP5OS3+ubbrpJ5uQUjlKPa665RtLJjmv48OFau3at9uzZo0WLFun666/XDz/8oGuvvdZlVVcAONvon6u2fw4ICNDhw4dL7T9PvX9gYKCOHj1a6mrNJUGWO57sg8r7e1Oea8ybN89tH1qygrZU+nOW59lPveeHH3542vvt27fPUT80NFTPPvusfvnlF23YsEHTpk1Tu3bttGTJEt1xxx0Vrvdn1eXzELwfgXE1ExUVJR8fH61du9bl3E8//aSGDRvqtttuU8eOHSVJn376aanX+fjjj1WvXr1Sh5GcKR8fH5ftKKSTQ3v+bMOGDRo3bpw+//xzSSeHstx1113Kzs7WBRdcoLVr1+qPP/4o9R6dOnVSfn6+cnJyXM59/PHHkqQLL7zwrz5Kmd544w3Z7Xa99NJLSkxMVHh4uONbxm+//VbSmX3beMkll0gq/ee2fv16p39Xxc+4NJ74mZRsrbB+/fpS37dnn31Wjz32mPbt26e9e/fq4Ycfdmwddc4552jo0KH6z3/+o3/84x86fPiw471iWDWAykL/XLX9c8n2R6W9r6fe/5JLLtGJEydctqCSTn7x7Y6n+yCbzabzzjtPGzZscHlvi4uLFRoaqri4OLfXKPmdKhne/GfHjh3TuHHjNHPmTEn/+5xR2nOW9ezlvef+/fs1ZswYLViwQNLJ7bnuuecebd++XRaLRR07dlRKSorWrl2rhg0bOrbzKm+9UwUGBio8PFxbt24t9UuJyvx8CO9GYFzN+Pn5adCgQdqyZYvmzp3rdG7q1KmSpL59+6pFixbq2bOnvvzyS5d6Tz75pL799ltdc801LnvA/RVt2rTRjh07HIGhJO3cuVOvvfaaU71jx47p6aef1qOPPuoUBBUUFOjAgQNq0qRJqUOhJOnmm2+WJI0ZM8ZpX98tW7Zo+vTpatCggQYNGuSxZzqdkiFFv/32m1P5hx9+qEWLFklSqd9El+XSSy9Vu3bttGjRIn322WeO8l9++UVPP/20U92q+BmXxhM/E19fX91444369ttv9dxzzzmdW7Vqle677z7NmzdPjRo1ks1m08yZMzVx4kTHMK0SO3fulHTyvZGkunVPzgY5k58FAFQE/XPV9s8l958wYYJTpnL37t2aOHGirFarEhMTJUn/+Mc/JEmTJk1yypqvXLlSS5YscXufs9EHDRs2TAcPHtQjjzziVP7cc8/pyJEj6tu3r9vXR0dHKyIiQnPnznUJbqdOnaqnn35aX375pSSpa9euatOmjRYuXOj0hXt+fr4efvhht/f5s+uuu06BgYGaNm2ayxcsKSkpeu655xxTzfLy8jRz5kw99dRTTvV+++03HTlyxPF+lbdeaW6++WYdOXJE9913n9MaL9nZ2Zo3b56aNm36l0b0ARJzjKulJ598UtnZ2Ro5cqQyMjJ04YUX6osvvtAnn3yigQMHOjZenzNnjrp3766RI0fqzTff1IUXXqj169dr1apVatmypePbQ08ZOXKk7r77bvXq1UtDhw7VkSNHtHjxYl100UVO3/JdeumlGjRokN566y1dfPHF6t27t44dO6alS5dq7969mjdv3mnvccstt+jtt9/W0qVL1aFDB1155ZXKz8/X0qVLdeTIEaWnp592jpMn/f3vf9dTTz2l0aNHa/Xq1WratKk2bdqkrKwshYSE6Pfff3fqmMvLYrHolVdeUd++fdW7d29df/31CggIUEZGRqnzjiv7Z1waT/1MZsyYoU8//VRjx47VkiVLdNlll+nnn39WRkaG6tatq3nz5slqtcpqterRRx/V3Xffrfbt2+u6665TgwYNtHr1an355Ze6+eabFRUVJUkKCwuTJC1evFgNGzbUzTffzDfGAM4a+ueq659jYmJ0zz336Pnnn1eHDh10zTXX6Pjx43r77be1d+9ePf74446scteuXTVq1Ci9/PLL6tSpk6666ir99ttveuuttxQUFKS9e/eqTp06pd6nXr16Hu+DHnzwQWVmZurxxx/X6tWrdfnll2vLli1atmyZLr30Uo0ZM8bts9epU0evvvqqrrzySnXv3l0DBw5URESEvvrqK3300Udq0aKF0tLSJJ38nDF79mzFxsaqe/fuGjRokGw2m959913HlwSne/Y/s9lsmjt3roYOHaqOHTvquuuuU7NmzbRq1Sp9+eWXuvjii3XfffdJkgYMGKCuXbvqpZde0ubNm9W1a1cVFBTozTfflMVicXwhUN56pXnggQf0/vvv69VXX9WGDRvUq1cv/fLLL1q6dKnq1aun11577bRf6gDldraXvcaZ+fXXX83tt99umjVrZurWrWtatmxpHnroIZel/nfu3GmGDx9umjZtanx8fEx4eLgZN26c0zYMxpS+DU8JnbKtgLstcJ599lnTunVr4+PjY1q1amWmT59u1q1b57RNgzEnN3FPS0sz7du3Nw0bNjQBAQEmJibGvPvuu07XO3U7BmNO7vX7zDPPmIsuusj4+vqa4OBgc80115js7Gyn17prp7vnLev+JT744APTrVs3ExgYaBo1amT+9re/mccee8z8+uuvxmq1mu7duzvqnrqtQFlt3Lhxo7nmmmuMzWYzQUFBZsSIEWb16tVO2zWV+Ks/45K9/0q2MShx6jYe7q5R3p9Jadf8s3379plx48aZiIgI4+PjY5o3b24SEhLM+vXrXer+61//MldccYUJDg42fn5+pkOHDuaZZ55x2lfSmJPbhISEhJj69eubV155pdT7AoCn0D9Xbf88f/58c9lll5n69esbm81mevfubd577z2X1x8/ftzMmDHD8Z6Eh4ebZ555xqSkpBhJ5quvvnLUPfV9Nuav9UGlXe/gwYNm/PjxpmXLlqZevXqmWbNm5u677zb5+fnlfm++/fZbM2TIEHPOOec4fs533323+fXXX13qfvrpp6Z3797G39/fBAYGmsTERMfP5c/bPrn7HGSMMZ999pm55pprTOPGjY2fn59p06aNmTRpklO7jTm5T/L9999voqKiTP369U2jRo1MfHy8yzZQ5a1X2ueqQ4cOmdTUVHPBBRcYHx8fc+6555ohQ4aYb775xqmeu88iZT0vajeLMSzLBgAAAO+wZ88e+fj4lLrC8S233KLXXntNe/bsKXXl5pru6NGj2rNnj8477zyXzPDKlSvVu3dvTZs2TSkpKVXUQqD6Yo4xAAAAvMbChQsVHBzsWECrxPbt27VkyRK1a9fOK4Ni6eTq1+Hh4YqNjXWaR37ixAnHWia9evWqquYB1RoZYwAAAHiNn3/+WRdddJEOHz6sAQMGKDIyUr/++qsyMjJUXFys//73v14dHF5//fV66623dMkllygmJkYnTpzQBx98oG+//Va33XabZs+eXdVNBKolAmMAAAB4ldzcXKWlpemjjz7Sr7/+qqCgIEVHR2vChAm6+OKLq7p5Z1VxcbFefPFFvfbaa/rhhx8knVy5/NZbb9XIkSPZ6hA4DQJjAAAAAECtxhxjAAAAAECtRmAMAAAAAKjVCIwBAAAAALVa3apuwJ9NZjEAVHOTj+yt6iYA7vkFe/ySnvrbPJklLSBJur2qGwC4ZZlS1S0A3DOpZ2NlcU/9ba65q56TMQYAAAAA1GrVKmMMAKh++AYVAADv5qkxXTV5/C+BMQDArZrcyQEAgLJ5arZTTZ4ZS2AMAHCLjDEAAN6NVUD4vAMAAAAAqOXIGAMA3OIbVAAAvJvHNo5gKDUAwFvV4D4OAACUA0OpCYwBAGUgYwwAgHfzWMa4BuPzDgAAAACgViNjDABwi6HUAAB4NxLGBMYAgDIwtAgAAO/GUGo+7wAAAAAAajkyxgAAt/gGFQAA70bCmMAYAFAG5hgDAODdGEpNYAwAKAMZYwAAvBtxMZ93AAAAAAC1HBljAIBbfIMKAIB3Yyg1n3cAAGWweOgAAADVk/HQ8VckJCSoZcuWTmVdunSRxWJxOT7//HNHncLCQo0aNUpNmjSRv7+/YmNjlZOTU+H7kzEGALjFN6gAAHi3qs4YL1y4UEuWLFGLFi0cZXa7XZs3b9b48eOVkJDgVL99+/aO/x4yZIjWrl2r6dOnKzAwUFOmTFHv3r2Vk5Ojxo0bl7sNBMYAAAAAgCqxe/du3XPPPQoLC3Mq37Ztmw4fPqz+/furS5cupb52zZo1yszMVGZmpuLj4yVJ0dHRCg8P16xZszRp0qRyt4NEAADALYZSAwDg3apyKPWtt96quLg49enTx6l8w4YNkqSOHTue9rVZWVny9/dXXFycoyw0NFQ9e/bUsmXLKtQOAmMAgFtWDx0AAKB6MsYzR0XNnTtX69at0wsvvOBybsOGDbLZbBozZoyCg4Pl5+en+Ph4bd261VFny5YtioiIUN26zgOhIyMjtW3btgq1hc8qAIBqr7QFObZu3ar+/fvLZrMpODhYI0aMUH5+vlMdTy3IAQAAylZcXKyCggKno7i4uNS6O3fuVHJysmbNmqWQkBCX8xs2bNDBgwcVGhqqpUuXau7cufr+++8VHR2t3bt3S5Ly8/MVGBjo8tqAgAAVFBRUqO0ExgAAt6o6Y1yyIMef5efnq0+fPsrLy9OCBQs0depUZWRk6IYbbnCqN2TIEGVkZGjq1KlasGCBfv/9d/Xu3Vv79+//Cy0CAMC7eGoodVpammw2m9ORlpbmej9jNHz4cMXHx2vQoEGltmnq1KnKzs7WjBkzFB0drWHDhikrK0sHDx7Uc889J+nkAl0Wi+uELWOMrNaKffpg8S0AgFtVOT/4dAtyvPTSSzpw4IC+/vprhYaGSpLCwsIUHx+v7Oxsde/e3aMLcgAA4M08tSr1hAkTlJyc7FTm6+vrUu/FF1/Upk2btHnzZh0/fvz/23CyEcePH5fValWnTp1cXhcREaG2bdtq48aNkqSgoKBSh0wXFRXJZrNVqO0ExgAAt6pyaFHJghx+fn5atWqVozwrK0vR0dGOoFiS+vXrp4CAAC1btkzdu3cvc0EOAmMAAE7y1G5Nvr6+pQbCp3rzzTe1d+9eNW3a1OVcvXr19OCDD+qCCy5QmzZtXFakPnLkiGPodVRUlLKysmS3250yxLm5uWrXrl2F2s5QagBAteRuQY4tW7aodevWTmVWq1Xh4eGOb449uSAHAADwnNmzZ+vLL790Oq6++mo1bdpUX375pe68806lpqYqJSXF6XXr169Xbm6uYmJiJElxcXEqLCxUVlaWo05eXp5Wr17t9MV4eZAxBgC45alvUIuLi10W4DjdN8slC3Kkp6eXuiBHeRbb8OSCHAAAeDNPDaUur6ioKJey4OBg+fj4qHPnzpKk1NRUjRgxQklJSUpMTNSPP/6ohx9+WB06dFBSUpIkqUePHoqJiVFiYqKmT5+u4OBgTZ48WUFBQRo1alSF2kTGGADglqf2MfbkghzGmDIX2/DkghwAAHizqtquyZ3hw4frjTfe0ObNmzVw4EBNnDhR1157rT788EOn0WAZGRkaMGCAxo8fr6SkJDVv3lwffvihGjVqVKH7kTEGALjlqRDSkwty2Gy2UrO+RUVFjoW6PLkgBwAA3qySE8almj9/vkvZjTfeqBtvvNHt6xo1aqT09HSlp6f/pfsTGAMAKoWnFuRITU1VVFSUcnNznc7Z7Xbt2LFDCQkJkjy7IAcAAPBujCUDALhV2fsYl7Ugx2233aa4uDitXr1aeXl5jtdlZWWpsLDQsdiGJxfkAADAm1XHodSVjYwxAMCtyt7HuDwLcowePVozZ85UbGysUlNTtW/fPqWkpOiqq65S165dJXl2QQ4AALxZDY9pPYLAGADgVnUcWhQSEqKVK1dqzJgxSkxMVEBAgAYPHqwnn3zSqV5GRoaSk5M1fvx42e12devWTYsXL67wghwAAHgzAmPJYkz1SXpPLmX1UKA6mXxkb1U3AXDPL9jjl/y3h/42/736dDeoUrdXdQMAtyxTqroFgHsmdbbHr/lrkWf+Njdt6Pm2VRYyxgAAt/jKEgAA78Z31wTGAIAyVMeh1AAAwHOIi/m8AwAAAACo5cgYAwDc4htUAAC8G0OpCYwBAGVgjjEAAN6NuJjAGABQBjLGAAB4NzLGfN4BAAAAANRyZIwBAG7xDSoAAN6NhDGBMQCgDMwxBgDAuzGUmsAYAFAGi5XQGAAAb0ZczAg5AAAAAEAtR8YYAOCWxULGGAAAb8ZQagJjAEAZrAylBgDAqxEXExgDAMpAxhgAAO9Gxpg5xgAAAACAWo6MMQDALValBgDAu5EwJjAGAJSBodQAAHg3hlIzlBoAAAAAUMuRMQYAuMVQagAAvBsJYwJjAEAZGEoNAIB3Yyg1gTEAoAxkjAEA8G7ExcwxBgAAAADUcmSMAQBuMZQaAADvxlBqAmMAQBmsDKUGAMCrERcTGAMAykDGGAAA70bGmDnGAAAAAIAqlpCQoJYtWzqVbd26Vf3795fNZlNwcLBGjBih/Px8pzqFhYUaNWqUmjRpIn9/f8XGxionJ6fC9ydjDABwi1WpAQDwblWdMF64cKGWLFmiFi1aOMry8/PVp08fNWvWTAsWLNBvv/2mlJQU7dq1S8uXL3fUGzJkiNauXavp06crMDBQU6ZMUe/evZWTk6PGjRuXuw0ExgAAtxhKDQCAd6vKodS7d+/WPffco7CwMKfyl156SQcOHNDXX3+t0NBQSVJYWJji4+OVnZ2t7t27a82aNcrMzFRmZqbi4+MlSdHR0QoPD9esWbM0adKkcreDodQAAAAAgCpx6623Ki4uTn369HEqz8rKUnR0tCMolqR+/fopICBAy5Ytc9Tx9/dXXFyco05oaKh69uzpqFNeBMYAALcsVotHDgAAUD0ZDx0VNXfuXK1bt04vvPCCy7ktW7aodevWTmVWq1Xh4eHatm2bo05ERITq1nUeCB0ZGemoU14MpQYAuMVQagAAvJunhlIXFxeruLjYqczX11e+vr4udXfu3Knk5GSlp6crJCTE5Xx+fr4CAwNdygMCAlRQUFDuOuVFxhgA4BYZYwAAvJunMsZpaWmy2WxOR1pamuv9jNHw4cMVHx+vQYMGld4mY0r9ct4YI6v1ZBhrt9vLrFNeZIwBAAAAAH/ZhAkTlJyc7FRWWrb4xRdf1KZNm7R582YdP35c0slgVpKOHz8uq9Uqm81Wata3qKjIsVBXUFBQqUOmi4qKZLPZKtR2AmMAgFsMpQYAwLt5aij16YZNn+rNN9/U3r171bRpU5dz9erVU2pqqqKiopSbm+t0zm63a8eOHUpISJAkRUVFKSsrS3a73SlDnJubq3bt2lWo7QylBgC4ZbVaPHIAAIDqqbIX35o9e7a+/PJLp+Pqq69W06ZN9eWXX+q2225TXFycVq9erby8PMfrsrKyVFhY6FiFOi4uToWFhcrKynLUycvL0+rVq51Wqi4PMsYAALfIGAMA4N0qex/jqKgol7Lg4GD5+Pioc+fOkqTRo0dr5syZio2NVWpqqvbt26eUlBRdddVV6tq1qySpR48eiomJUWJioqZPn67g4GBNnjxZQUFBGjVqVIXaRMYYAAAAAFCthISEaOXKlQoJCVFiYqImTpyowYMH69///rdTvYyMDA0YMEDjx49XUlKSmjdvrg8//FCNGjWq0P3IGAMA3GJFaQAAvFtlZ4xLM3/+fJey9u3ba8WKFW5f16hRI6Wnpys9Pf0v3Z/AGADgFkOpAQDwbtUgLq5yBMYAALcsTLoBAMCrVYeMcVXj4w4AAAAAoFYjYwwAcIuh1AAAeDcSxgTGAIAysPgWAADejcCYodQAAAAAgFqOjHEN9ve33lLTiy/Ws+HhjrKWPXsqZsoUnduhg04UF2vXZ5/pg5QU7d++3VGnb1qauj/wgMv1VjzwgLKnTauUtsP7/brnN11z/U168ZmpuvzSix3lX3y5XjNfmqut27bLx6ee/tbxIo0fe6danB9W6nW+yflOf79ppB59+AElDOhfWc3Hn1gZSg1Umbvu+kY5OYX66KOukqSoqFWnrXvZZUFasKCTJOmXX45q+vTtWrs2X3a70SWX2PTAA5E6//z6ldBq1CY9W7TWqqRxpz2fuupdPbL6Pce/61qtyv5Hiv6b+42m/KkcVYvFtwiMa6wOiYlqm5Cg/B9/dJSFdemimz74QFvfeUcZiYmq16CBekyapOHZ2ZrVvr0O79snSWrSqZN+WLFCH02a5HTNgz/9VJmPAC/2y+5fNeKOsSosLHIq/3rjZg0fda9694zWk2mpOnL0qF7656samjRK7761UI0bBTnV/+OPP/TApEd1/PiJSmw9TsVQaqBqvP32Hn3wwV41b+7rKPv3v//mUm/58r2aN2+XbryxqSTpyJETGj58o44fN3rooQvk42PVc8/t0E03bdC773ZWYGC9SnsGeL/1v/6kLnOnupQ/1nuALm3WUm9sXuso86tbTwuvG67Lw8L139xvKrOZKANxMUOpa6SApk111fPP6+CuXU7l0RMmaO+WLfrP4MH6/r//Vc5bb2nhVVepQUiIOiUlOeo16dRJOz/5RD9/8YXTUfjrr5X8JPA2drtdby19Twk3/kMHDuS7nJ89b4EiwlvquScfU8/oK3RlbG/988WndCD/oJa8s8yl/rMv/lOFRYcqoeVwx2KxeOSoiBMnTmjq1KmKjIxU/fr11bFjRy1cuNCpTpcuXUq9z+eff+6oU1hYqFGjRun/2rv/8KjKO+/jnwnIICGZ/CAQEbUJKZEfgo+CoiEhAgmS7CpyFQSCJYX6GLF9xFwmPkCWgL8mJq6tD1ssiAYboZYtKbs1qaNYjFKjrGAQG0oMIstqMSmQX4JRnHn+SJl2mjCT4GEyOfN+Xdd9Xe4595y5z5TNme98v/d9x8bGKjQ0VGlpaaqtrTXkcwEups8/b9djj9UrNtbqcfzaa20ebehQq7Zt+0xZWcOVmTlMkrR3b7M++eSMHn00URkZQzVjxhD99KdjdPx4u15//URv3A5MrPWrL/Xup0c82rDB4ZoRP1pL//MX+uhkgyRpypUJeveH/1ep3xnVyyNGV1wuY1pfRmDcB922aZMOv/qqjrz+usfxT/fs0Ts//alcf/evsu34cbW3tChy5EhJUujQoRocG6vjNTX+HDKCxKG6eq157EnN/udZKn5sdafz48eN1uKsOxUS8rc/PUNjhmhw6CD997FPPfq+v/+AXvzlv2v1yvOXZ8G8Vq5cqdWrV+vuu+/Wyy+/rBkzZuiuu+7S1q1bJXX8CHPgwAHl5eWpurrao40bN859nQULFqi8vFxFRUUqKytTQ0ODpk2bppMnT/bWrQHdUlBwSElJkbrppgiv/YqK6jVwYD/l5sa7j331lVOSFBraz30sMrIjS9zU9LXxgwX+zsD+l2jdrPl6ue4DbT+4z338P+cv09Gmk7pu42O9ODrg/HpcSt3e3q633npLdXV1amlpUUhIiGw2m8aMGaMbbrhBVqvV90Vwwa5bulSXXX+91o8dq/Qnn/Q49+Zjnf/QfCc1VZdGRanhw45ylcv+V0cJ1tW3365Z/+//KWz4cDV8+KFeX7lS9a+8cvFvAKZ22WWxeu3lbYodNlTv/te+TueX/e8fdDr2zp69am5p1aiEv32p+/LLdv3fgkd1z9LFSvxuwkUdM3zzdyl1W1ub1q1bpwceeEAPPfSQJGn69Onau3ev1q1bp4ULF6qurk6nT59WZmamJk+e3OV1qqurVVFRoYqKCmVkZEiSkpOTFRcXp/Xr16vgH6aT4G941veuf//3z/THP7bq5Zcnqbj48Hn77dvXLIfjL7LbEzV48N++0iUlRWrUqFCVlHysxx9P1MCBIXr88XoNGtRPM2YM8cctIIg9MHmGhofZNO2FpzyOp2x+Uh82fNZLo4IvfTzZa4geBcZ2u11FRUVqbW3t8nxkZKRWrVql3NxcQwYHT7Yrr9TMp57Sjh/8wD1f2JtBQ4botmefVfOxY9r/wguSOsqopY7M8X/+8Ifqb7Xqhh//WAtffllbMjJ0+NVXL+YtwOQibOGSLbzb/U+ePKV/ebhIscOGavZts9zHn/zpzzRo0KW6Z+ldOv5548UYKnrA3/sYDxw4UNXV1YqNjfU4PmDAALW0tEiSav5a9TJhwoTzXsfhcCg0NFTp6enuYzExMZo6daoqKysJjM+DZ33v+vTTL2W3H5bdfrWiogZ47fvcc8d0+eUDddttwzyOW6399PDDo5STc0AzZrwrSRowwKKf//waXXEFi2/h4rkkpJ/+z4236KUP39PhU57Pb4LiwNbXy6CN0O1S6qKiIhUUFOjee+/Vnj17dPLkSX399df6+uuvdfLkSe3Zs0c//OEPlZ+fr5/+9KcXccjB6/bnn9dHlZU6WF7us2/YZZdp8e9/r9ChQ/WrOXP01Rcd8zQP/PKXenHWLP3yttv08c6dqquo0NZ/+iedOHRItzz88MW+BcDt84ZGLb77xzpx8pTWPfW4QgcNktSxavWvtv+n7I8UqH9/1gcMBJYQiyGtvb1dLS0tHq29vb3T+/Xv318TJkzQsGHD5HK5dPz4cdntdu3cuVP33XefpI7A2Gazafny5YqOjtbAgQOVkZGhQ4cOua9z8OBBxcfHd/p3lJCQoLq6uov7ofVRPOt7l8vl0sqVf9LUqVGaOTPGa98///lL/f73f9HixSPUv7/n17l33z2l73+/RldfPVgbNlyjZ5+9RlOmROlHP/pQ773XdBHvAMFu7tjrFTvYppK3SbT0NS6DWl/W7W+dP//5z7Vq1So93EXwFBERoYkTJ2rixIkaMGCA/u3f/k3Lly/3er329vZOX4jO9mRAQeaG++7TsPHj9cw11yik31/nDP01ixPSr59cTqd7bvHQceOUVVGhAYMH68Vbb9Vn773nvk7zf/93p9WnnWfP6vCrr+r6e+7xz80g6B366LDu+dGDOn36tDat/4nGjxsjSfri9GmtWP2Y7v7BIiXEf0dnz56V09mxIrXT6dLZs2cJlvswu92utWvXehwrLCzUmjVrzvuarVu3atGiRZKkjIwM3XnnnZI6AuPm5mbFxMRox44dOnr0qNauXavk5GTV1NRo+PDhampqUnh45wqGsLAwd+YZnvzxrLdav5HV2u88rwhuW7Z8qkOHvtBvfztRZ892zBM+l8U5e9apkBCLQv46teHVV/8ii0XKzBza6TobNvy3hg2z6tlnx2vAgI6gecqUKN155z49/ni9yssn+ueGEHS+N/o6fdjwqT74/H96eyhAj3U7Y9zY2KgpU6b47JeUlKRPP/3UZz+73S6bzebRdnd3MEFozPe+p9CYGD14/LhWnz2r1WfP6trFixXxne9o9dmzmrq6Y6GjuFtu0dI//EGyWFSakqJj1dUe1/luRoZG33FHp+v3v/RSnelGeTbwbVW/+54WLL5HLpdLLz6/Xtdde4373Id//JM+/ezP+tmG5zX2+hSNvT5Faf80T5K0as3jGnt9Sm8NO6gZtSr1ihUr1Nzc7NFWrFjh9b1vvPFGVVVVaePGjdq3b59uvvlmffnllyoqKtLu3btVUlKi5ORkLVq0SA6HQ83NzXr66acldSzQ1VUZuMvl8lgADn/jj2e93f6+EUM1JYejUadOfa0pU6o1duybGjv2Te3Y8bk+/bRdY8e+qZ/97BN33zfeOKGJEyM0ZEjncutPP/1S48aFuYNiSQoJsWjiRJvq60/741YQhPqHhCh95Bht++Pe3h4KLgCrUvcgQTtmzBht3brVY65WV55//nklJib6vN6KFSs6zU8qttm6O5yg89t77pE1LMzj2NTCQg2//nr98rbb1PrZZ4q99lot+O1v1XTkiMpmzlTrZ53ncoy7804l3n67juzapS+bmiRJlwwapFGZmfrkjTf8cCcIZrUHD+ne/5OvEZdfpuee+amGDfMsFRw7JlG/3vqcx7HGxhO69/58/ShniVJTkvw5XPyVxaAg0mq19njRpoSEBCUkJCglJUUjR47U9OnTtX37dmVlZXXqGx8fr9GjR2v//v2SOjKcXZVMt7W1ycbzpkv+eNZbrcxNPp+1axP1xRdnPY797GdH9eGHrXrmmXEaOrTj/39cLpcOHGjRokUjurxOfPwgffBBq776yukOjl0ul95/v0UjRgy8uDeBoHXN0MsVOsCqPxw7/4JxCFx9PKY1RLcD40ceeUS33367/vSnP2nOnDkaNWqUwsPDZbFY1NLSovr6epWXl+vdd9/V9u3bfV6vqy9IFEie34kuvtydOXFC33z1lT7b2/HL3MKXX1a/Sy7RG2vWyHbFFbJdcYW77xeNjTr18cf6Q3Gxxnzve8qqrNRuu10h/fsr6aGHNGDwYO0qLPTb/SA4rVpj19mzZ/Wje5fqz59/rj9//rn7XFRkhK68YoSuGTva4zX/82nH/tqXD7+s0zmYU0NDg373u99p1qxZGjr0b2WikyZNkiR9/PHH2rx5s66++upOK1KfOXNGQ4Z0rLqbmJgoh8Mhp9PpkSGur6/XmDFj/HAnfY8/nvUSZdTnEx8/qNOxiIj+GjDAomuu+du0gM8+a1dr6zdKSOjcX5KWLbtKCxe+rx/+8IO/zkG2aPv2P6umpkVPPz32oo0fwe2aYZdLkmobWWQLfVO3Y9Fbb71Vu3bt0tq1a7V69Wp99dVXHuf79eunlJQUvfbaa7rlllsMHyi8i4yL02XXXSdJmvfrX3c6X7N5s3b84Adq+OMfVZqSommPPabbS0vVb8AAHX3zTT2/dKlOffyxv4eNIHLsfz5V7Z86fuC5/8HOqwHfcVuGih5hleBA1BvbNWVnZ+uxxx7TypUr3cdf+euWchMnTlROTo6uuuoqvfnmm+7z+/btU319vfLy8iRJ6enpeuyxx+RwODRrVseq542NjaqqqtKqVav8eEd9B8/6vuHEiY7/XcLDu/4ad8014Xrxxf+lp58+ogcfrNUll4QoMXGwfvGLa3XDDRF+HCmCybDQjh9vTp2hXL8v6utl0EawuFw9/xi++uorffzxx2pqapLT6VRERIRGjhz5rfc1XOPnLUGAnlpz5i+9PQTAu4HRhl+y7gZj9pIetae+230XL16sX/3qV1q7dq0mTZqk9957T48++qhuvvlm/e53v1NpaamWLl2qxYsXKysrS5988olWr16t2NhY/dd//Zd7kbZbbrlF+/fvV3FxsaKjo7VmzRqdOHFCBw4cUGRkpCH3ZVYX61kvsdAjAptlre8+QG9yFW4w/JqvHzHmb/P0OOPH5i8XVL08YMAAXX311UaPBQAQgPydMZakjRs3atSoUXr++edVWFioyy67TPfff78KCgpksVi0ZMkSDRo0SCUlJZo9e7ZCQ0N1xx13yG63e6xcXl5ertzcXOXl5cnpdCopKUnbtm0jKO4GnvUAEDzIGF9gxvhiIWOMQEfGGAHvImSMP5r8XUOu8913PjLkOujryBgjsJExRqC7GBnjnR8b87d5RnyQZYwBAMHDqFWpAQBAYAqYTGkvIjAGAHjV1V7AAADAPAKnhrj3EBgDALzrhTnGAADAf4iLJerjAAAAAABBjYwxAMAr5hgDAGBulFITGAMAfGCOMQAA5kZcTGAMAPChN/YxBgAA/kPGmDnGAAAAAIAgR8YYAOAdGWMAAEyNhDEZYwCADxZLiCENAAAEJpfLmNYT33zzjYqKipSQkKBLL71UEyZM0IsvvujRZ/LkybJYLJ3aO++84+7T2tqqnJwcxcbGKjQ0VGlpaaqtre3xZ0DGGAAAAADgVytXrtRPfvITPfLII5o4caIqKyt11113KSQkRAsXLpTT6dSBAweUl5enOXPmeLx23Lhx7v9esGCB9uzZo+LiYoWHh2vt2rWaNm2aamtrFRUV1e3xEBgDALxi8S0AAMzN36XUbW1tWrdunR544AE99NBDkqTp06dr7969WrdunRYuXKi6ujqdPn1amZmZmjx5cpfXqa6uVkVFhSoqKpSRkSFJSk5OVlxcnNavX6+CgoJuj4naNgCAV5YQiyENAAAEJn+XUg8cOFDV1dXKzc31OD5gwAC1t7dLkmpqaiRJEyZMOO91HA6HQkNDlZ6e7j4WExOjqVOnqrKysvsDEoExAMAXS4gxDQAABCSXQa27+vfvrwkTJmjYsGFyuVw6fvy47Ha7du7cqfvuu09SR2Bss9m0fPlyRUdHa+DAgcrIyNChQ4fc1zl48KDi4+PVv79nIXRCQoLq6up69BlQSg0AAAAA+Nba29vdGd9zrFarrFbreV+zdetWLVq0SJKUkZGhO++8U1JHYNzc3KyYmBjt2LFDR48e1dq1a5WcnKyamhoNHz5cTU1NCg8P73TNsLAwtbS09Gjs/IQPAPCKUmoAAMzNqFJqu90um83m0ex2u9f3vvHGG1VVVaWNGzdq3759uvnmm/Xll1+qqKhIu3fvVklJiZKTk7Vo0SI5HA41Nzfr6aefliQ5nU5ZLJ2/Y7hcLoWE9CzUJWMMAPCKoBYAAHMzavGtFStWdJo37C1bLHWUPSckJCglJUUjR47U9OnTtX37dmVlZXXqGx8fr9GjR2v//v2SpIiIiC5Lptva2mSz2Xo0djLGAACvuto/8EIaAAAITEZljK1Wq8LDwz1aV4FxQ0ODXnjhBTU0NHgcnzRpkiTp448/1ubNmz32Kz7nzJkzGjJkiCQpMTFRR44ckdPp9OhTX1+vMWPG9OgzIDAGAAAAAPhNW1ubsrOztWnTJo/jr7zyiiRp4sSJKiwsVH5+vsf5ffv2qb6+XqmpqZKk9PR0tba2yuFwuPs0NjaqqqrKY6Xq7qCUGgDgXQ/n6AAAgL7F3/sYx8fH6/vf/74efvhh9evXT5MmTdJ7772nRx99VDNnztStt96qwsJCLV26VNnZ2crKytInn3yi1atXa/z48crOzpYkpaSkKDU1VVlZWSouLlZ0dLTWrFmjiIgI5eTk9GhMBMYAAK+YYwwAgLn1ZA9io2zcuFGjRo3S888/r8LCQl122WW6//77VVBQIIvFoiVLlmjQoEEqKSnR7NmzFRoaqjvuuEN2u91je6by8nLl5uYqLy9PTqdTSUlJ2rZtmyIjI3s0HovL1RsfQ9fWMAcNAW7Nmb/09hAA7wZGG37J499LMuQ6sb/+gyHXQV93T28PAPDKsra3RwB45yrcYPg1//2PxvxtnjvW+LH5C/VxAAAAAICgRik1AMArC3OMAQAwtYApIe5FBMYAAK+YYwwAgLkFzuTa3kMaAAAAAAAQ1MgYAwC8Y2FEAABMjYQxgTEAwAdKqQEAMDcCYwJjAIAPLL4FAIC5MceYOcYAAAAAgCBHxhgA4JWFOcYAAJgaCWMCYwCAL8wxBgDA1CilJjAGAPjAHGMAAMyNuJg5xgAAAACAIEfGGADgFXOMAQAwN0qpCYwBAD6wjzEAAOZGXEwpNQAAAAAgyJExBgB4Ryk1AACmRik1gTEAwAdKqQEAMDfiYgJjAIAvxMUAAJgaGWPmGAMAAAAAghwZYwCAd8wxBgDA1EgYExgDAHwgLgYAwNwopSYwBgD4wuJbAACYGnExc4wBAAAAAEGOwBgA4JXFYkzriW+++UZFRUVKSEjQpZdeqgkTJujFF1/06HPo0CFlZmbKZrMpOjpaS5cuVVNTk0ef1tZW5eTkKDY2VqGhoUpLS1Ntbe23/EQAADAXl8uY1pdRSg0A8K4XJhmvXLlSP/nJT/TII49o4sSJqqys1F133aWQkBAtXLhQTU1Nmj59uoYPH66ysjJ9/vnnys/P17Fjx/Tqq6+6r7NgwQLt2bNHxcXFCg8P19q1azVt2jTV1tYqKirK7/cFAEAg6uMxrSEIjAEA3vm5tqitrU3r1q3TAw88oIceekiSNH36dO3du1fr1q3TwoUL9cwzz+jUqVN6//33FRMTI0kaMWKEMjIytHv3bk2ZMkXV1dWqqKhQRUWFMjIyJEnJycmKi4vT+vXrVVBQ4N8bAwAgQPX1bK8RKKUGAASUgQMHqrq6Wrm5uR7HBwwYoPb2dkmSw+FQcnKyOyiWpJkzZyosLEyVlZXuPqGhoUpPT3f3iYmJ0dSpU919AAAAJAJjAIAPFovFkNZd/fv314QJEzRs2DC5XC4dP35cdrtdO3fu1H333SdJOnjwoEaNGuXxupCQEMXFxamurs7dJz4+Xv37exZHJSQkuPsAAICOUmojWl9GKTUAwDuD5hi3t7e7M77nWK1WWa3W875m69atWrRokSQpIyNDd955pySpqalJ4eHhnfqHhYWppaWl230AAACl1BIZYwCAn9jtdtlsNo9mt9u9vubGG29UVVWVNm7cqH379unmm2/Wl19+KZfL1WUW2uVyKSSk49HmdDp99gEAAJAIjAEAPhi1XdOKFSvU3Nzs0VasWOH1vRMSEpSSkqK7775bW7Zs0YEDB7R9+3bZbLYus75tbW2y2WySpIiICJ99AABA75RSB9rWjATGAADvQiyGNKvVqvDwcI/WVRl1Q0ODXnjhBTU0NHgcnzRpkiTp2LFjSkxMVH19vcd5p9OpI0eOaMyYMZKkxMREHTlyRE6n06NffX29uw8AAOidfYxXrlyp1atX6+6779bLL7+sGTNm6K677tLWrVslyb01Y2Njo8rKylRUVKTy8nLNmzfP4zoLFixQeXm5ioqKVFZWpoaGBk2bNk0nT57s0XgIjAEA3lkMat3U1tam7Oxsbdq0yeP4K6+8IkmaMGGC0tPTVVVVpcbGRvd5h8Oh1tZW9yrU6enpam1tlcPhcPdpbGxUVVWVx0rVAAAEO39njP9xa8bp06frX//1XzV16lStW7dOktxbM1ZUVOi2227T3Xffra1bt+q1117T7t27Jcm9NePmzZuVnZ2tOXPmaOfOnWpra9P69et79Bmw+BYAIKDEx8fr+9//vh5++GH169dPkyZN0nvvvadHH31UM2fO1K233qpJkyZp3bp1SktLU2FhoU6cOKH8/HzNmjVLN910kyQpJSVFqampysrKUnFxsaKjo7VmzRpFREQoJyenl+8SAIDgdW5rxtjYWI/jAwYMcE+D8rU145QpU3xuzVhQUNDtMREYAwC86slWS0bZuHGjRo0apeeff16FhYW67LLLdP/996ugoEAWi0VDhgzRrl27tHz5cmVlZSksLExz587Vk08+6XGd8vJy5ebmKi8vT06nU0lJSdq2bZsiIyP9fk8AAAQqf69KfW5rxo73dunzzz9XaWmpdu7cqWeffVZSx7aL53ajOKcnWzNu2bKlZ2O60JsBAAQJ/8fFslqtWrVqlVatWnXePuPGjdPOnTu9XicyMlKlpaUqLS01eogAAJiGUXFxX96akTnGAACvLCEWQxoAAAhMRi2+1Ze3ZiRjDAAAAAD41lasWKHc3FyPY96yxVJH2fO57RlHjhyp6dOn+9yaccSIEZI6tmY8V1b9j316ujUjGWMAgHd+XpUaAAD4l1GrUvflrRkJjAEA3lksxjQAABCQ/L2PcSBuzUgpNQAAAADAbwJxa0YCYwCAVyR7AQAwNz/v1iQp8LZmJDAGAHjHitIAAJiav/cxlgJva0YCYwCAV2SMAQAwt94IjAMNi28BAAAAAIIaGWMAgHekjAEAMDUSxgTGAAAfiIsBADA3SqkJjAEAvrD4FgAApkZczBxjAAAAAECQI2MMAPCOWmoAAEyNjDGBMQDAB+JiAADMjTnGBMYAAF+IjAEAMDXiYuYYAwAAAACCHBljAIBXFn5CBQDA1CilJjAGAPhCKTUAAKZGXEwpNQAAAAAgyJExBgB4R8IYAABTo5SawBgA4IOFUmoAAEyNuDjAAuPCEx/29hAAr5xHX+3tIQBehSQuuAgXJTCGcWa92NsjALybNLy3RwD4Hxlj5hgDAAAAAIJcQGWMAQABiFJqAABMjYQxgTEAwBdKqQEAMDVKqQmMAQC+WJh1AwCAmREXM8cYAAAAABDkyBgDALxjjjEAAKZGKTWBMQDAF+YYAwBgasTFBMYAAF+YYwwAgKmRMWaOMQAAAAAgyJExBgB4Ryk1AACmRsKYwBgA4AuLbwEAYGqUUlNKDQAAAAAIcmSMAQDehfAbKgAAZkbCmMAYAOALpdQAAJgapdSUUgMAfAkJMaYBAICA5DKo9eg9XS5t3LhR48eP1+DBgxUfH6/ly5erpaXF3Wfy5MmyWCyd2jvvvOPu09raqpycHMXGxio0NFRpaWmqra3t8WdAxhgAAAAA4FclJSVauXKl8vLyNH36dNXX1+tf/uVf9OGHH+q1116Ty+XSgQMHlJeXpzlz5ni8dty4ce7/XrBggfbs2aPi4mKFh4dr7dq1mjZtmmpraxUVFdXt8RAYAwC8o5QaAABT83cptdPplN1u1z333CO73S5JmjFjhqKjozVv3jzt3btXgwcP1unTp5WZmanJkyd3eZ3q6mpVVFSooqJCGRkZkqTk5GTFxcVp/fr1Kigo6PaYqG0DAHhnsRjTAABAQPJ3KXVLS4sWLVqkhQsXehwfNWqUJOnw4cOqqamRJE2YMOG813E4HAoNDVV6err7WExMjKZOnarKysoejIjAGADgC3OMAQAwNZfLmNZdERERWrdunZKSkjyOl5eXS+oola6pqZHNZtPy5csVHR2tgQMHKiMjQ4cOHXL3P3jwoOLj49W/v2chdEJCgurq6nr0GfBNBQAAAADwrbW3t6ulpcWjtbe3d+u1b7/9tp544gnNnj1bY8eOVU1NjZqbmxUTE6MdO3Zo06ZN+uijj5ScnKzPPvtMktTU1KTw8PBO1woLC/NYxKs7CIwBAN5RSg0AgKkZVUptt9tls9k82rk5xN689dZbysjI0MiRI/Xcc89JkoqKirR7926VlJQoOTlZixYtksPhUHNzs55++mlJHXOVLV18x3C5XArpYbUai28BALyyhBDUAgBgZkYtvrVixQrl5uZ6HLNarV5f89JLLyk7O1uJiYlyOBzulaSvvfbaTn3j4+M1evRo7d+/X1JHSXZXJdNtbW2y2Ww9GjsZYwBAwAm0vQ0BAIBvVqtV4eHhHs1bYFxSUqKFCxdq8uTJevPNNxUbGytJ+vrrr7V582aPZ/o5Z86c0ZAhQyRJiYmJOnLkiJxOp0ef+vp6jRkzpkdjJzAGAHhnCTGm9UBJSYmWLVumzMxM7dixQ/n5+dqyZYvmzJkjl8slp9Pp3tuwurrao/3j3obl5eUqKipSWVmZGhoaNG3aNJ08edLoTwkAgD7L36tSS9KGDRuUn5+vuXPn6tVXX/XI8F5yySUqLCxUfn6+x2v27dun+vp6paamSpLS09PV2toqh8Ph7tPY2KiqqiqPlaq7g1JqAIB3fi6lDsS9DQEAMDN/72N8/PhxPfDAA7rqqqv04x//WPv27fM4P3LkSBUWFmrp0qXKzs5WVlaWPvnkE61evVrjx49Xdna2JCklJUWpqanKyspScXGxoqOjtWbNGkVERCgnJ6dHYyIwBgB45+eFs87tbTh//nyP43+/t6Hrr0/wb7O3IYExAAAd/BwXq7KyUmfOnNHRo0eVnJzc6XxpaamWLFmiQYMGqaSkRLNnz1ZoaKjuuOMO2e12j+2ZysvLlZubq7y8PDmdTiUlJWnbtm2KjIzs0ZgopQYABJRA3NsQAAAYZ8mSJXK5XOdt5zLC8+fP1969e/XFF1+ooaFBGzZscC/OdU5kZKRKS0t16tQpNTc3q7KyUomJiT0eExljAIB3Pdzu4Hza29s77WVotVp9rlYp+d7b8OjRo1q7dq2Sk5NVU1Oj4cOHG7q3IQAAZubvUupARMYYAOCdQfsY9+W9DQEAMDOXy5jWl5ExBgB4Z9Ac4768tyEAAGbWx2NaQ/CTOQDAL/ry3oYAAMDcCIwBAN6FhBjTeiDQ9jYEAMDMKKWmlBoA4Iuft2sKxL0NAQAwsz4e0xqCwBgA4F2IfwPjQNzbEAAAMyMwJjAGAASYJUuWaMmSJT77zZ8/X/Pnz/fa59zehqWlpUYNDwAAmBCBMQDAOwvLUQAAYGZ9fX6wEQiMAQDe+bmUGgAA+BdxMatSAwAAAACCHBljAIB3fl6VGgAA+Bel1ATGAABfergHMQAA6FuIiwmMAQC+kDEGAMDUyBgzxxgAAAAAEOTIGAMAvCNjDACAqZEwJjAGAPjCPsYAAJgapdQExgAAX0gYAwBgasTFzDEGAAAAAAQ5MsYAAO+YYwwAgKlRSk1gDADwhcAYAABTIy4mMAYA+EJgDACAqZExZo4xAAAAACDIkTEGAPhAxhgAADMjYUxgDADwhbgYAABTo5SaUmoAAAAAQJAjYwwA8I7FtwAAMDUSxgTGAABfCIwBADA1SqkJjAEAvhAYAwBgasTFzDEGAAAAAAQ5AmMAgA8WgxoAAAhELpcxrWfv6dLGjRs1fvx4DR48WPHx8Vq+fLlaWlrcfQ4dOqTMzEzZbDZFR0dr6dKlampq8rhOa2urcnJyFBsbq9DQUKWlpam2trbHnwGBMQDAO+JiAABMzWVQ64mSkhItW7ZMmZmZ2rFjh/Lz87VlyxbNmTNHLpdLTU1Nmj59uhobG1VWVqaioiKVl5dr3rx5HtdZsGCBysvLVVRUpLKyMjU0NGjatGk6efJkj8bDHGMAgHfMMQYAwNT8vfiW0+mU3W7XPffcI7vdLkmaMWOGoqOjNW/ePO3du1evvfaaTp06pffff18xMTGSpBEjRigjI0O7d+/WlClTVF1drYqKClVUVCgjI0OSlJycrLi4OK1fv14FBQXdHhMZYwAAAACA37S0tGjRokVauHChx/FRo0ZJkg4fPiyHw6Hk5GR3UCxJM2fOVFhYmCorKyVJDodDoaGhSk9Pd/eJiYnR1KlT3X26i8AYAOCdxWJMAwAAAcnfpdQRERFat26dkpKSPI6Xl5dLksaNG6eDBw+6A+VzQkJCFBcXp7q6OknSwYMHFR8fr/79PQuhExIS3H26i1JqAIAPBLUAAJiZUaXU7e3tam9v9zhmtVpltVp9vvbtt9/WE088odmzZ2vs2LFqampSeHh4p35hYWHuBbq606e7yBgDAAAAAL41u90um83m0c7NIfbmrbfeUkZGhkaOHKnnnntOUseq1ZYuKs5cLpdCQjrCWKfT6bNPd5ExBgB4Rxk0AACmZtTaWytWrFBubq7HMV/Z4pdeeknZ2dlKTEyUw+FQVFSUJMlms3WZ9W1ra9OIESMkdZRkd1Uy3dbWJpvN1qOxkzEGAHjHHGMAAEzNqH2MrVarwsPDPZq3wLikpEQLFy7U5MmT9eabbyo2NtZ9LjExUfX19R79nU6njhw5ojFjxrj7HDlyRE6n06NffX29u093ERgDALxjH2MAAEytN/Yx3rBhg/Lz8zV37ly9+uqrnTK86enpqqqqUmNjo/uYw+FQa2urexXq9PR0tba2yuFwuPs0NjaqqqrKY6Xq7qCUGgAAAADgN8ePH9cDDzygq666Sj/+8Y+1b98+j/MjR47UsmXLtG7dOqWlpamwsFAnTpxQfn6+Zs2apZtuukmSlJKSotTUVGVlZam4uFjR0dFas2aNIiIilJOT06MxERgDALyjDBoAAFMzalXq7qqsrNSZM2d09OhRJScndzpfWlqq7Oxs7dq1S8uXL1dWVpbCwsI0d+5cPfnkkx59y8vLlZubq7y8PDmdTiUlJWnbtm2KjIzs0ZgIjAEAPhAYAwBgZn6Oi7VkyRItWbLEZ79x48Zp586dXvtERkaqtLRUpaWl32pMBMYAAO/IGAMAYGr+zhgHIhbfAgAAAAAENTLGAADvyBgDAGBqZIwJjAEAvhAXAwBgasTFBMYAAF/IGAMAYGpkjJljDAAAAAAIcmSMAQA+kDEGAMDMSBgTGAMAfKGUGgAAUyMwppQaAAAAABDkyBgDALwjYwwAgKmx+BYZYwCALxaLMa0HXC6XNm7cqPHjx2vw4MGKj4/X8uXL1dLS4u5z6NAhZWZmymazKTo6WkuXLlVTU5PHdVpbW5WTk6PY2FiFhoYqLS1NtbW1RnwqAACYhsug1pcRGAMAAk5JSYmWLVumzMxM7dixQ/n5+dqyZYvmzJkjl8ulpqYmTZ8+XY2NjSorK1NRUZHKy8s1b948j+ssWLBA5eXlKioqUllZmRoaGjRt2jSdPHmyl+4MAIDA43IZ0/oySqkBAAHF6XTKbrfrnnvukd1ulyTNmDFD0dHRmjdvnvbu3avXXntNp06d0vvvv6+YmBhJ0ogRI5SRkaHdu3drypQpqq6uVkVFhSoqKpSRkSFJSk5OVlxcnNavX6+CgoJeu0cAABBYyBgDALzzcyl1S0uLFi1apIULF3ocHzVqlCTp8OHDcjgcSk5OdgfFkjRz5kyFhYWpsrJSkuRwOBQaGqr09HR3n5iYGE2dOtXdBwAAUEotERgDAHzxc2AcERGhdevWKSkpyeN4eXm5JGncuHE6ePCgO1A+JyQkRHFxcaqrq5MkHTx4UPHx8erf37M4KiEhwd0HAABQSi1RSg0A8MWgVanb29vV3t7uccxqtcpqtfp87dtvv60nnnhCs2fP1tixY9XU1KTw8PBO/cLCwtwLdHWnDwAA6PvZXiP0KDB+8803e3TxlJSU857r6gvSgPavZLUO6NF7AAD6BrvdrrVr13ocKyws1Jo1a7y+7q233tI///M/a+TIkXruueckdaxabekiYHe5XAoJ6SiGcjqdPvugs4v9rHd+/Y1CLul3QWMDAOBi6VFgfPvtt7t/ZT/fl5K/P/fNN9+c91pdfUFanX+v1jx0X0+GBAC46IzJGK9YsUK5ubkex3xli1966SVlZ2crMTFRDodDUVFRkiSbzdZl1retrU0jRoyQ1FGS3VXJdFtbm2w224Xehuld7Gf9yDuu03fnTDRuwACAb62vl0EboUeB8QcffKC0tDSdOHFCv/jFLzRo0KALfuOuviAN+OLwBV8PAHCRGFRK3d2y6XNKSkr00EMPKSUlRf/xH//hEcwmJiaqvr7eo7/T6dSRI0c0Z84cdx+HwyGn0+mRIa6vr9eYMWO+5d2Y18V+1s/dnnue3gCA3kJc3MPA+IorrpDD4dD111+v3//+9yopKbngN+7qC5LrLGXUAABpw4YNys/P17x581RWVqYBAzyfD+np6SouLlZjY6N7ZWqHw6HW1lb3KtTp6el67LHH5HA4NGvWLElSY2OjqqqqtGrVKv/eUB9ysZ/1lFEDAAKRxeXqeeK8tLRUy5Yt0+HDhzV8+HDDBuM6+UfDrgVcDK7GD3p7CIBXIYkLDL+m82CZIdcJGX1Xt/odP35c8fHxGjp0qF588cVOq0qPHDlSFotFo0eP1uWXX67CwkKdOHFC+fn5mjx5ssdWTLfccov279+v4uJiRUdHa82aNTpx4oQOHDigyMhIQ+7LrC7Ws37Wi/cYdi3gYjhxprdHAHi35+4Nhl8zpdSYv81v/sD4sfnLBa1KnZ2dreuuu+5blVcBAPoKY0qpu6uyslJnzpzR0aNHlZyc3Ol8aWmpsrOztWvXLi1fvlxZWVkKCwvT3Llz9eSTT3r0LS8vV25urvLy8uR0OpWUlKRt27YRFHcDz3oACB6UUl9gxvhiIWOMQEfGGIHuomSM/7TFkOuEXJ1lyHXQt5ExRqAjY4xAdzEyxlOeN+Zv8+4lfTdjzH4VAAAAAICgdkGl1ACAIGLhN1QAAMwsYEqIexGBMQDAB//OMQYAAP4VOJNrew+BMQDAO4P2MQYAAIGJuJg5xgAAAACAIEfGGADgA7+hAgBgZpRSExgDAHyhlBoAAFMjLiYNAADwxWIxpgEAgIDkchnTLtSxY8cUERGhN954w+P45MmTZbFYOrV33nnH3ae1tVU5OTmKjY1VaGio0tLSVFtb2+MxkDEGAAAAAPSKo0ePaubMmWpubvY47nQ6deDAAeXl5WnOnDke58aNG+f+7wULFmjPnj0qLi5WeHi41q5dq2nTpqm2tlZRUVHdHgeBMQDAB7K9AACYWW+UUjudTr3wwgt68MEHuzxfV1en06dPKzMzU5MnT+6yT3V1tSoqKlRRUaGMjAxJUnJysuLi4rR+/XoVFBR0ezyUUgMAvLOEGNMAAEBA6o1S6g8++ED33nuvFi9erLKysk7na2pqJEkTJkw47zUcDodCQ0OVnp7uPhYTE6OpU6eqsrKyR+PhmwoAAAAAwK+uvPJK1dfX66mnntKgQYM6na+pqZHNZtPy5csVHR2tgQMHKiMjQ4cOHXL3OXjwoOLj49W/v2chdEJCgurq6no0HgJjAIB3LL4FAICpuQxq7e3tamlp8Wjt7e1dvmdUVJRGjBhx3jHV1NSoublZMTEx2rFjhzZt2qSPPvpIycnJ+uyzzyRJTU1NCg8P7/TasLAwtbS09OgzIDAGAPhgMagBAIBAZFQptd1ul81m82h2u/2CxlRUVKTdu3erpKREycnJWrRokRwOh5qbm/X0009L6pinbOnix3eXy6WQkJ6Fuiy+BQDwjvnBAACYmlGLb61YsUK5ubkex6xW6wVd69prr+10LD4+XqNHj9b+/fslSREREV2WTLe1tclms/Xo/fi2AwAAAAD41qxWq8LDwz3ahQTGX3/9tTZv3uyxX/E5Z86c0ZAhQyRJiYmJOnLkiJxOp0ef+vp6jRkzpkfvSWAMAPDKYrEY0gAAQGDqjVWpvbnkkktUWFio/Px8j+P79u1TfX29UlNTJUnp6elqbW2Vw+Fw92lsbFRVVZXHStXdQWAMAPCBOcYAAJiZUYtvGamwsFBvvfWWsrOz9dprr+nZZ59VZmamxo8fr+zsbElSSkqKUlNTlZWVpU2bNuk3v/mNZsyYoYiICOXk5PTo/ZhjDADwjjnGAACYmpHZXqMsWbJEgwYNUklJiWbPnq3Q0FDdcccdstvtHtszlZeXKzc3V3l5eXI6nUpKStK2bdsUGRnZo/cjMAYAAAAA9JrU1FS5uojO58+fr/nz53t9bWRkpEpLS1VaWvqtxkBgDADwgTJoAADMLAATxn5HYAwA8I6FswAAMLVALKX2NwJjAIB3zDEGAMDUiItZlRoAAAAAEOTIGAMAfKCUGgAAM6OUmsAYAOALc4wBADA1AmNKqQEAAAAAQY6MMQDAOxbfAgDA1EgYExgDAHyilBoAADMjMCYwBgD4whxjAABMjTnGzDEGAAAAAAQ5MsYAAB/4DRUAADMjYUxgDADwhVJqAABMjVJqAmMAgC8ExgAAmBpxMfVxAAAAAIAgR8YYAOADv6ECAGBmlFITGAMAfKGUGgAAUyMuJg0AAAAAAAhyZIwBAD6QMQYAwMwopSYwBgD4Qik1AACmRlxMYAwA8InAGAAAMyNjzBxjAAAAAECQI2MMAPCOUmoAAEyNhDGBMQDAJ4qLAAAwM0qpCYwBAL6QMQYAwNSIi0kDAAAC3LFjxxQREaE33njD4/jkyZNlsVg6tXfeecfdp7W1VTk5OYqNjVVoaKjS0tJUW1vr5zsAAACBjowxAMCH3ssYHz16VDNnzlRzc7PHcafTqQMHDigvL09z5szxODdu3Dj3fy9YsEB79uxRcXGxwsPDtXbtWk2bNk21tbWKioryyz0AABDoKKUmMAYA+NILpdROp1MvvPCCHnzwwS7P19XV6fTp08rMzNTkyZO77FNdXa2KigpVVFQoIyNDkpScnKy4uDitX79eBQUFF238AAD0JcTFlFIDAHyyGNS674MPPtC9996rxYsXq6ysrNP5mpoaSdKECRPOew2Hw6HQ0FClp6e7j8XExGjq1KmqrKzs0XgAADAzl8uY1pcRGAMAAs6VV16p+vp6PfXUUxo0aFCn8zU1NbLZbFq+fLmio6M1cOBAZWRk6NChQ+4+Bw8eVHx8vPr39yyOSkhIUF1d3UW/BwAA0HcQGAMAvLNYDGnt7e1qaWnxaO3t7V2+ZVRUlEaMGHHeIdXU1Ki5uVkxMTHasWOHNm3apI8++kjJycn67LPPJElNTU0KDw/v9NqwsDC1tLQY89kAAGACLoPahTrfQpuHDh1SZmambDaboqOjtXTpUjU1NXn0MWqhTQJjAIAPxpRS2+122Ww2j2a32y9oREVFRdq9e7dKSkqUnJysRYsWyeFwqLm5WU8//bSkjnnKli7mR7tcLoWE8PgDAOCc3iylPnr0qNLS0jottNnU1KTp06ersbFRZWVlKioqUnl5uebNm+fRb8GCBSovL1dRUZHKysrU0NCgadOm6eTJkz0aB4tvAQD8YsWKFcrNzfU4ZrVaL+ha1157badj8fHxGj16tPbv3y9JioiI6LJkuq2tTTab7YLeFwAAGMPXQpvPPPOMTp06pffff18xMTGSpBEjRigjI0O7d+/WlClTDF1ok5/MAQDeGVRKbbVaFR4e7tEuJDD++uuvtXnzZo/9is85c+aMhgwZIklKTEzUkSNH5HQ6PfrU19drzJgxF/ZZAABgQr1RSu1roU2Hw6Hk5GR3UCxJM2fOVFhYmHsRTSMX2iQwBgD4EGJQM8Yll1yiwsJC5efnexzft2+f6uvrlZqaKklKT09Xa2urHA6Hu09jY6Oqqqo8HqAAAAS73iil9rXQ5sGDBzVq1CiPYyEhIYqLi3NXhBm50Cal1AAA73phH2NfCgsLtXTpUmVnZysrK0uffPKJVq9erfHjxys7O1uSlJKSotTUVGVlZam4uFjR0dFas2aNIiIilJOT07s3AABAADFqp6X29vZOC2tardYuK8SioqIUFRV13mt1ZxFNIxfaJGMMAOhzlixZol/+8pc6cOCAZs+erVWrVum2227T66+/7vGrcXl5uW6//Xbl5eUpOztbl19+uV5//XVFRkb24ugBADAnIxfadLlcPhfRNHKhTTLGAAAfejdjnJqaKlcX9Vnz58/X/Pnzvb42MjJSpaWlKi0tvVjDAwCgz7vQFaX/kZELbdpsti6zvm1tbe4tHY1caJOMMQDAB2O2awIAAIHJqMW3jFpoU+pYRLO+vt7jmNPp1JEjR9yLaBq50CaBMQDAK4vFYkgDAACBqTf3MT6f9PR0VVVVqbGx0X3M4XCotbXVvYimkQttEhgDAAAAAALKsmXLdOmllyotLU2/+c1vtGnTJmVlZWnWrFm66aabJHkutLlp0yb95je/0YwZMy5ooU3mGAMAfCDbCwCAmRmc7DXEkCFDtGvXLi1fvlxZWVkKCwvT3Llz9eSTT3r0Ky8vV25urvLy8uR0OpWUlKRt27b1eKFNAmMAgHeUQQMAYGpGl0H31PkW2hw3bpx27tzp9bVGLbRJKTUAAAAAIKiRMQYA+EDGGAAAMwvEUmp/IzAGAHhnobgIAAAz6+1S6kBAYAwA8IGMMQAAZkZczBxjAAAAAECQI2MMAPCOVakBADA1SqkJjAEAPhEYAwBgZsTFBMYAAF/IGAMAYGpkjJljDAAAAAAIcmSMAQA+kDEGAMDMSBgTGAMAfKGUGgAAU6OUmsAYAOATgTEAAGZGXMwcYwAAAABAkCNjDADwzsJvqAAAmBml1ATGAACfKKUGAMDMiIsppQYAAAAABDkyxgAA71iVGgAAU6OUmsAYAOATgTEAAGZGXCxZXC5+HzCj9vZ22e12rVixQlartbeHA3TCv1EA+Pb4W4pAx79R9BUExibV0tIim82m5uZmhYeH9/ZwgE74NwoA3x5/SxHo+DeKvoLFtwAAAAAAQY3AGAAAAAAQ1AiMAQAAAABBjcDYpKxWqwoLC1nkAAGLf6MA8O3xtxSBjn+j6CtYfAsAAAAAENTIGAMAAAAAghqBMQAAAAAgqBEYAwAAAACCGoGxCb3yyiuaOHGiBg0apKuuukp2u11MJUegOnbsmCIiIvTGG2/09lAAoE/heY++gmc9+gICY5N5++23ddttt2n06NEqLy/XXXfdpVWrVunxxx/v7aEBnRw9elRpaWlqbm7u7aEAQJ/C8x59Bc969BWsSm0yM2fO1KlTp7Rnzx73sYceekjr169XQ0ODLr300l4cHdDB6XTqhRde0IMPPihJOnnypHbt2qXU1NTeHRgA9BE87xHoeNajryFjbCLt7e164403NGfOHI/j3/ve99TW1qa33nqrl0YGePrggw907733avHixSorK+vt4QBAn8LzHn0Bz3r0NQTGJvLxxx/rq6++0qhRozyOJyQkSJLq6up6Y1hAJ1deeaXq6+v11FNPadCgQb09HADoU3jeoy/gWY++pn9vDwDGaWpqkiSFh4d7HA8LC5MktbS0+HtIQJeioqIUFRXV28MAgD6J5z36Ap716GvIGJuI0+mUJFksli7Ph4TwPzcAAH0dz3sAMB5/OU0kIiJCUudfiltbWyVJNpvN30MCAAAG43kPAMYjMDaRkSNHql+/fqqvr/c4fu7/HjNmTG8MCwAAGIjnPQAYj8DYRAYOHKiUlBSVl5fr73fh+vWvf62IiAjdcMMNvTg6AABgBJ73AGA8Ft8ymYKCAs2YMUPz5s3TkiVL9Pbbb6ukpERPPPEEexoCAGASPO8BwFhkjE1m2rRp2r59uw4dOqTZs2dry5YtKikpUV5eXm8PDQAAGITnPQAYy+L6+xocAAAAAACCDBljAAAAAEBQIzAGAAAAAAQ1AmMAAAAAQFAjMAYAAAAABDUCYwAAAABAUCMwBgAAAAAENQJjAAAAAEBQIzAGAAAAAAQ1AmMAAAAAQFAjMAYAAAAABDUCYwAAAABAUCMwBgAAAAAEtf8PCQgFy5ZPMf8AAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 1200x1000 with 8 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "f,ax=plt.subplots(2,2,figsize=(12,10))\n",
    "y_pred = cross_val_predict(svm.SVC(kernel='rbf'),data,target,cv=10)\n",
    "sns.heatmap(confusion_matrix(target,y_pred),ax=ax[0,0],annot=True,fmt='2.0f',cmap='GnBu')\n",
    "ax[0,0].set_title('confusion SVC')\n",
    "y_pred = cross_val_predict(KNeighborsClassifier(n_neighbors=9),data,target,cv=10)\n",
    "sns.heatmap(confusion_matrix(target,y_pred),ax=ax[0,1],annot=True,fmt='2.0f',cmap='coolwarm')\n",
    "ax[0,1].set_title('confusion KNN')\n",
    "y_pred = cross_val_predict(RandomForestClassifier(n_estimators=100),data,target,cv=10)\n",
    "sns.heatmap(confusion_matrix(target,y_pred),ax=ax[1,0],annot=True,fmt='2.0f',cmap='OrRd')\n",
    "ax[1,0].set_title('confusion randomforest')\n",
    "y_pred = cross_val_predict(LogisticRegression(),data,target,cv=10)\n",
    "sns.heatmap(confusion_matrix(target,y_pred),ax=ax[1,1],annot=True,fmt='2.0f',cmap='summer')\n",
    "ax[1,1].set_title('confusion logisticregression')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "0a692dde-9611-4da9-8ccd-ae109834fdee",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pylab as plt\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "b4a7de3d-c3c8-4b87-a9be-052bed7b9368",
   "metadata": {},
   "outputs": [],
   "source": [
    "data= pd.read_csv(\"train.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "b9014516-dd4d-48ab-aec0-1e23a57d2073",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Embarked</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>C</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>female</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>female</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>male</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>S</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass     Sex   Age  SibSp  Parch     Fare Embarked\n",
       "0         0       3    male  22.0      1      0   7.2500        S\n",
       "1         1       1  female  38.0      1      0  71.2833        C\n",
       "2         1       3  female  26.0      0      0   7.9250        S\n",
       "3         1       1  female  35.0      1      0  53.1000        S\n",
       "4         0       3    male  35.0      0      0   8.0500        S"
      ]
     },
     "execution_count": 92,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "data= data.drop(labels=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)\n",
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "67f47722-506e-433f-b0c7-fcb4fe7f8050",
   "metadata": {},
   "outputs": [],
   "source": [
    "data= data.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "f3b97011-0b89-4193-8467-27ebd0b4a0c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_dummy = pd.get_dummies(data[['Sex', 'Embarked']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "ace86f07-8396-4748-9f2f-dde1d8264589",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_conti = pd.DataFrame(data,columns=['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare'], index=data.index)\n",
    "data = data_conti.join(data_dummy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 96,
   "id": "407ff862-f75f-47d7-b7a9-6c810ad1166e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Survived</th>\n",
       "      <th>Pclass</th>\n",
       "      <th>Age</th>\n",
       "      <th>SibSp</th>\n",
       "      <th>Parch</th>\n",
       "      <th>Fare</th>\n",
       "      <th>Sex_female</th>\n",
       "      <th>Sex_male</th>\n",
       "      <th>Embarked_C</th>\n",
       "      <th>Embarked_Q</th>\n",
       "      <th>Embarked_S</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>22.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>7.2500</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>38.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>71.2833</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>26.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>7.9250</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>35.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>53.1000</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>35.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>8.0500</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "      <td>False</td>\n",
       "      <td>False</td>\n",
       "      <td>True</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Survived  Pclass   Age  SibSp  Parch     Fare  Sex_female  Sex_male  \\\n",
       "0         0       3  22.0      1      0   7.2500       False      True   \n",
       "1         1       1  38.0      1      0  71.2833        True     False   \n",
       "2         1       3  26.0      0      0   7.9250        True     False   \n",
       "3         1       1  35.0      1      0  53.1000        True     False   \n",
       "4         0       3  35.0      0      0   8.0500       False      True   \n",
       "\n",
       "   Embarked_C  Embarked_Q  Embarked_S  \n",
       "0       False       False        True  \n",
       "1        True       False       False  \n",
       "2       False       False        True  \n",
       "3       False       False        True  \n",
       "4       False       False        True  "
      ]
     },
     "execution_count": 96,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 97,
   "id": "294462a6-531a-41e3-8819-24f103044014",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import  train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 98,
   "id": "c03cfcad-aa91-47cc-abdd-c7f12584c4fe",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = data.iloc[:, 1:]\n",
    "y = data.iloc[:, 0]\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 99,
   "id": "8c729641-75d8-43ee-8778-a69aba99689f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import cross_val_score \n",
    "from sklearn.model_selection import cross_val_predict"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 100,
   "id": "46c772ef-7474-45ab-a038-c8f1cc26aeb8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.78      0.84      0.81       125\n",
      "           1       0.75      0.67      0.71        89\n",
      "\n",
      "    accuracy                           0.77       214\n",
      "   macro avg       0.77      0.76      0.76       214\n",
      "weighted avg       0.77      0.77      0.77       214\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/opt/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:458: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "\n",
    "classifier = LogisticRegression(random_state=0)\n",
    "classifier.fit(X_train, y_train)\n",
    "y_pred = classifier.predict(X_test)\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 101,
   "id": "f8d1b5af-ffd5-4cb4-81ef-d2a2070c992f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.svm import SVC, LinearSVC"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 102,
   "id": "92ba9a0c-383a-41c4-b1bd-0f82cd670bca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.64      0.87      0.74       125\n",
      "           1       0.63      0.30      0.41        89\n",
      "\n",
      "    accuracy                           0.64       214\n",
      "   macro avg       0.63      0.59      0.57       214\n",
      "weighted avg       0.63      0.64      0.60       214\n",
      "\n"
     ]
    }
   ],
   "source": [
    "\n",
    "classifier1 = SVC()\n",
    "classifier1.fit(X_train, y_train)\n",
    "y_pred = classifier1.predict(X_test)\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 103,
   "id": "f8784fc4-d3ee-415e-bd2f-6e0cbe6db01e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.69      0.76      0.73       125\n",
      "           1       0.61      0.53      0.57        89\n",
      "\n",
      "    accuracy                           0.66       214\n",
      "   macro avg       0.65      0.64      0.65       214\n",
      "weighted avg       0.66      0.66      0.66       214\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "classifier2=KNeighborsClassifier(n_neighbors = 3)\n",
    "classifier2.fit(X_train, y_train)\n",
    "y_pred = classifier2.predict(X_test)\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 104,
   "id": "09a64f26-5dcc-402c-b98a-571a167484c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "id": "591d0db1-624e-4ad1-8518-b14a0c9063d6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.79      0.80      0.80       125\n",
      "           1       0.72      0.71      0.71        89\n",
      "\n",
      "    accuracy                           0.76       214\n",
      "   macro avg       0.75      0.75      0.75       214\n",
      "weighted avg       0.76      0.76      0.76       214\n",
      "\n"
     ]
    }
   ],
   "source": [
    "\n",
    "classifier3=RandomForestClassifier()\n",
    "classifier3.fit(X_train, y_train)\n",
    "y_pred =classifier3.predict(X_test)\n",
    "print(classification_report(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "78f01ca9-aea8-411f-a5ed-7b36107ad605",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
