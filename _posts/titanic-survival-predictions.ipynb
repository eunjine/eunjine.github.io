{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "cc656203",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:33.603993Z",
     "iopub.status.busy": "2022-04-09T15:43:33.602873Z",
     "iopub.status.idle": "2022-04-09T15:43:34.851794Z",
     "shell.execute_reply": "2022-04-09T15:43:34.851079Z",
     "shell.execute_reply.started": "2022-04-09T15:42:56.888935Z"
    },
    "papermill": {
     "duration": 1.270807,
     "end_time": "2022-04-09T15:43:34.851954",
     "exception": false,
     "start_time": "2022-04-09T15:43:33.581147",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import OneHotEncoder\n",
    "from sklearn.pipeline import Pipeline\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import OrdinalEncoder\n",
    "from sklearn.preprocessing import MinMaxScaler"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d237c1e0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:34.891374Z",
     "iopub.status.busy": "2022-04-09T15:43:34.890683Z",
     "iopub.status.idle": "2022-04-09T15:43:34.921229Z",
     "shell.execute_reply": "2022-04-09T15:43:34.921690Z",
     "shell.execute_reply.started": "2022-04-09T15:42:56.912272Z"
    },
    "papermill": {
     "duration": 0.052806,
     "end_time": "2022-04-09T15:43:34.921872",
     "exception": false,
     "start_time": "2022-04-09T15:43:34.869066",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "train_data = pd.read_csv(\"../input/titanic/train.csv\")\n",
    "test_data = pd.read_csv(\"../input/titanic/test.csv\")\n",
    "gender_submission = pd.read_csv('../input/titanic/gender_submission.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "032a4db0",
   "metadata": {
    "papermill": {
     "duration": 0.016223,
     "end_time": "2022-04-09T15:43:34.954809",
     "exception": false,
     "start_time": "2022-04-09T15:43:34.938586",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Let's fetch the data and load it:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0a26ba4a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:34.997218Z",
     "iopub.status.busy": "2022-04-09T15:43:34.996528Z",
     "iopub.status.idle": "2022-04-09T15:43:35.012400Z",
     "shell.execute_reply": "2022-04-09T15:43:35.012849Z",
     "shell.execute_reply.started": "2022-04-09T15:42:56.934278Z"
    },
    "papermill": {
     "duration": 0.040857,
     "end_time": "2022-04-09T15:43:35.013022",
     "exception": false,
     "start_time": "2022-04-09T15:43:34.972165",
     "status": "completed"
    },
    "tags": []
   },
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
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7a4d8dc7",
   "metadata": {
    "papermill": {
     "duration": 0.0175,
     "end_time": "2022-04-09T15:43:35.048030",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.030530",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "attributes :\n",
    "* **PassengerId**: a unique identifier for each passenger\n",
    "* **Survived**: that's the target, 0 means the passenger did not survive, while 1 means he/she survived.\n",
    "* **Pclass**: passenger class.\n",
    "* **Name**, **Sex**, **Age**: self-explanatory\n",
    "* **SibSp**: how many siblings & spouses of the passenger aboard the Titanic.\n",
    "* **Parch**: how many children & parents of the passenger aboard the Titanic.\n",
    "* **Ticket**: ticket id\n",
    "* **Fare**: price paid (in pounds)\n",
    "* **Cabin**: passenger's cabin number\n",
    "* **Embarked**: where the passenger embarked the Titanic"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6be7e1d0",
   "metadata": {
    "papermill": {
     "duration": 0.016997,
     "end_time": "2022-04-09T15:43:35.082174",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.065177",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "Age, Cabin, *Embarked attributes are sometimes null (less than 891 non-null). especially the Cabin (77% are null). The Age attribute has about 19% null values, so we will need to decide what to do with them. Replacing null values with the median age seems reasonable."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "509fac27",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:35.120270Z",
     "iopub.status.busy": "2022-04-09T15:43:35.119603Z",
     "iopub.status.idle": "2022-04-09T15:43:35.126935Z",
     "shell.execute_reply": "2022-04-09T15:43:35.127471Z",
     "shell.execute_reply.started": "2022-04-09T15:42:56.963676Z"
    },
    "papermill": {
     "duration": 0.028438,
     "end_time": "2022-04-09T15:43:35.127654",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.099216",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "PassengerId      0\n",
      "Survived         0\n",
      "Pclass           0\n",
      "Name             0\n",
      "Sex              0\n",
      "Age            177\n",
      "SibSp            0\n",
      "Parch            0\n",
      "Ticket           0\n",
      "Fare             0\n",
      "Cabin          687\n",
      "Embarked         2\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "print(pd.isnull(train_data).sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "597f518e",
   "metadata": {
    "papermill": {
     "duration": 0.017379,
     "end_time": "2022-04-09T15:43:35.162799",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.145420",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "예측: Sex: Females are more likely to survive. SibSp/Parch: People traveling alone are more likely to survive. Age was less than 30 years old. Pclass: People of higher socioeconomic class are more likely to survive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e508bb1a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:35.201839Z",
     "iopub.status.busy": "2022-04-09T15:43:35.201177Z",
     "iopub.status.idle": "2022-04-09T15:43:35.213359Z",
     "shell.execute_reply": "2022-04-09T15:43:35.213946Z",
     "shell.execute_reply.started": "2022-04-09T15:42:56.976829Z"
    },
    "papermill": {
     "duration": 0.033625,
     "end_time": "2022-04-09T15:43:35.214112",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.180487",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "27.0"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data[train_data[\"Sex\"]==\"female\"][\"Age\"].median()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "316d8587",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:35.254113Z",
     "iopub.status.busy": "2022-04-09T15:43:35.253483Z",
     "iopub.status.idle": "2022-04-09T15:43:35.286907Z",
     "shell.execute_reply": "2022-04-09T15:43:35.286144Z",
     "shell.execute_reply.started": "2022-04-09T15:42:56.993542Z"
    },
    "papermill": {
     "duration": 0.054386,
     "end_time": "2022-04-09T15:43:35.287071",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.232685",
     "status": "completed"
    },
    "tags": []
   },
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
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>714.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "      <td>891.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.383838</td>\n",
       "      <td>2.308642</td>\n",
       "      <td>29.699118</td>\n",
       "      <td>0.523008</td>\n",
       "      <td>0.381594</td>\n",
       "      <td>32.204208</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>257.353842</td>\n",
       "      <td>0.486592</td>\n",
       "      <td>0.836071</td>\n",
       "      <td>14.526497</td>\n",
       "      <td>1.102743</td>\n",
       "      <td>0.806057</td>\n",
       "      <td>49.693429</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.420000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>223.500000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>2.000000</td>\n",
       "      <td>20.125000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>7.910400</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>446.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>28.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>14.454200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>668.500000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>38.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>31.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>891.000000</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>80.000000</td>\n",
       "      <td>8.000000</td>\n",
       "      <td>6.000000</td>\n",
       "      <td>512.329200</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       PassengerId    Survived      Pclass         Age       SibSp  \\\n",
       "count   891.000000  891.000000  891.000000  714.000000  891.000000   \n",
       "mean    446.000000    0.383838    2.308642   29.699118    0.523008   \n",
       "std     257.353842    0.486592    0.836071   14.526497    1.102743   \n",
       "min       1.000000    0.000000    1.000000    0.420000    0.000000   \n",
       "25%     223.500000    0.000000    2.000000   20.125000    0.000000   \n",
       "50%     446.000000    0.000000    3.000000   28.000000    0.000000   \n",
       "75%     668.500000    1.000000    3.000000   38.000000    1.000000   \n",
       "max     891.000000    1.000000    3.000000   80.000000    8.000000   \n",
       "\n",
       "            Parch        Fare  \n",
       "count  891.000000  891.000000  \n",
       "mean     0.381594   32.204208  \n",
       "std      0.806057   49.693429  \n",
       "min      0.000000    0.000000  \n",
       "25%      0.000000    7.910400  \n",
       "50%      0.000000   14.454200  \n",
       "75%      0.000000   31.000000  \n",
       "max      6.000000  512.329200  "
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fb2885fb",
   "metadata": {
    "papermill": {
     "duration": 0.017846,
     "end_time": "2022-04-09T15:43:35.325099",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.307253",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "전처리 파이프라인을 구축 :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "85a2d77d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:35.366075Z",
     "iopub.status.busy": "2022-04-09T15:43:35.364210Z",
     "iopub.status.idle": "2022-04-09T15:43:35.368258Z",
     "shell.execute_reply": "2022-04-09T15:43:35.368764Z",
     "shell.execute_reply.started": "2022-04-09T15:42:57.046020Z"
    },
    "papermill": {
     "duration": 0.025595,
     "end_time": "2022-04-09T15:43:35.368936",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.343341",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "num_pipeline = Pipeline([\n",
    "        (\"imputer\", SimpleImputer(strategy=\"median\")),\n",
    "        (\"scaler\", StandardScaler())\n",
    "    ])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "79f17df0",
   "metadata": {
    "papermill": {
     "duration": 0.017873,
     "end_time": "2022-04-09T15:43:35.405271",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.387398",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "범주 속성을 위한 파이프라인을 구축:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "622a2989",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:35.444934Z",
     "iopub.status.busy": "2022-04-09T15:43:35.444275Z",
     "iopub.status.idle": "2022-04-09T15:43:35.448209Z",
     "shell.execute_reply": "2022-04-09T15:43:35.448614Z",
     "shell.execute_reply.started": "2022-04-09T15:42:57.052712Z"
    },
    "papermill": {
     "duration": 0.025549,
     "end_time": "2022-04-09T15:43:35.448810",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.423261",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "cat_pipeline = Pipeline([\n",
    "        (\"ordinal_encoder\", OrdinalEncoder()),    \n",
    "        (\"imputer\", SimpleImputer(strategy=\"most_frequent\")),\n",
    "        (\"cat_encoder\", OneHotEncoder(sparse=False)),\n",
    "    ])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "f8677c8e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:35.491328Z",
     "iopub.status.busy": "2022-04-09T15:43:35.490695Z",
     "iopub.status.idle": "2022-04-09T15:43:35.492344Z",
     "shell.execute_reply": "2022-04-09T15:43:35.492928Z",
     "shell.execute_reply.started": "2022-04-09T15:42:57.071642Z"
    },
    "papermill": {
     "duration": 0.025838,
     "end_time": "2022-04-09T15:43:35.493094",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.467256",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "num_attribs = [\"Age\", \"SibSp\", \"Parch\", \"Fare\"]\n",
    "cat_attribs = [\"Pclass\", \"Sex\", \"Embarked\"]\n",
    "\n",
    "preprocess_pipeline = ColumnTransformer([\n",
    "        (\"num\", num_pipeline, num_attribs),\n",
    "        (\"cat\", cat_pipeline, cat_attribs),\n",
    "    ])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "41b8c3d2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:35.533422Z",
     "iopub.status.busy": "2022-04-09T15:43:35.532803Z",
     "iopub.status.idle": "2022-04-09T15:43:35.550106Z",
     "shell.execute_reply": "2022-04-09T15:43:35.550667Z",
     "shell.execute_reply.started": "2022-04-09T15:42:57.089326Z"
    },
    "papermill": {
     "duration": 0.039395,
     "end_time": "2022-04-09T15:43:35.550838",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.511443",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-0.56573646,  0.43279337, -0.47367361, ...,  0.        ,\n",
       "         0.        ,  1.        ],\n",
       "       [ 0.66386103,  0.43279337, -0.47367361, ...,  1.        ,\n",
       "         0.        ,  0.        ],\n",
       "       [-0.25833709, -0.4745452 , -0.47367361, ...,  0.        ,\n",
       "         0.        ,  1.        ],\n",
       "       ...,\n",
       "       [-0.1046374 ,  0.43279337,  2.00893337, ...,  0.        ,\n",
       "         0.        ,  1.        ],\n",
       "       [-0.25833709, -0.4745452 , -0.47367361, ...,  1.        ,\n",
       "         0.        ,  0.        ],\n",
       "       [ 0.20276197, -0.4745452 , -0.47367361, ...,  0.        ,\n",
       "         1.        ,  0.        ]])"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train = preprocess_pipeline.fit_transform(train_data)\n",
    "X_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "4be235df",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:35.590710Z",
     "iopub.status.busy": "2022-04-09T15:43:35.590136Z",
     "iopub.status.idle": "2022-04-09T15:43:35.593562Z",
     "shell.execute_reply": "2022-04-09T15:43:35.594053Z",
     "shell.execute_reply.started": "2022-04-09T15:42:57.122605Z"
    },
    "papermill": {
     "duration": 0.024967,
     "end_time": "2022-04-09T15:43:35.594220",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.569253",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "y_train = train_data[\"Survived\"]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7acafe76",
   "metadata": {
    "papermill": {
     "duration": 0.017956,
     "end_time": "2022-04-09T15:43:35.630856",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.612900",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**RandomForestClassifier** :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "669abaa0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:35.671764Z",
     "iopub.status.busy": "2022-04-09T15:43:35.671124Z",
     "iopub.status.idle": "2022-04-09T15:43:35.939808Z",
     "shell.execute_reply": "2022-04-09T15:43:35.940291Z",
     "shell.execute_reply.started": "2022-04-09T15:42:57.128502Z"
    },
    "papermill": {
     "duration": 0.290692,
     "end_time": "2022-04-09T15:43:35.940462",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.649770",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestClassifier(random_state=42)"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.ensemble import RandomForestClassifier\n",
    "\n",
    "forest_clf = RandomForestClassifier(n_estimators=100, random_state=42) #n_estimators=100 결정트리개수\n",
    "forest_clf.fit(X_train, y_train) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "dbe3d328",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:35.981032Z",
     "iopub.status.busy": "2022-04-09T15:43:35.980430Z",
     "iopub.status.idle": "2022-04-09T15:43:36.007969Z",
     "shell.execute_reply": "2022-04-09T15:43:36.008401Z",
     "shell.execute_reply.started": "2022-04-09T15:42:57.460151Z"
    },
    "papermill": {
     "duration": 0.049216,
     "end_time": "2022-04-09T15:43:36.008573",
     "exception": false,
     "start_time": "2022-04-09T15:43:35.959357",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "X_test = preprocess_pipeline.transform(test_data)\n",
    "y_pred = forest_clf.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "421a729a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:36.051901Z",
     "iopub.status.busy": "2022-04-09T15:43:36.051220Z",
     "iopub.status.idle": "2022-04-09T15:43:37.794084Z",
     "shell.execute_reply": "2022-04-09T15:43:37.793589Z",
     "shell.execute_reply.started": "2022-04-09T15:42:57.494394Z"
    },
    "papermill": {
     "duration": 1.76656,
     "end_time": "2022-04-09T15:43:37.794216",
     "exception": false,
     "start_time": "2022-04-09T15:43:36.027656",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8092759051186016\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10, scoring='accuracy')\n",
    "acc_randomforest = forest_scores.mean() \n",
    "print(acc_randomforest)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "420ad5db",
   "metadata": {
    "papermill": {
     "duration": 0.019038,
     "end_time": "2022-04-09T15:43:37.832833",
     "exception": false,
     "start_time": "2022-04-09T15:43:37.813795",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "**SVC** :"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a7d67564",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:37.876318Z",
     "iopub.status.busy": "2022-04-09T15:43:37.875730Z",
     "iopub.status.idle": "2022-04-09T15:43:38.168644Z",
     "shell.execute_reply": "2022-04-09T15:43:38.168131Z",
     "shell.execute_reply.started": "2022-04-09T15:42:59.791017Z"
    },
    "papermill": {
     "duration": 0.316568,
     "end_time": "2022-04-09T15:43:38.168781",
     "exception": false,
     "start_time": "2022-04-09T15:43:37.852213",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8249313358302123\n"
     ]
    }
   ],
   "source": [
    "from sklearn.svm import SVC\n",
    "\n",
    "svm_clf = SVC(gamma=\"auto\")\n",
    "svm_clf.fit(X_train, y_train)\n",
    "X_test = preprocess_pipeline.transform(test_data)\n",
    "y_pred = svm_clf.predict(X_test)\n",
    "svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)\n",
    "acc_svc = svm_scores.mean()\n",
    "print(acc_svc)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8a92b512",
   "metadata": {
    "papermill": {
     "duration": 0.019936,
     "end_time": "2022-04-09T15:43:38.208810",
     "exception": false,
     "start_time": "2022-04-09T15:43:38.188874",
     "status": "completed"
    },
    "tags": []
   },
   "source": [
    "SVC가 더 좋음"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "a62198ba",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2022-04-09T15:43:38.252031Z",
     "iopub.status.busy": "2022-04-09T15:43:38.251102Z",
     "iopub.status.idle": "2022-04-09T15:43:38.270250Z",
     "shell.execute_reply": "2022-04-09T15:43:38.270731Z",
     "shell.execute_reply.started": "2022-04-09T15:43:00.107808Z"
    },
    "papermill": {
     "duration": 0.041909,
     "end_time": "2022-04-09T15:43:38.270929",
     "exception": false,
     "start_time": "2022-04-09T15:43:38.229020",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "gender_submission['Survived'] = svm_clf.predict(X_test).astype(int)\n",
    "\n",
    "gender_submission.to_csv('submission.csv', index=False)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.7.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 13.924845,
   "end_time": "2022-04-09T15:43:39.000789",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2022-04-09T15:43:25.075944",
   "version": "2.3.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
