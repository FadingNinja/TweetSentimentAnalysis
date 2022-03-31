from dataclasses import replace
from pyexpat import model
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("twitter_sentiment_data.csv")

df["format"] = (df["message"]
.replace(r"@[A-Za-z0-9_]+", "", regex = True)
.replace(r"#[A-Za-z0-9_]+", "", regex = True)
.replace(r"http\S+", "", regex = True)
)
#print(df["format"][1])

x_train, x_test, y_train, y_test = train_test_split(
df["format"], df["sentiment"], test_size = 0.20, 
random_state = 1
)

#print(x_train) #Name: format, Length: 35154, dtype: object
#print(x_test) #Name: format, Length: 8789, dtype: object
#print(y_train) #Name: sentiment, Length: 35154, dtype: int64
#print(y_test) #Name: sentiment, Length: 8789, dtype: int64

count_vector = CountVectorizer()
training_data = count_vector.fit_transform(x_train)
testing_data = count_vector.transform(x_test)

model = lr(random_state = 0, max_iter = 10000)
model.fit(training_data, y_train)

predictions = model.predict(testing_data)

print("Accuracy Score : {}".format(accuracy_score(y_test, predictions)))

con_inp = input("Enter a tweet : ")
inp = np.array(con_inp)
inp = np.reshape(inp, (1, -1))
inp_conv = count_vector.transform(inp.ravel())
result = model.predict(inp_conv)
#print(result)

for i in result :
    if i == 0 :
        print("Nuetral tweet")
    elif i == 1 :
        print("Positive tweet")
    elif i == -1 :
        print("Negative tweet")
    else :
        print("Excellent tweet")