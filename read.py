from dataclasses import replace
import pandas as panda

df = panda.read_csv("twitter_sentiment_data.csv")
#print(df)

df["format"] = (df["message"]
.replace(r"@[A-Za-z0-9_]+", "", regex = True)
.replace(r"#[A-Za-z0-9_]+", "", regex = True)
.replace(r"http\S+", "", regex = True)
)

print(df["format"][10])