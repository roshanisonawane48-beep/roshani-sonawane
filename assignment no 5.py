import pandas as pd

df1=pd.read_csv("output.csv")
print(df1)
print("\nmarks Greater than 80:\n")
df2=df1[(df1["Branch"]=="CSE") & (df1["Marks"] > 80)]
print(df2)
print("\nmarks greater than 70:\n")
df3=df1[(df1["Branch"]=="ECE") & (df1["Marks"] > 70)]
print("\nsorting:\n")
print(df3)
df4= df1.sort_values(by=["Branch", "Marks"])
print(df4)
