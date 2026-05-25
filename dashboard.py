import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("Personal Expenses Analytics Dashboard")

# Load data
df = pd.read_csv("personal_finance.csv")

# Convert Date properly
df['Date'] = pd.to_datetime(df['Date'])

st.subheader("Dataset")
st.dataframe(df)

# ---------------- TOTAL EXPENSE ----------------
st.subheader("Total Expense")
st.write(df['Amount'].sum())

# ---------------- CATEGORY WISE EXPENSE ----------------
st.subheader("Category Wise Expense")

category_expense = df.groupby('Category')['Amount'].sum()

fig, ax = plt.subplots()
category_expense.plot(kind='bar', ax=ax)
ax.set_ylabel("Amount")
st.pyplot(fig)

# ---------------- CATEGORY PIE CHART ----------------
st.subheader("Category Wise Distribution (Pie Chart)")

fig2, ax2 = plt.subplots()
ax2.pie(category_expense, labels=category_expense.index, autopct='%1.1f%%')
ax2.axis('equal')

st.pyplot(fig2)

# ---------------- MONTHLY EXPENSE ----------------
st.subheader("Monthly Expense Trend")

df['Month'] = df['Date'].dt.to_period('M')

monthly_expense = df.groupby('Month')['Amount'].sum().sort_index()

fig3, ax3 = plt.subplots()
monthly_expense.plot(kind='line', marker='o', ax=ax3)
ax3.set_ylabel("Amount")
ax3.set_xlabel("Month")

st.pyplot(fig3)

# ---------------- EXPENSE DISTRIBUTION ----------------
st.subheader("Expense Distribution")

fig4, ax4 = plt.subplots()
ax4.hist(df['Amount'], bins=10, edgecolor='black')
ax4.set_xlabel("Expense Amount")
ax4.set_ylabel("Frequency")

st.pyplot(fig4)

# ---------------- PREDICTION ----------------
st.subheader("Expense Prediction (Next Month)")

# Prepare data for ML model
X = np.arange(len(monthly_expense)).reshape(-1, 1)
y = monthly_expense.values

model = LinearRegression()
model.fit(X, y)

next_month_index = np.array([[len(monthly_expense)]])
predicted = model.predict(next_month_index)

st.write("Predicted Next Month Expense:", round(predicted[0], 2))
