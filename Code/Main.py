"""
===============================================================================
Project Name : Bank Marketing Campaign Analysis
Author       : Rana Daniyal
Language     : Python
Description  : Phase 1 (Data Loading & Understanding)
               Phase 2 (Data Cleaning & Preprocessing)
===============================================================================
"""

# =============================================================================
# IMPORT LIBRARIES
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# =============================================================================
# SETTINGS
# =============================================================================

sys.stdout.reconfigure(encoding="utf-8")

plt.style.use("ggplot")
sns.set_theme(style="whitegrid")

os.makedirs("images", exist_ok=True)

# =============================================================================
# LOAD DATASET
# =============================================================================

df = pd.read_csv("Bank_Marketing_Dataset.csv")

print("=" * 80)
print("BANK MARKETING DATASET LOADED SUCCESSFULLY")
print("=" * 80)

# =============================================================================
# PHASE 1 : DATASET UNDERSTANDING
# =============================================================================

print("\nFirst 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

print("\nDataset Shape")
print(df.shape)

print("\nRows :", df.shape[0])
print("Columns :", df.shape[1])

print("\nDataset Size")
print(df.size)

print("\nColumn Names")
print(df.columns.tolist())

print("\nData Types")
print(df.dtypes)

print("\nDataset Information")
df.info()

print("\nStatistical Summary")
print(df.describe(include="all"))

print("\nMissing Values")
print(df.isnull().sum())

print("\nDuplicate Records")
print(df.duplicated().sum())

print("\nUnique Values")
print(df.nunique())

print("\nMean")
print(df.mean(numeric_only=True))

print("\nMedian")
print(df.median(numeric_only=True))

print("\nMode")
print(df.mode().iloc[0])

# =============================================================================
# PHASE 2 : DATA CLEANING
# =============================================================================

print("\n" + "=" * 80)
print("DATA CLEANING STARTED")
print("=" * 80)

# -----------------------------------------------------------------------------
# Rename Columns
# -----------------------------------------------------------------------------

df.columns = [
    "Age",
    "Job",
    "Marital_Status",
    "Education",
    "Default",
    "Balance",
    "Housing_Loan",
    "Personal_Loan",
    "Contact",
    "Day",
    "Month",
    "Duration",
    "Campaign",
    "Previous_Days",
    "Previous_Contacts",
    "Previous_Outcome",
    "Class"
]

print("\nColumns Renamed Successfully.")

# -----------------------------------------------------------------------------
# Missing Values
# -----------------------------------------------------------------------------

print("\nMissing Values")
print(df.isnull().sum())

# -----------------------------------------------------------------------------
# Duplicate Records
# -----------------------------------------------------------------------------

duplicates = df.duplicated().sum()

print(f"\nDuplicate Records : {duplicates}")

if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicate Records Removed.")
else:
    print("No Duplicate Records Found.")

# -----------------------------------------------------------------------------
# Unknown Values
# -----------------------------------------------------------------------------

print("\nUnknown Values")

categorical_columns = df.select_dtypes(include="object").columns

for col in categorical_columns:

    unknown = (df[col] == "unknown").sum()

    if unknown > 0:
        print(f"{col:<20} : {unknown}")

# -----------------------------------------------------------------------------
# Convert Data Types
# -----------------------------------------------------------------------------

category_columns = [
    "Job",
    "Marital_Status",
    "Education",
    "Default",
    "Housing_Loan",
    "Personal_Loan",
    "Contact",
    "Month",
    "Previous_Outcome",
    "Class"
]

for col in category_columns:
    df[col] = df[col].astype("category")

print("\nCategorical Columns Converted Successfully.")

# -----------------------------------------------------------------------------
# Negative Balance
# -----------------------------------------------------------------------------

negative_balance = (df["Balance"] < 0).sum()

print(f"\nNegative Balance Records : {negative_balance}")

# -----------------------------------------------------------------------------
# Outlier Detection (IQR)
# -----------------------------------------------------------------------------

print("\nOutlier Detection")

numeric_columns = [
    "Age",
    "Balance",
    "Duration",
    "Campaign",
    "Previous_Days",
    "Previous_Contacts"
]

for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"{col:<20} : {len(outliers)} Outliers")

# -----------------------------------------------------------------------------
# Target Variable Analysis
# -----------------------------------------------------------------------------

print("\nTarget Variable Distribution")

print(df["Class"].value_counts())

print("\nTarget Percentage")

print(round(df["Class"].value_counts(normalize=True) * 100, 2))

# -----------------------------------------------------------------------------
# Final Dataset Information
# -----------------------------------------------------------------------------

print("\nFinal Dataset Shape")
print(df.shape)

print("\nFinal Data Types")
print(df.dtypes)

# -----------------------------------------------------------------------------
# Save Clean Dataset
# -----------------------------------------------------------------------------

df.to_csv("Clean_Bank_Marketing_Dataset.csv", index=False)

print("\nClean Dataset Saved Successfully.")

print("\n" + "=" * 80)
print("PHASE 1 & PHASE 2 COMPLETED SUCCESSFULLY")
print("=" * 80)





# =============================================================================
# PHASE 3 : EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 3 : EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# =============================================================================
# CREATE IMAGE FOLDER
# =============================================================================

os.makedirs("images", exist_ok=True)

# =============================================================================
# AGE DISTRIBUTION
# =============================================================================

plt.figure(figsize=(10,6))
sns.histplot(df["Age"], bins=30, kde=True)

plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")

plt.savefig("images/01_age_distribution.png", dpi=300)
plt.show()

print("\nInsight:")
print("- Most customers belong to the middle-age group.")
print("- The bank mainly targets working-age customers.")

# =============================================================================
# JOB DISTRIBUTION
# =============================================================================

plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    y="Job",
    order=df["Job"].value_counts().index
)

plt.title("Customers by Job")

plt.savefig("images/02_job_distribution.png", dpi=300)
plt.show()

print("\nInsight:")
print("- Identify which professions represent the largest customer base.")

# =============================================================================
# MARITAL STATUS
# =============================================================================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Marital_Status"
)

plt.title("Marital Status Distribution")

plt.savefig("images/03_marital_status.png", dpi=300)
plt.show()

print("\nInsight:")
print("- Compare married, single and divorced customers.")

# =============================================================================
# EDUCATION
# =============================================================================

plt.figure(figsize=(9,5))

sns.countplot(
    data=df,
    x="Education"
)

plt.title("Education Level")

plt.savefig("images/04_education.png", dpi=300)
plt.show()

print("\nInsight:")
print("- Understand educational background of customers.")

# =============================================================================
# BALANCE DISTRIBUTION
# =============================================================================

plt.figure(figsize=(10,6))

sns.histplot(df["Balance"], bins=40)

plt.title("Account Balance Distribution")

plt.savefig("images/05_balance_distribution.png", dpi=300)
plt.show()

print("\nInsight:")
print("- Check how customer balances are distributed.")

# =============================================================================
# HOUSING LOAN
# =============================================================================

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="Housing_Loan"
)

plt.title("Housing Loan")

plt.savefig("images/06_housing_loan.png", dpi=300)
plt.show()

# =============================================================================
# PERSONAL LOAN
# =============================================================================

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="Personal_Loan"
)

plt.title("Personal Loan")

plt.savefig("images/07_personal_loan.png", dpi=300)
plt.show()

# =============================================================================
# CONTACT METHOD
# =============================================================================

plt.figure(figsize=(7,5))

sns.countplot(
    data=df,
    x="Contact"
)

plt.title("Contact Method")

plt.savefig("images/08_contact_method.png", dpi=300)
plt.show()

# =============================================================================
# MONTH
# =============================================================================

plt.figure(figsize=(12,6))

sns.countplot(
    data=df,
    x="Month"
)

plt.title("Campaign Month")

plt.savefig("images/09_campaign_month.png", dpi=300)
plt.show()

# =============================================================================
# TARGET VARIABLE
# =============================================================================

plt.figure(figsize=(6,5))

sns.countplot(
    data=df,
    x="Class"
)

plt.title("Subscription Result")

plt.savefig("images/10_target_distribution.png", dpi=300)
plt.show()

print("\nInsight:")
print("- Compare customers who subscribed vs those who did not.")

# =============================================================================
# CORRELATION HEATMAP
# =============================================================================

plt.figure(figsize=(10,8))

sns.heatmap(
    df.select_dtypes(include=np.number).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.savefig("images/11_correlation_heatmap.png", dpi=300)
plt.show()

print("\nInsight:")
print("- Identify relationships among numerical variables.")

# =============================================================================
# BOXPLOT
# =============================================================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df["Balance"]
)

plt.title("Balance Outliers")

plt.savefig("images/12_balance_boxplot.png", dpi=300)
plt.show()

# =============================================================================
# AGE VS BALANCE
# =============================================================================

plt.figure(figsize=(8,6))

sns.scatterplot(
    data=df,
    x="Age",
    y="Balance"
)

plt.title("Age vs Balance")

plt.savefig("images/13_age_balance.png", dpi=300)
plt.show()

print("\nInsight:")
print("- Observe relationship between age and account balance.")

print("\n" + "=" * 80)
print("PHASE 3 COMPLETED")
print("=" * 80)




# =============================================================================
# PHASE 4 : ADVANCED BUSINESS ANALYSIS
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 4 : ADVANCED BUSINESS ANALYSIS")
print("=" * 80)

# =============================================================================
# AGE GROUP ANALYSIS
# =============================================================================

df["Age_Group"] = pd.cut(
    df["Age"],
    bins=[18,30,40,50,60,100],
    labels=["18-30","31-40","41-50","51-60","60+"]
)

age_conversion = pd.crosstab(df["Age_Group"], df["Class"])

print("\nAge Group Conversion")
print(age_conversion)

plt.figure(figsize=(8,5))
age_conversion.plot(kind="bar")
plt.title("Subscription by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Customers")
plt.tight_layout()
plt.savefig("images/14_age_group_subscription.png", dpi=300)
plt.show()

# =============================================================================
# JOB VS SUBSCRIPTION
# =============================================================================

job_conversion = pd.crosstab(df["Job"], df["Class"])

print("\nJob vs Subscription")
print(job_conversion)

job_conversion.plot(
    kind="bar",
    figsize=(12,6)
)

plt.title("Job Category vs Subscription")
plt.tight_layout()
plt.savefig("images/15_job_subscription.png", dpi=300)
plt.show()

# =============================================================================
# EDUCATION VS SUBSCRIPTION
# =============================================================================

education_conversion = pd.crosstab(df["Education"], df["Class"])

print("\nEducation vs Subscription")
print(education_conversion)

education_conversion.plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Education vs Subscription")
plt.tight_layout()
plt.savefig("images/16_education_subscription.png", dpi=300)
plt.show()

# =============================================================================
# MARITAL STATUS VS SUBSCRIPTION
# =============================================================================

marital_conversion = pd.crosstab(df["Marital_Status"], df["Class"])

marital_conversion.plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Marital Status vs Subscription")
plt.tight_layout()
plt.savefig("images/17_marital_subscription.png", dpi=300)
plt.show()

# =============================================================================
# BALANCE VS SUBSCRIPTION
# =============================================================================

plt.figure(figsize=(10,6))

sns.boxplot(
    data=df,
    x="Class",
    y="Balance"
)

plt.title("Balance vs Subscription")

plt.savefig("images/18_balance_subscription.png", dpi=300)
plt.show()

# =============================================================================
# DURATION VS SUBSCRIPTION
# =============================================================================

plt.figure(figsize=(10,6))

sns.boxplot(
    data=df,
    x="Class",
    y="Duration"
)

plt.title("Call Duration vs Subscription")

plt.savefig("images/19_duration_subscription.png", dpi=300)
plt.show()

# =============================================================================
# MONTH VS SUBSCRIPTION
# =============================================================================

month_conversion = pd.crosstab(df["Month"], df["Class"])

month_conversion.plot(
    kind="bar",
    figsize=(12,6)
)

plt.title("Campaign Month vs Subscription")

plt.tight_layout()

plt.savefig("images/20_month_subscription.png", dpi=300)
plt.show()

# =============================================================================
# CONTACT METHOD
# =============================================================================

contact_conversion = pd.crosstab(df["Contact"], df["Class"])

contact_conversion.plot(
    kind="bar",
    figsize=(8,5)
)

plt.title("Contact Method vs Subscription")

plt.tight_layout()

plt.savefig("images/21_contact_subscription.png", dpi=300)
plt.show()

# =============================================================================
# PREVIOUS CAMPAIGN
# =============================================================================

previous_conversion = pd.crosstab(df["Previous_Outcome"], df["Class"])

previous_conversion.plot(
    kind="bar",
    figsize=(10,5)
)

plt.title("Previous Campaign Outcome")

plt.tight_layout()

plt.savefig("images/22_previous_campaign.png", dpi=300)
plt.show()

# =============================================================================
# TOP 10 JOBS BY AVERAGE BALANCE
# =============================================================================

top_balance = (
    df.groupby("Job")["Balance"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,6))

top_balance.plot(kind="bar")

plt.title("Top 10 Jobs by Average Balance")

plt.ylabel("Average Balance")

plt.tight_layout()

plt.savefig("images/23_top_balance_jobs.png", dpi=300)
plt.show()

# =============================================================================
# CAMPAIGN DISTRIBUTION
# =============================================================================

plt.figure(figsize=(10,6))

sns.histplot(df["Campaign"], bins=30)

plt.title("Campaign Contact Distribution")

plt.savefig("images/24_campaign_distribution.png", dpi=300)
plt.show()

# =============================================================================
# BUSINESS KPIs
# =============================================================================

print("\n" + "="*80)
print("BUSINESS KPIs")
print("="*80)

print(f"\nTotal Customers : {len(df):,}")

print(f"Average Age : {df['Age'].mean():.2f}")

print(f"Average Balance : {df['Balance'].mean():,.2f}")

print(f"Average Call Duration : {df['Duration'].mean():.2f}")

print(f"Average Campaign Contacts : {df['Campaign'].mean():.2f}")

print("\nSubscription Percentage")

print(round(df["Class"].value_counts(normalize=True)*100,2))

# =============================================================================
# TOP CUSTOMER PROFILE
# =============================================================================

print("\nMost Common Job")
print(df["Job"].mode()[0])

print("\nMost Common Education")
print(df["Education"].mode()[0])

print("\nMost Common Marital Status")
print(df["Marital_Status"].mode()[0])

print("\nMost Used Contact Method")
print(df["Contact"].mode()[0])

print("\nBest Campaign Month")
print(df["Month"].mode()[0])

print("\n" + "="*80)
print("PHASE 4 COMPLETED SUCCESSFULLY")
print("="*80)




# =============================================================================
# PHASE 5 : BUSINESS INSIGHTS & FINAL REPORT
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 5 : BUSINESS INSIGHTS & FINAL REPORT")
print("=" * 80)

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================

total_customers = len(df)

subscribed = (df["Class"] == "yes").sum()
not_subscribed = (df["Class"] == "no").sum()

subscription_rate = (subscribed / total_customers) * 100

print("\nEXECUTIVE SUMMARY")
print("-" * 80)

print(f"Total Customers           : {total_customers:,}")
print(f"Subscribed Customers      : {subscribed:,}")
print(f"Not Subscribed Customers  : {not_subscribed:,}")
print(f"Subscription Rate         : {subscription_rate:.2f}%")

# =============================================================================
# TOP JOBS
# =============================================================================

print("\nTop 10 Customer Jobs")

print(df["Job"].value_counts().head(10))

# =============================================================================
# AGE INSIGHTS
# =============================================================================

print("\nAge Statistics")

print(df["Age"].describe())

# =============================================================================
# BALANCE INSIGHTS
# =============================================================================

print("\nBalance Statistics")

print(df["Balance"].describe())

# =============================================================================
# CAMPAIGN INSIGHTS
# =============================================================================

print("\nCampaign Statistics")

print(df["Campaign"].describe())

# =============================================================================
# BUSINESS RECOMMENDATIONS
# =============================================================================

recommendations = [

"1. Focus marketing campaigns on the highest converting customer segments.",

"2. Increase campaign efforts during the best-performing months.",

"3. Prioritize customers with higher account balances.",

"4. Improve campaign strategies for low-performing customer groups.",

"5. Optimize call duration to improve conversion rates.",

"6. Use previous campaign results to target likely subscribers.",

"7. Develop personalized offers for different job categories.",

"8. Strengthen digital marketing for younger customers.",

"9. Reduce unnecessary campaign contacts to lower marketing costs.",

"10. Build predictive models for future marketing campaigns."

]

print("\nBUSINESS RECOMMENDATIONS")

for rec in recommendations:
    print(rec)

# =============================================================================
# EXPORT BUSINESS REPORT
# =============================================================================

with open("Business_Report.txt","w",encoding="utf-8") as file:

    file.write("="*80 + "\n")
    file.write("BANK MARKETING CAMPAIGN ANALYSIS\n")
    file.write("="*80 + "\n\n")

    file.write(f"Total Customers : {total_customers}\n")
    file.write(f"Subscribed : {subscribed}\n")
    file.write(f"Not Subscribed : {not_subscribed}\n")
    file.write(f"Subscription Rate : {subscription_rate:.2f}%\n\n")

    file.write("TOP JOBS\n")
    file.write(str(df["Job"].value_counts().head(10)))
    file.write("\n\n")

    file.write("AGE SUMMARY\n")
    file.write(str(df["Age"].describe()))
    file.write("\n\n")

    file.write("BALANCE SUMMARY\n")
    file.write(str(df["Balance"].describe()))
    file.write("\n\n")

    file.write("BUSINESS RECOMMENDATIONS\n\n")

    for rec in recommendations:
        file.write(rec + "\n")

print("\nBusiness_Report.txt Created Successfully.")

# =============================================================================
# SAVE FINAL DATASET
# =============================================================================

df.to_csv("Final_Bank_Marketing_Dataset.csv", index=False)

print("\nFinal Dataset Saved Successfully.")

# =============================================================================
# FINAL PROJECT SUMMARY
# =============================================================================

print("\n" + "="*80)
print("FINAL PROJECT SUMMARY")
print("="*80)

print("""
Project Completed Successfully.

Modules Completed

✔ Phase 1 : Data Loading & Understanding
✔ Phase 2 : Data Cleaning
✔ Phase 3 : Exploratory Data Analysis
✔ Phase 4 : Advanced Business Analysis
✔ Phase 5 : Business Report & Recommendations

Outputs Generated

✔ Clean Dataset
✔ Final Dataset
✔ Business Report
✔ 24+ Professional Charts
✔ Business KPIs
✔ Executive Summary
✔ Business Recommendations

Project Status : COMPLETED
""")

print("="*80)