import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

########## PROCESS DATA ##########

# Load the dataset
df = pd.read_csv('employee_attrition.csv')
# Display the first 5 rows of data
df.head()

# Display the descriptive statistics for the dataset
df.describe()

# Drop columns that do not provide helpful information
# 'EmployeeCount', 'Over18', 'StandardHours' all have one value
# 'EmployeeNumber' does not provide helpful info
df.drop(['EmployeeCount', 'EmployeeNumber', 'Over18',
        'StandardHours'], axis="columns", inplace=True)
# Display the new first 5 rows of data
df.head()

# Count the number of employees who left vs stayed
df['Attrition'].value_counts()

# Convert 'Attrition' to binary equivalent
df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

########## VISUALIZE DATA ##########

# Analyze the attrition rate by monthly income and job role
# along with the percent difference in income between employees who left and stayed

# Group by 'JobRole' and calculate the attrition rate for each group
attrition_by_job_role = df.groupby('JobRole')['Attrition'].mean().reset_index()
attrition_by_job_role = attrition_by_job_role.sort_values(
    by='Attrition', ascending=False)
# Display the first 5 rows of attrition rate by job role
attrition_by_job_role.head()

# Calculate the percent difference in income between employees who left and stayed
income_diff = df.groupby(['JobRole', 'Attrition'])[
    'MonthlyIncome'].mean().unstack()
income_diff['Percent Difference'] = (
    (income_diff[1] - income_diff[0]) / income_diff[0]) * 100

# Align the sorting of income_diff with attrition_by_job_role
income_diff = income_diff.reindex(attrition_by_job_role['JobRole'])

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
palette = sns.color_palette("pastel", len(attrition_by_job_role['JobRole']))
color_map = dict(zip(attrition_by_job_role['JobRole'], palette))

# Plot the attrition rate by job role
sns.barplot(y='Attrition', x='JobRole', data=attrition_by_job_role,
            ax=axes[0], hue='JobRole', palette=color_map, dodge=False)
axes[0].set_ylabel('Employee Attrition Rate')
axes[0].set_xlabel('Job Role')
axes[0].set_title('Employee Attrition Rate by Job Role')
axes[0].set_xticks(range(len(attrition_by_job_role['JobRole'])))
axes[0].set_xticklabels(attrition_by_job_role['JobRole'], rotation=45)

# Plot the percent difference in income
sns.barplot(x=income_diff.index, y=income_diff['Percent Difference'],
            ax=axes[1], hue=income_diff.index, palette=color_map, dodge=False)
axes[1].set_xlabel('Job Role')
axes[1].set_ylabel('Percent Difference (%) in Monthly Income')
axes[1].set_title(
    'Percent Difference (%) in Monthly Income Between Employees Who Left and Stayed by Job Role')
axes[1].set_xticks(range(len(income_diff.index)))
axes[1].set_xticklabels(income_diff.index, rotation=45)

plt.tight_layout()
plt.show()

# Analyze the attrition rate by distance from home

# Group by 'DistanceFromHome' and calculate the attrition rate for each group
attrition_by_distance = df.groupby('DistanceFromHome')[
    'Attrition'].mean().reset_index()
# Display the first 5 rows of attrition rate by distance
attrition_by_distance.head()

# Plot the attrition rate by distance from home with a trend line on a bar plot
plt.figure(figsize=(12, 8))
sns.barplot(x='DistanceFromHome', y='Attrition',
            data=attrition_by_distance, label='Attrition Rate')
sns.regplot(x='DistanceFromHome', y='Attrition', data=attrition_by_distance,
            scatter=False, color='red', label='Linear Regression (Trend Line)')
plt.xlabel('Office Distance From Home (No Unit Provided)')
plt.ylabel('Employee Attrition Rate')
plt.title('Employee Attrition Rate by Office Distance From Home')
plt.legend()
plt.show()

# Analyze the attrition rate by satisfaction / performance metrics

# Group by satisfaction / performance metrics and calculate the attrition rate for each group
satisfaction_metrics = ['EnvironmentSatisfaction', 'JobSatisfaction',
                        'RelationshipSatisfaction', 'JobInvolvement', 'PerformanceRating', 'WorkLifeBalance']
attrition_by_metrics = df.groupby(satisfaction_metrics)[
    'Attrition'].mean().reset_index()
# Display the first 5 rows of attrition rate by satisfaction / performance metrics
attrition_by_metrics.head()

# Melt the data to have metrics in a single column
melted_attrition_rates = attrition_by_metrics.melt(
    id_vars='Attrition', value_vars=satisfaction_metrics, var_name='SatisfactionMetric', value_name='SatisfactionLevel')

# Plot the attrition rate by satisfaction / performance metric
plt.figure(figsize=(14, 8))
sns.lineplot(x='SatisfactionLevel', y='Attrition',
             hue='SatisfactionMetric', data=melted_attrition_rates, marker='o', errorbar=None)
plt.xlabel('Level of Degree (Low to High)')
plt.ylabel('Employee Attrition Rate')
plt.title('Employee Attrition Rate by Various Performance & Satisfaction Metrics')
plt.legend(title='Satisfaction / Performance Metric')
plt.show()

# Analyze the attrition rate by business travel frequency and education field

# Group by 'BusinessTravel' and 'EducationField' and calculate the attrition rate for each group
attrition_by_travel_education = df.groupby(['BusinessTravel', 'EducationField'])[
    'Attrition'].mean().reset_index()
# Display the first 5 rows of attrition rate by business travel frequency and education field
attrition_by_travel_education.head()

# Plot the attrition rate by business travel frequency and education field
plt.figure(figsize=(12, 8))
palette = sns.color_palette("pastel", len(df['BusinessTravel'].unique()))
color_map = dict(zip(df['BusinessTravel'].unique(), palette))
sns.lineplot(x='EducationField', y='Attrition', data=attrition_by_travel_education,
             hue='BusinessTravel', palette=color_map, marker='o')
plt.xlabel('Education Field')
plt.ylabel('Employee Attrition Rate')
plt.title('Employee Attrition Rate by Business Travel Frequency and Education Field')
plt.legend(title='Business Travel Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
