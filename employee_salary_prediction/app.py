import numpy as np
import pandas as pd
import plotly.express as px
from plotly.offline import iplot

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")
pd.set_option('future.no_silent_downcasting', True)
pd.options.mode.copy_on_write = "warn"


def add_line(x0=0, y0=0, x1=0, y1=0, 
             line_color='#00DFA2', font_color = "#3C486B", 
             xposition = "right", text = "Text"):
    fig.add_shape(type='line',
                  x0 = x0,
                  y0 = y0,
                  x1 = x1,
                  y1 = y1 + 2,
                  line= {
                      "color": line_color,
                      "width": 3,
                      "dash": "dashdot"
                  },
                  label = {
                      "text": f"\t{text}: {x1: 0.1f}\t".expandtabs(5),
                      "textposition": "end",
                      "yanchor" :"top",
                      "xanchor" :xposition,
                      "textangle" :0,
                      "font": {
                          "size": 14,
                          "color" :font_color,
                          "family" : "arial"

                      },
                  })

def custom_layout(title_size = 28, hover_font_size = 16, showlegend=False):
    fig.update_layout(
        showlegend = showlegend,
        title= {
            "font": {
                "size": title_size,
                "family": "tahoma"
            }
        },
        hoverlabel = {
            "bgcolor": "#111",
            "font_size": hover_font_size,
            "font_family": "arial"
        }
    )



df = pd.read_csv('employee_salary_dataset.csv')
df.info()
print("mean values are: \n", df.describe().T)

print("isna sum is: \n", df.isna().sum())
print(df[df["Age"].isna()])

df.dropna(inplace=True)
print("isna sum is: \n", df.isna().sum())

print("Duplicated Values: \n", df.duplicated().sum())

print(df[df.duplicated()].head(15))
df.drop_duplicates(inplace=True)

df.reset_index(inplace=True, drop=True)
print(df.head())

# Age Column
mean_of_age = df["Age"].mean()
median_of_age = df["Age"].median()

fig = px.box(
    y = df["Age"],
    title = "Ages Distribution",
    template = "plotly_dark",
    labels = {"y", "Age"}
)
custom_layout()
# iplot(fig)

fig = px.histogram(
    df["Age"],
    nbins=25,
    title="Age Distribution",
    template="plotly_dark",
    labels = {"value": "Age"}
)

custom_layout()
fig.update_traces(
    textfont= {
        "size": 20,
        "family": "tahoma",
        "color": "#fff"
    },
    hovertemplate = "Age: %{x}<br>Frequency: %{y}",
    marker = dict(line=dict(color="#000", width=0.1))
)

# Adding mean line
add_line(x0=median_of_age, y0=0, x1=median_of_age, y1=30+2, line_color='#FFE5F1',
         font_color="#fff", xposition="right", text="Median")

# iplot(fig)

# Gender Column
gender = df["Gender"].value_counts(normalize=1) * 100
gender.apply(lambda x:f"{x:0.2f}%")

fig = px.bar(data_frame = gender,
             x = gender.index,
             y = gender,
             color = gender.index,
             title = "Gender Frequency (PCT)",
             color_discrete_sequence=["#45FFCA", "#FF9B9B"],
             labels= {"index" :"Gender", "y": "Frequency in PCT(%)"},
             template="plotly_dark",
             text = gender.apply(lambda x: f"{x:0.0f}%"))

custom_layout()


fig.update_traces(
    textfont = {
        "size" : 16,
        "family" :"arial",
        "color": "#222"
    },
    hovertemplate = "Gender: %{x}<br>Percentage: %{y:0.1f}%",
)

# iplot(fig)

# Education Column

education = df["Education Level"].value_counts(normalize=1) * 100
education.apply(lambda x: f"{x:0.2f}%")

fig = px.bar(data_frame = education,
             x = education.index,
             y = education,
             color = education.index,
             title = "Education Frequency (PCT)",
             color_discrete_sequence=["#45FFCA", "#D09CFA", "#FF9B9B"],
             labels= {"index" :"Education", "y": "Frequency in PCT(%)"},
             template="plotly_dark",
             text = education.apply(lambda x: f"{x:0.0f}%"))

custom_layout()


fig.update_traces(
    textfont = {
        "size" : 16,
        "family" :"arial",
        "color": "#222"
    },
    hovertemplate = "Education: %{x}<br>Percentage: %{y:0.1f}%",
)

# iplot(fig)

# Experience Column
fig = px.box(
    y=df["Years of Experience"], 
    title= "Experience Years Distribution",
    template="plotly_dark",
    labels={"y" :"EXP Years"},
)
custom_layout()

# iplot(fig)

# Salary Column
fig = px.box(
    x = df["Education Level"], y = df["Salary"],
    title= "Salary Vs. Education Level",
    template="plotly_dark",
    labels={"x": "Education Level", "y" :"Salary"}
)

custom_layout(hover_font_size=13)

# iplot(fig)


# What is the average salary of each gender??
salary_by_gender = df.groupby("Gender")["Salary"].mean().sort_values(ascending=False)
salary_by_gender.apply(lambda x: f"${x:,.2f}")
print(salary_by_gender)

fig = px.bar(data_frame = salary_by_gender,
             x = salary_by_gender.index,
             y = salary_by_gender,
             color = salary_by_gender.index,
             title = "AVG Salary By Gender",
             color_discrete_sequence=["#45FFCA", "#D09CFA", "#FF9B9B"],
             labels= {"index" :"Education", "y": "Frequency in PCT(%)"},
             template="plotly_dark",
             text_auto = "0.4s" 
            )

custom_layout()


fig.update_traces(
    textfont = {
        "size" : 16,
        "family" :"arial",
        "color": "#222"
    },
    hovertemplate = "Gender: %{x}<br>Average Salary: $%{y:0.4s}",
)

# iplot(fig)


# Salary based on education level??
salary_by_education = df.groupby("Education Level")["Salary"].mean().sort_values(ascending=False)
salary_by_education.apply(lambda x: f"${x:,.2f}")
print(salary_by_education)

fig = px.bar(data_frame = salary_by_education,
             x = salary_by_education.index,
             y = salary_by_education,
             color = salary_by_education.index,
             title = "AVG Salary Via Education Level",
             color_discrete_sequence=["#45FFCA", "#D09CFA", "#FF9B9B"],
             labels= {"index" :"Education", "y": "Frequency in PCT(%)"},
             template="plotly_dark",
             text_auto = "0.4s" 
            )

custom_layout()


fig.update_traces(
    textfont = {
        "size" : 16,
        "family" :"arial",
        "color": "#222"
    },
    hovertemplate = "Education Level: %{x}<br>Average Salary: $%{y:0.4s}",
)

# iplot(fig)

# How does years of experience influence salary??
def grouping_exp(exp):
    if exp >=0 and exp <=5:
        return "0-5 years"
    elif exp >5 and exp <= 10:
        return "5-10 years"
    elif exp >10 and exp <= 15:
        return "10-15 years"
    elif exp >15 and exp <= 20:
        return "15-20 years"
    elif exp > 20:
        return "20+ years"
    
salary_by_exp = df.groupby(df["Years of Experience"].apply(grouping_exp))["Salary"].mean().sort_values(ascending=False)
salary_by_exp.apply(lambda x: f"${x:,.2f}")

fig = px.bar(data_frame = salary_by_exp,
             x = salary_by_exp.index,
             y = salary_by_exp,
             color = salary_by_exp.index,
             title = "AVG Salary By Experience",
             color_discrete_sequence=["#45FFCA", "#D09CFA", "#FF9B9B", "#F875AA", "#3EDBF0"],
             labels= {"index" :"Education", "y": "Frequency in PCT(%)"},
             template="plotly_dark",
             text_auto = "0.4s" 
            )

custom_layout()


fig.update_traces(
    textfont = {
        "size" : 16,
        "family" :"arial",
        "color": "#222"
    },
    hovertemplate = "Gender: %{x}<br>Average Salary: $%{y:0.4s}",
)

# iplot(fig)


# Correlation
correlation = df.corr(numeric_only=True)

fig = px.imshow(
    correlation,
    template = "plotly_dark",
    text_auto = "0.2f",
    aspect = 1,
    color_continuous_scale = "orrd",
    title = "Correlations Between Data"
)

fig.update_layout(
    title = {
        "font": {
            "size": 28,
            "family": "tahoma"
        }
    }
)
# iplot(fig)



fig = px.scatter_matrix(
    df,
    dimensions=df.select_dtypes(include="number").columns,
    height=800,
    color="Salary",
    opacity=0.65,
    title= "Relationships Between Numerical Data",
    template="plotly_dark"
    
)

fig.update_layout(
    title = {
        "font" :{
            "size" : 28,
            "family" : "tahoma"
        }
    }
)
# iplot(fig)

# Building the model

# Encoding Categorical Data: Converting Categorical into Numerical
df_encoded = pd.get_dummies(df, columns=["Education Level"], drop_first=True)*1
print(df_encoded.head(5))

# Selecting the features
X = df_encoded.drop(columns=["Job Title", "Salary", "Gender"])
y = df_encoded["Salary"]

print(X.head())
print(y.head())

# Splitting our data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=90)

# Cross Validation Score
kf = KFold(n_splits=10, shuffle=True, random_state=30)
rf = RandomForestRegressor(n_estimators=500, random_state=11)
scores = cross_val_score(rf, X, y, cv=kf)
print(f"Cross Validation Score: {np.mean(scores)*100:0.2f}")

# Fitting the Model
rf.fit(X_train, y_train)

score = rf.score(X_train, y_train)*100
print(f"Model Score: {np.round(score, 2)}%")

# Prediction
predicted_salary = np.round(rf.predict(X_test))

d = {
    "Actual_Salary": y_test,
    "Predicted_Salary": predicted_salary,
    "error": predicted_salary - y_test
}
predected_df = pd.DataFrame(d)
print(predected_df.head())

score = r2_score(y_test, predicted_salary)*100
print(f"Model Score: {np.round(score, 2)}%")

rmse = np.sqrt(mean_squared_error(y_test, predicted_salary))
print(f"Error Ratio: {rmse:.3f}")


fig = px.scatter(
    predected_df, 
    x = "Actual_Salary", 
    y = "Predicted_Salary",
    color = "error",
    opacity=0.8,
    title= "Predicted Vs. Actual",
    template="plotly_dark",
    trendline="ols"
    
)

fig.update_layout(
    title = {
        "font" :{
            "size" : 28,
            "family" : "tahoma"
        }
    }
)
iplot(fig)

