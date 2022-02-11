# HR Analytics: Employee Promotion
by Charlie Team from Purwadhika Startup and Coding School
- Jeremy Bona Carlo
- Stanley Martin
- Rifa Faisyah

This Repository Contains:
- [Dataset](https://github.com/PurwadhikaDev/CharlieTeam_JC_DS_FT_BSD_JKT_14_FinalProject/blob/main/train_emp_prom.csv)
- [Source Code](https://github.com/PurwadhikaDev/CharlieTeam_JC_DS_FT_BSD_JKT_14_FinalProject/blob/main/Charlie_Team_HR_Analytics_Employee_Promotion.ipynb)

Source of data : https://www.kaggle.com/arashnic/hr-ana

# Define Problem
- Produce a predictive machine learning program that can predict whether a user is going to be promoted or not considering the features have by this Multinational Company, therefore it can ease up the HR Department to faster the determination time; an automation can also reduce human error in manual determination
- Although this model is purposed to determine whether an employee is to be promoted or not, we cannot conclude the performance of this employee is good or bad generally
- Machine Learning that is going to be used for this project is supervised learning, classification
- Business wise, performance metrics is recall; we want to promote people that really deserve to be promoted, because it is rather a detrimental issue for the company not to promote the deserving ones, and to be more convincing, we would like to plot a cost loss scenario as to what this company would loss if they were to apply this model to the field.
- The risk that is going to be faced by this MNC is that although the performance is acceptable, there always an external issue that can decrease this model performance in the long run, therefore, it is better to watch out for this 'external issue' such as the dynamics of the salary also can be a potential issue considering that a stagnant in a salary in a certain period can decrease one's performance in the company.

# Data Preparation
We import the dataset, EDA, choosing relevant features to be used, impute missing value and binning

# Modeling
We use Logistic Regression model since it has a wider range between the training's performance and the test's performance. We tuned hyperparameter for them using GridSearchCV. The following below are the parameters:
**SMOTENC**
For the imbalance issue in our data, we manage to handle them with SMOTENC method that works well with categorical data. Below is the parameter for the SMOTENC method:
```
smotenc = SMOTENC(categorical_features=[0,1,2,3,6], random_state=22, sampling_strategy=4/5)

X_train_smotenc, y_train_smotenc = smotenc.fit_resample(X_train, y_train)
```
Below is also the parameter that we use to be tuned before and after we do random oversampling using SMOTENC
**_Parameter_**
```
weights = np.linspace(0.05, 0.95, 21)

param_grid = {
    'log_reg__penalty':['l1', 'l2', 'none'],
    'log_reg__fit_intercept':[True, False],
    'log_reg__C':[0.01, 0.1, 1.0],
    'log_reg__class_weight':[{0: x, 1: 1.0-x} for x in weights]
}
```

# Result
![download](https://user-images.githubusercontent.com/87014423/143854789-2e505435-414d-4cce-bd7c-9b63032c7fd9.png)
- After having a thorough look of this project, we would like to conclude that this project is an eye-opener to the industry life condition. There are many imbalances between the promoted and not promoted in a company, preferably to the not promoted, and also the prediction model can be evaluated by a probable cost loss for the company, and determined to be used.
- The model coefficients however, gives a positive insight for the HR to predict whether an employee is to be promoted or not, and to give a promotion more wisely.
- This model's coefficient to their classification has a high alignment to us author's domain knowledges. Perhaps when determining a promotion candidate, the HR Department can further use this model's consideration, considering it's recall performance. Further works such as deep learning can also boost this prediction project performance, can also speed up automation of promotion determination in this company.
- For further business use, we also include a scenario containing employees data that have a high probability chance to be promoted in this data, giving HR as the business user to evaluate the data, and take action such as giving the high probability employees to be promoted to go straight to HR interview, managing the company safe some few bucks of their funds.
- A more thorough feature engineering can also be done to enhance the model's performance. Current features such as region is impactful to the model but also not clear enough whether the region has another feature to explain their impact to the model.
