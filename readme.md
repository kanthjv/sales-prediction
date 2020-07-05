1.	importing training data set
2.	returns the shape of training data set
3.	importing test data set
4.	returns total number of null values in each column of train data set
5.	plotting correlation matrix of test dataset and train dataset
DATA PREPROCESSING
The main aim of this step is to standardize the data which means removing all the empty values and zero values and filling with respective values with mean or mode of respective column
6.	find the total number of each item in Item_Fat_Content column
7.	find the total number of each item in Item_Type column
8.	find the total number of each item in Outlet_Identifier column
9.	find the total number of each item in Outlet_Size column
10.	find the total number of each item in Outlet_Type column
11.	appending test dataset to train dataset
12.	find the shape of dataset
13.	find the total number of null values in each column of dataset
14.	find total number of each item in every column of dataset
15.	find the datatype of each column
16.	find the label of each column of dataset
17.	filtering out categorical columns 
18.	find the frequencies of these categories
Three columns of data contain empty cells. They are” Item_Weight”, “Outlet_Size” and “Item_Outlet_Sales”.
19.	So, initially replace all the zero values of the column with mean of the respective column.
Then, fill the empty values of ‘Outlet_Size’ column with the mode of respective column
20.	import mode function
21.	mapping the items of Outlet_Size column
22.	creating a table with items of "Outlet_Type" as columns and 'Outlet_Size' as row and mode of each
23.	returns "True" if the value of 'Outlet_Size' column is null
24.	filling empty values of each column with mode of respective type


25.	find the total number of empty cells in each column
26.	print items of every column whose count is <30
27.	creating a table with 'Outlet_Type' as column and 'Item_Outlet_Sales' as row
28.	finding the mean of 'Item_Visibility' column
29.	finding the location of 0 values of selected column and filling such values with mean of respected column
30.	creating new column using Item_Type_Combined by extracting frst two letters of "Item_Identifier"
31.	verifying count of each item_type_combined
32.	updating Item_Type_Combined
33.	creating new coulmn with how many years it passed since 2013
34.	updating "Item_Fat_Content" column
Prediction Step
Finally, uses three different algorithms for sales prediction. They are 
linear regression
ridge
decision tree regression and their respective scores are
Linear Regression
1202.1221141316933
CV_SCORE : mean - 1203 | std - 41.73 | max - 1284 | min – 1151

ridge
1058.9440596918816
CV_SCORE : mean - 1093 | std - 45.53 | max - 1184 | min – 1011

Decision Tree regression
1070.1302266626526
CV_SCORE : mean - 1097 | std - 43.03 | max - 1174 | min - 1027

