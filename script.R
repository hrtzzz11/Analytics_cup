#set your working path
setwd("/Users/tomasbarbosa/Desktop/ML/analytics cup/training_data")
library(tidyverse)
library(dplyr)
library(caret)

set.seed(2022)

#read all the data
classification = read.csv("classification.csv")
customers = read.csv("customers.csv")
sales_orders_header = read.csv("sales_orders_header.csv")
sales_orders = read.csv("sales_orders.csv")
service_map = read.csv("service_map.csv")
random = read.csv("submission_random.csv")
business_units  = read.csv("business_units.csv")

#merge the Sales_order with Sales_order header to get more information about the orders
#note that this causes the creation of "net value.x" and "net value.y" since both of these tables have
#an attribute "net value" with different meaninga
merged = inner_join(sales_orders,sales_orders_header,
                    by = c('Sales_Order' = 'Sales_Order'))


#merge customer with classification table and remove the unclassified instances,
#since we want to do this to build the model and those observations are useless
customers_classified = inner_join(customers,classification,
                                  by = c('Customer_ID' = 'Customer_ID'))
customers_classified <- customers_classified %>% 
  filter(!is.na(Reseller))

#set the item_positions of those intances where there is no matching pair to 0
customers_classified$Item_Position <- ifelse(!(customers_classified$Sales_Order %in% merged$Sales_Order &
        customers_classified$Item_Position %in% merged$Item_Position), 0, customers_classified$Item_Position)

merged$Item_Position <- ifelse(!(merged$Sales_Order %in% customers_classified$Sales_Order &
              merged$Item_Position %in% customers_classified$Item_Position), 0, merged$Item_Position)

#merging the 2 previously merged tables by both sales order and item position
customer_orders <- inner_join(customers_classified, merged, by=c("Sales_Order", "Item_Position"))


#add info about the business centers (not sure if this table is actually relevant)
almost_complete = inner_join(customer_orders, business_units, by="Cost_Center")
#adding information about if a certain order is a service or not
almost_complete$is_service <- ifelse(almost_complete$Material_Class %in% service_map$MATKL_service, 1, 0)
df <- almost_complete

#adjust variable types
df <- df %>% mutate(Type = as.factor(Type))
df <- df %>% mutate(is_service = as.factor(is_service))
df <- df %>% mutate(Reseller = as.factor(Reseller))
df <- df %>% mutate(Business_Unit = as.factor(Business_Unit))
df <- df %>% mutate(Delivery = as.factor(Delivery))
df <- df %>% mutate(YHKOKRS = as.factor(YHKOKRS)) 
df <- df %>% mutate(Sales_Organization = as.factor(Sales_Organization)) 
df <- df %>% mutate(Cost_Center = as.factor(Cost_Center)) 
df <- df %>% mutate(Material_Class = as.factor(Material_Class)) 

df <- df %>% mutate(Document_Type = as.factor(Document_Type)) 
#YHKOKRS is a factorial with only 1 level, so we have to remove it from the table
#most of the rest of the stuff we remove since it's not relevant for the predictions
df <- df %>% select(-YHKOKRS)
df <- df %>% select(-Test_set_id)
df <- df %>% select(-Sales_Order)
df <- df %>% select(-Item_Position)
df <- df %>% select(-Customer_ID)
df <- df %>% select(-Material_Code)
df <- df %>% select(-Creation_Date)
df <- df %>% select(-Release_Date)
df <- df %>% select(-Delivery)
df <- df %>% select(-Creator)
df <- df %>% select(-Material_Class)
df <- df %>% select(-Cost_Center)



#df <- df %>% mutate(Creation_Date = as.Date(Creation_Date, format = "YYYY-MM-DDTHH:MM:SSZ"))
#df <- df %>% mutate(Release_Date = as.Date(Release_Date, format = "YYYY-MM-DD HH:MM:SS.SSS"))
#df <- df %>% mutate(Delivery = as.Date(Delivery, format = "YYYY-MM-DD HH:MM:SS.SSS"))

#variable types have been adjusted
#maybe we want to remove some non Resellers 
#from the data in order to balance out our specificity and sensitivity
#since otherwise our balanced accuracy sucks
train_ind <- sample(1:nrow(df), 0.7*nrow(df))
train <- df[train_ind, ]
test <- df[-train_ind, ]


# Fit the logistic regression model on the training set
model <- glm(Reseller ~ ., data=train, family=binomial)

# Make predictions on the test set
predictions <- predict(model, test, type="response")

# Convert the predictions to a binary outcome (0 or 1)
predictions <- ifelse(predictions > 0.5, 1, 0)


error_rate <- mean(test$Reseller != predictions)
predictions <- as.factor(predictions)
actual <- as.factor(test$Reseller)
confusion_matrix <- confusionMatrix(predictions, actual)


error_rate
confusion_matrix
