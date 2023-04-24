# Blog Recommendation System: BlogRadar

A project by COMP-313 Software Project Sec 002 Team #4

This is a content-based blog recommendation where user can find relevant blogs using a query. The system finds relevant blog titles from the dataset related to the search query. This makes use of machine learning techniques to classify the input search query to one of the predefined categories (multiclass classification) and run a similariy search using cosine similarity on the subset identified by the classified category.

Techniques used:

- We created a pipeline to train a number of classifiers on our dataset, and we chose the classifier that gave the best accuracy for the project. Since this is a multiclass classification problem, the following classifiers were trained:

            - Logistic Regression
                        - Accuracy:  0.8046272493573264
                        - Precision:  0.8164173560788747
                        - Recall:  0.8046272493573264
                        - F1 Score:  0.8068210090961856
                        
            - RandomForestClassifier
                        - Accuracy:  0.6529562982005142
                        - Precision:  0.7755265603381907
                        - Recall:  0.6529562982005142
                        - F1 Score:  0.6911412310458983

            - SVC
                        - Accuracy:  0.7489288774635818
                        - Precision:  0.7502701606843972
                        - Recall:  0.7489288774635818
                        - F1 Score:  0.7255811760742747
                        
            - DecisionTreeClassifier
                        - Accuracy:  0.3856041131105398
                        - Precision:  0.7811635300192267
                        - Recall:  0.3856041131105398
                        - F1 Score:  0.49748450176490144          - 
            
            - ExtraTreesClassifier
                        - Accuracy:  0.6606683804627249
                        - Precision:  0.7608671573422545
                        - Recall:  0.6606683804627249
                        - F1 Score:  0.6845460784728071

            - MultinomialNB
                        - Accuracy:  0.6041131105398457
                        - Precision:  0.6639214713589054
                        - Recall:  0.6041131105398457
                        - F1 Score:  0.5302988206773825

            
- The best classifier from the above was chosen based on the accuracy - Logistic Regression.
- Once the category was identified by the classifier, we run similarity score of the user input with each of the blogs we have in the dataset tagged with the identified category. This was done using `cosine_similarity`.
- We returned the top 10 similiar blogs to user.
