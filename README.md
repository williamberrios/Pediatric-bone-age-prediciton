# Pediatric Bone Age Regression

+ The objective of this github is to present a method to create a model that predict the Bone Age of children from a Hand X-Ray.
+ The training data comes from the RSNA 2017 comp. ,but the test dataset is from a real brazilian hospital provided by [DATA H](http://www.datah.ai/)
![Alt Text](TestImage.png =250x)

+ The training dataset contains only images with the left hand (RSNA 2017 competition) but the test dataset contains some images with both hands from a hospital in Brazil. It means that, even though the model will be trained in a different distribution and only with images containing a single hand, it should work with images containing both hands.
+ Both datasets (training and test) have a tabular variable which is gender

Besides, a web application was created in order to use the model in a server using Flask.
![Alt Text](WebAplication.gif)





