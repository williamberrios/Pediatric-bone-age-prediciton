# Pediatric Bone Age Regression

+ The objective is to create a deep learning model that predicts the Bone Age (months) of children from a Hand X-Ray . The training data comes from the competition of RSNA 2017 ,but the test dataset is from a real brazilian hospital provided by [DATA H](http://www.datah.ai/)

+ Both datasets (training and test) have a tabular variable which is gender so an architecture combining the tabular and image data was created and it gave better results


<p align="center">
  <img  src=Files/Architecture.png>
</p>

+ The training dataset contains only images with the left hand (RSNA 2017 competition) but the test dataset contains some images with both hands from a hospital in Brazil. It means that, even though the model will be trained in a different distribution and only with images containing a single hand, a preprocess step should be applied to the test dataset.

<p align="center">
  <img width="400" height="300" src=Files/TestImage.png>
</p>


Download Dataset: [Pediatric Bone Dataset](https://www.kaggle.com/t/660042d6e9134e368f567c6ab02ada19)

# Web Aplication
In addition a web application was created using Flask and python.

Run :

```bash
python api.py
```

![Alt Text](Files/WebAplication.gif)





