# Rock Paper Scissors

---

Rock Paper Scissors is a hand game usually played between two people,
in which each player simultaneously forms one of three shapes with an outstretched hand. 
These shapes are "rock" (a closed fist), "paper" (a flat hand), and "scissors"
(a fist with the index finger and middle finger extended, forming a V). 
"Scissors" is identical to the two-fingered V sign (also indicating "victory" or "peace") 
except that it is pointed horizontally instead of being held upright in the air.

---

In this project I have mostly used *openCV* to get camera video and prepare the photos to predict.
Model was created with *tensorflow's keras*.

---

To create your own dataset you need to run **createData.py**, then take photos of your hand showing
rock, paper and scissors and also your background (to predict that you are showing nothing).
In my X.pickle file I saved about 4800 photos (each form of hand had 1200 photos), my background was a gray blanket.
Remember to press 's' button to save your data!
The file **trainModel.py** creates a model which will be used to predict what you are showing on camera based on
dataset that you have created.
**RockPaperScissors.py** is the last file which allows you to test your model and play vs computer that is randomly
selecting it's "hand shape".

I made this project to learn how to use *tensorflow* and *openCV* and train my skills in machine learning.

__Some screens to show how it's working:__

![alt text](https://raw.githubusercontent.com/thepyrkowaty/RockPaperScissors/main/readmeImages/1.png)


![alt text](https://raw.githubusercontent.com/thepyrkowaty/RockPaperScissors/main/readmeImages/2.png)


![alt text](https://raw.githubusercontent.com/thepyrkowaty/RockPaperScissors/main/readmeImages/3.png)


![alt text](https://raw.githubusercontent.com/thepyrkowaty/RockPaperScissors/main/readmeImages/4.png)


![alt text](https://raw.githubusercontent.com/thepyrkowaty/RockPaperScissors/main/readmeImages/5.png)


<sub><sup>Hand on photos is not mine haha!!!</sup></sub>