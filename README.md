# Analyzing sentiments from social media

A Machine learning project for predicting the sentiment of a tweet. Under the guidance of Dr. Jainendra Shukla
<p align="center">
  <img src="https://media.giphy.com/media/lMdzpEp18hMd2/giphy.gif" />
</p>
Expressing sentiments on social media has become an important way for people to express their opinions. With the increase in use of social media, people are able to access information much more easily. But with information, also comes opinions of other individuals, including the positive and negative perspective on various topics. With the differences in opinions and biases in viewpoints, these discussions can easily grow into bullying, hate comments, and harassment. It is important to maintain a positive environment so that its safe for everyone to communicate, discuss, debate and give opinions on social media, without being worried any backlash they might get. Social media is the ideal medium to observe and study the expression of emotions, and in turn, study the sentimental values each expression holds.

# Credits to dataset
The dataset titled "Sentiment140 dataset with 1.6 million tweets" on Kaggle has about 1.6 million tweets which have been extracted using the twitter API. The tweets have been annotated with a positive and negative label that can be used to detect sentiment. You can find dataset [here](https://www.kaggle.com/kazanova/sentiment140). Achnoledgement to stanford to conduct such research ([reserch paper](https://www-cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf)) and making dataset public.

# Demo-Preview
[src directory](https://github.com/AnikaitSahota/ML-project/tree/main/src) contains all the code used in the project. [preprocessing.py](https://github.com/AnikaitSahota/ML-project/tree/main/src/preprocessing.py) is used for all kind of preprocessing over the provided dataset. 
For testing the best model use [runner.py](https://github.com/AnikaitSahota/ML-project/tree/main/src/runner.py). It uses all the all other file present in src to show the perforamance of the best model and its metrics.

![Random GIF](https://media.giphy.com/media/xUNen15tsNHWnIvY4M/giphy.gif)

    cd src
    python runner.py


# Table of contents

After you have introduced your project, it is a good idea to add a **Table of contents** or **TOC** as **cool** people say it. This would make it easier for people to navigate through your README and find exactly what they are looking for.

Here is a sample TOC(*wow! such cool!*) that is actually the TOC for this README.

- [Project Title](#project-title)
- [Demo-Preview](#demo-preview)
- [Table of contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Authors](#authors)
- [License](#license)
- [Footer](#footer)

# Installation

To use this project, first clone this git repository using following commands

    git init
    git clone https://github.com/AnikaitSahota/ML-project.git

Then comes the, installation of dependences

I have use following dependences
- numpy
- pandas
- matplotlib
- sklearn
- pytorch
- xgboost
- joblib
- re

for installing all the dependencies, use requirement file with following command

    pip3 install -r requirment.txt
<!-- # TODO : need to define requiremtn.txt -->
# Usage

There are various models implemented in the [modeling.py](https://github.com/AnikaitSahota/ML-project/tree/main/src/modeling.py) file. You can use any of the defined model from its pickle file (stored at [src/models](https://github.com/AnikaitSahota/ML-project/tree/main/src/models)) for retriving the results.

# Authors

- [AnikaitSahota](https://github.com/AnikaitSahota)

- [Kartikeya Gupta](https://github.com/Kartikeya18153)

See also the list of [contributors](https://github.com/AnikaitSahota/ML-project/contributors) who participated in this project.

# License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

# Footer
- If you want to contribute, fork the repository for yourself. Check out [Fork list](https://github.com/AnikaitSahota/ML-project/network/members)
- If you liked the project, leave a star. Check out [stargazers](https://github.com/AnikaitSahota/ML-project/stargazers)

