# Sentiment-Analysis-in-Recommendation-Systems

### Matrix Factorization + Incorporating Text
Matrix factorization takes advantage of the ratings that users
give items in order to make predictions. Expansions have
been made to solve some of the longstanding problems with
this technique of collaborative filtering, such as sparsity.
This makes it more difficult to find similar users to compare against. 
This problem is exacerbated for new users/users with very little reviews.
More commonly, models introduce more information into
the system to draw more latent variables. Reviews have been a
primary example of this, as they provide a rich source of text
users provide justifying their rating.
Another factor often overlooked in collaborative filtering
methods and offer additional information to better define user
preferences are item descriptions. Item descriptions can help
find hidden patterns within user behavior that are not expressed
within user reviews. For example, if a user likes an item that
falls under a particular brand, they might not write the name
of the brand in their review. However, by examining the item
descriptions, it is possible to notice this pattern.
In our research, we present the following:
1. An analysis on various inclusions of LDA within a MF-
based model.
2) A comparison of different scopes of databases to under-
stand categorization better

### LDA Explanation
Latent Dirichlet Allocation (LDA) is a probabilistic
generative model for topic modeling, and is designed to
discover topics in a collection of documents and understand
how words are distributed across those topics. LDA is a
bayesian version of PLSA and uses the dirichlet distribution,
which is a distribution over distributions. LDA assumes that
each document is a mixture of various topics, and each topic
is a distribution of words. LDA assigns words to topics
probabilities, and documents to topics based on the frequency
of these topics in the document. The main idea is that the
LDA is that the model infers the latent topics and their
word distributions that are most likely to have generated the
observed documents in the document set. LDA is preferred
over PLSA as it can generalize to new document easily, while
in PLSA document probability is fixed.

One of our proposed directions we intended to utilize the
LDA to initialize the values of the user and item embedding’s
layer. The way this was done was to train the LDA on
a corpus of text and to extract the user and item topics
distribution. Afterwards we mapped our user-id’s and item-id’s
from the raw data-set containing hashed strings from Amazon
to integers which would prove to be useful when initializing
the weights for each row in the embedding’s layer. The final
adjustment before running our hypothesis model against the
baseline is to create another mapping by adjusting the topic
distribution of the users and items. Since by default Torch’s
embedding’s layer initialized the weights randomly from a
range from -1 to 1, we mapped the topic distribution for each
item and user such that the largest value is equal to 1, smallest
value is equal to -1, and all values in between are mapped
accordingly

### Topic Distribution Running LDA on Amazon Reviews
After running the LDA on reviews based on the user’s
account we see a glimpse into the online behavior of internet
users on the Amazon platform.
- Topic 0: Games

   **game**, **app**, **fun**, **play**, love, like, **kindle**, games, great, really, time, just, good, free, **playing**, don, easy, **graphics**, recommend, enjoy, awesome, use, think, challenging, way, lot, **played**, **levels**, want, quot.

- Topic 1: Product Reviews

   **product**, use, **good**, **great**, **like**, **used**, just, **using**, ve, **works**, **price**, **work**, does, really, don, 34, **better**, day, taking, **feel**, time, **taste**, **razor**, years, clean, **bought**, water, little, tried, did.

- Topic 3: Toys

   **toy**, old, **loves**, **years**, **son**, **daughter**, **kids**, **little**, set, **christmas**, great, bought, **toys**, **play**, **doll**, **grandson**, pieces, **gift**, cute, fun, **child**, 34, love, got, **loved**, just, **birthday**, **granddaughter**, really, **quality**.

Additionally we ran another LDA on review based on each
item and this has given us in insight on the groupings of what
reviews are like based on each item.
- Topic 0: Gaming Apps

   **game**, **app**, **fun**, **play**, **games**, kindle, like, love, free, just, time, really, **playing**, good, great, **graphics**, don, **puzzles**, **played**, **puzzle**, quot, **apps**, **easy**, **challenging**, version, screen, enjoy, **levels**, cards, **download**.
- Topic 1: Toys

   **product**, use, like, great, just, **toy**, good, **little**, **old**, really, year, 34, time, loves, used, set, price, don, bought, love, work, works, **son**, ve, day, using, **daughter**, does, **kids**, better.
- Topic 4: Health and Wellness

   potholders, **tecnu**, **omegavia**, **omapure**, **mgsdha**,
   **mgsepa**, **mgspercentage**, urinals, mgstotal, **innovix**,
   formnot, slingo, 1other, **vegetarianifos**, clarinet,
   **mercurymolecularly**, **ifos**, **nclex**, **chaga**, minami,
   potholder, liana, moen, **gmoscontains**, babbel, dis-
   tillednot, ourworld, **hydraplenish**, rosalie, twista-bles

As you can see the distinct topics showcase above
demonstrate a diverse range of user interests and engagement
patterns on the Amazon platform, spanning across gaming,
general products, toys, and health-related categories.

### Model Results
![Percent Comparison](/findings/Compare%20percentage%20with%20baseline.png)
![Percent Comparison with Baseline](/findings/Compare%20with%20baseline.png)

As you can see with the images above, the LDA did show improvement to the the model in comparison to the baseline. But unfortunately this improvement was under the 5% threshold to be significant improvement. We will be looking back at
our code to reexamine the MF-LDA archeture to see what change can lead to better results.


## Important Files
- [Final Presentation](/findings/Final%20Presentation.pdf)
- [Final Report](/findings/Final%20Project%20Report.pdf)
- [Running LDA Model](/src/src/MF_LDA_Preprocess.ipynb)


## Collaborators
[<img src="https://avatars.githubusercontent.com/NicoPang" alt="Nick Pang" width="50" height="50">](https://github.com/NicoPang)

<small>Nick Pang</small>

## **[Amazon Product Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html)**
The dataset we have chosen is Amazon's 2014 review dataset hosted by the University of California San Diego. The Amazon datasets have been preprocessed to remove any duplicate items as well as removing all users and items with less than 5 reviews associated with them. We chose these 3 subsets of the Amazon dataset in order to observe how the model performs with a variety of both similar *(Toys vs Android)* and different *(Toys/Android vs Health)* categories.

1. **Toys & Games**
   - *[5-Core Data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games_5.json.gz)*
   - *[All Data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Toys_and_Games.json.gz)*

2. **Apps for Android**
   - *[5-Core Data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Apps_for_Android_5.json.gz)*
   - *[All Data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Apps_for_Android.json.gz)*

3. **Health & Personal Care**
   - *[5-Core Data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Health_and_Personal_Care_5.json.gz)*
   - *[All Data](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Health_and_Personal_Care.json.gz)*

## Dependencies
The code was run using Python *3.10.9*. All python dependencies are stored in **dependencies.txt**, which can be installed using the following command.

```
python3 -m pip install -r dependencies.txt
```