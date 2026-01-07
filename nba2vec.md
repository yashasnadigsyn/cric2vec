```
Abstract— Understanding a player’s performance in a bas-
ketball game requires an evaluation of the player in the context
of their teammates and the opposing lineup. Here, we present
NBA2Vec, a neural network model based onWord2Vec[1] which
extracts dense feature representations of each player by predict-
ing play outcomes without the use of hand-crafted heuristics or
aggregate statistical measures. Specifically, our model aimed to
predict the outcome of a possession given both the offensive and
defensive players on the court. By training on over 3.5 million
plays involving 1551 distinct players, our model was able to
achieve a 0.3 K-L divergence with respect to the empirical
play-by-play distribution. The resulting embedding space is
consistent with general classifications of player position and
style, and the embedding dimensions correlated at a significant
level with traditional box score metrics. Finally, we demonstrate
that NBA2Vec accurately predicts the outcomes to various
2017 NBA Playoffs series, and shows potential in determining
optimal lineup match-ups. Future applications of NBA2Vec
embeddings to characterize players’ style may revolutionize
predictive models for player acquisition and coaching decisions
that maximize team success.
```
```
I. INTRODUCTION
Successful coaches construct optimal lineups for given
situations in basketball games based on a deep understanding
of each player’s play-style, strengths, and weaknesses in
the context of all other players on the court. Studying the
distribution of contexts and their outcomes in which a player
takes part may provide insights into aspects of player’s
performance and play style that are not otherwise reflected
in traditional basketball statistics. While much of basketball
analytics relies on the use of hand-crafted advanced statis-
tics (e.g. Wins Above Replacement and Offensive/Defensive
rating) and aggregate statistics (e.g. FG%, assists), they tend
to not capture these contextual influences and effects not
present in box scores. Models capable of characterizing
players based on these contextual factors would offer greater
insight into individual player performance and play-style,
and may shed light on how to construct optimal lineups
for specific situations. Constructing such frameworks may
be possible given the wealth of play-by-play game data and
recent advances in machine learning and natural language
processing (NLP) algorithms.
In particular, the problem of generating accurate repre-
sentations of players in sports analytics is analogous to
the problem of word embeddings in NLP. Word embedding
models aim to create real-valued vector embeddings of words
```
(^1) Department of Chemical Engineering, Massachusetts Institute of Tech-
nology, Cambridge, MA, USA.wjguan@mit.edu
(^2) Harvard-MIT Health Sciences and Technology, Cambridge, MA, USA.
njaved@mit.edu
(^3) Department of Physics, Massachusetts Institute of Technology, Cam-
bridge, MA, USA.lup@mit.edu
that encode complex semantic structure.Word2Vecis a class
of word embedding models that extract useful features of
words given the sentences, known as the “context,” in which
the words are commonly used [1]. This allows Word2Vec
to be trained in an unsupervised way on a large corpus of
written works and extract meaningful relationships. Once
trained, the word embeddings can then be applied to a variety
of other tasks as a pretrained initial transformation in manner
analogous to transfer learning.
In the training phase, the way in which the context of
each word is used can be different; in particular, Word2Vec
uses either a continuous bag-of-words (CBOW) model, or a
skip-gram model. The skip-gram method (Figure 1a) takes
the target word as input to a neural network’s hidden layer(s)
and attempts to predict the words that immediately surround
the target word in a given sentence. On the other hand,
the CBOW method (Figure 1b) takes the words around a
target word as input, predicting the target word as output.
The result of training these models is a dense vectorial
word representation that captures the meaning of each word.
For example, Word2Vec finds more similarity between the
words “king” and “queen” than between “king” and “vindica-
tion.” The ability of word embeddings to accurately capture
the relationship and analogies among words is shown by
Word2Vec arithmetic: for instance,Paris−France+Italy
=Rome.
The success of word embeddings in NLP has inspired its
recent application in sports analytics to characterizing batter
and pitcher performance and play style in baseball [2]. In
that study, a neural network was trained using pitcher-batter
matchups as inputs and the outcome of each at-bat (e.g.
single, home run, flyout to right field) as outputs to create
the player embeddings. The author was able to successfully
visualize macroscopic differences in embedding clusters (e.g.
right-handed vs. left-handed vs. switch hitters, power hitters
vs. on-base hitters) and model previously unseen at-bat
matchups, suggesting that the word embedding concept may
be feasible and promising for creating player representations.
In this study, we applied this concept to extract represen-
tations of different NBA players by producing a embeddings
of every player, which we termNBA2Vec. Similar to [2],
the embedding for each player was generated by training
a neural network aimed at predicting the outcome of each
possession given the ten players on the court. Unlike in
[2], which takes advantage of the mostly one-on-one nature
of baseball dynamics, we used all players on the court
to ensure accurate modeling of holistic relationships and
dynamics between players on the same and different teams.
This increased the complexity of the network, and due to the

Fig. 1. (a) Skip-gram word embedding model. The embedding is extracted from the hidden layer after training to a target ofncontext words.vrepresents
the vocabulary size, or length of the word vectors, whilehrepresents the embedding length. (b) CBOW model. A given word’s embedding is computed
by averaging the hidden layer representations of all contexts in which it appears.

n-body nature of the problem, required the network to exhibit
permutation invariance. Unlike previous attempts to generate
NBA player representations [3] using purely high level,
aggregate statistics and hand-picked features (e.g. number
of shots taken in the paint, FG%, FT%, assists, etc.), our
embedding approach learns directly from raw play-by-play
data by automatically generating rich features which account
for the “context” that affects a player’s style and statistics.
The latent features encoded in these player embeddings can
shed light on both the play style and effectiveness of different
types of players, and can be used as inputs for further
downstream processing and prediction tasks.

```
II. METHODS
```
A. Data Sets and Preprocessing

We used play-by-play and players-on-court data provided
by the NBA, which featured over 9 million distinct plays,
with 1551 distinct players taking the court in these plays. To
create the input to the network, we denoted each player with
an index from 0 to 1550. For the outputs to the network,
we needed to encode the possible outcomes of each play.
In order to encourage learning, we only considered key
outcomes, omitting rebounds and defensive plays. We chose
to use 23 distinct outcomes, some examples of which are
included in Table 1. The provided raw play-by-play data was
more specific on outcomes of plays, but we grouped many
of these plays (e.g. “driving layup shot, dunk shot, reverse
dunk shot, hook shot” were all considered “close-up shots”)

```
together for simplicity. This preprocessing resulted in 4.
million plays, of which we used 3.7 million for a training
set and the remainder as a validation set. We used the Pandas
library to preprocess the data [4].
```
```
TABLE 1
TYPES OF PLAY OUTCOMES.
```
```
Play Index Play Outcome
0 Mid-range jump shot made
1 Mid-range jump shot missed
2 Mid-range jump shot made + 1 free throw made
3 Mid-range jump shot made + 1 free throw missed
4 Close-range shot made
5 Close-range shot missed
6 Close-range shot made + 1 free throw made
7 Close-range shot made + 1 free throw missed
8 0/1 FT made
9 1/1 FT made
10 0/2 FT made
11 1/2 FT made
12 2/2 FT made
13 0/3 FT made
14 1/3 FT made
15 2/3 FT made
16 3/3 FT made
17 3PT shot made
18 3PT shot missed
19 3PT shot made + 1 free throw made
20 3PT shot made + 1 free throw missed
21 Turnover
22 Foul
```

B. NBA2Vec: Player Embedding Model

To train informative embeddings for players in the NBA,
we created a neural network architecture that predicts the
distribution of play outcomes given a particular offensive and
defensive lineup (Figure 2). For each play, we first embed
the 10 players on the court using an 8 dimensional shared
player embedding. We then separately average the 5 offensive
and 5 defensive player embedding vectors. These two mean
player embeddings (i.e. an offensive and a defensive lineup
embedding) are concatenated and fed through one additional
hidden layer of size 128 with a ReLU activation before
outputting 23 outcome scores. Applying a softmax activation
to the scores produces probabilities that we interpret as the
distribution of play outcomes (Table 1). The entire network
is trained end-to-end with a cross entropy loss function
that stochastically minimizes the K–L divergence between
the true play outcomes from the data and the predicted
distribution from the model. This model was built and trained
using the PyTorch framework [5].

C. Validation and Post-processing

1) Validation of NBA2Vec: To evaluate the efficacy of
the NBA2Vec model used to generate the embeddings, we
characterized the difference between the predicted and em-
pirical distributions of play outcomes. The Kullback–Leibler
(K–L) divergence was used as the metric to compare the
distributions. K–L divergence is given by

This measures the number of encoded bits lost when mod-
eling a target distributionp(x)(in this case, the empirical
distribution) with some approximate distributionq(x)(in this
case, our predictive model). Thus, a low K–L divergence
value (DKL≈0) implies a good approximation of the target
distribution, while a larger value (DKL0) implies poor
approximation.
Due to the large number of unique lineup-matchup com-
binations, some of which do not appear enough for a proper
empirical distribution to be generated, we decided to only
look at K–L divergences for lineup-matchup combinations
with more than 15 plays. This analysis was performed on
the last 25 games of the data set (corresponding to the last
25 playoff games in the 2018 NBA playoffs, and 5102 plays).
2) Embedding Analysis: After training the model, we
extracted the shared embedding layer and use dimensionality
reduction, clustering, and visualization methods to explore
the learned player embeddings. In particular, we used t-
SNE—a dimensionality reduction method based on local
neighbor structure—to visualize our 8 dimensional embed-
dings in 2 dimensions [6]. We also used 2D principal com-
ponent analysis (PCA) for dimensionality reduction before
performing k-means clustering. PCA is a statistical method
that uses orthogonal decomposition to transform a set of
observations of correlated variables into a set of observations
of uncorrelated variables, or principal components. Each suc-
cessive principal component explains the maximal variance

```
in the data while remaining orthogonal to the preceding com-
ponent. K-means clustering is a simple clustering method
that aims to partition n observations into k clusters, where
each observation belongs to the cluster with the nearest mean.
Our approaches are further described in section III-A. These
dimensionality and clustering techniques were conducted
using implementations from the Scikit-learn library [7].
3) Exploring Lineup Combinations:To further explore the
macroscopic predictive nature allowed by these embeddings,
we used the neural network model to predict the outcomes
of games based on each team’s most frequent 5-player
lineup. For each pair of teams, the model would output
the distribution of possible play outcomes. We would then
sample these distributions to determine which plays would
occur in a given game, and based on this, predict a game
score. Assuming 100 possessions for each team and that no
substitutions are ever made, we ran the model on various
playoff series match-ups from the 2016-17 season, simulating
1000 best-of-7 series between each pair of teams. Certain
playoff series were not simulated because the most frequent
lineups contained players that were not among the raw data’s
most common 500 players.
III. RESULTS AND DISCUSSION
A. Embedding Analysis
In order to better understand the generated player rep-
resentations, we used t-SNE (described in section II-C.2)
to visualize the 8 dimensional feature vector in 2D. As
depicted in the 2D t-SNE plot (described in section II-C.2)
3, we see that the centers and guards separate nicely, while
forwards (yellow) are scattered throughout both groups. This
is consistent with our intuition about the roles and play-
styles of these classes of players—guards and centers fulfill
very distinct roles while forwards can be more center-like
or guard-like and fulfill multiple roles (i.e. LeBron James).
Despite t-SNE’s utility in preserving high dimensional local
structure, it is not effective at preserving the global structure
of data, which along with the high dimensionality of our
player representations, may help explain why it is difficult
to visualize clear separation between different groups of
players.
To further characterize the learned embeddings, we k-
means clustered 8 scaled and centered embedded dimensions.
We decided to select 3 clusters for our initial analysis after
observing a decreased rate of decline in the variance as a
function of the number of clusters, as seen in the elbow
plot in Figure 4. In Figure 5 we see that k-means yields
3 distinct clusters, which roughly group players with similar
roles/playing styles. For example, the yellow cluster seems to
correspond to guards, the green cluster with centers/forwards,
and the red cluster with forwards. Note that because of the
high dimensional nature of our player representations and
the fact that we are projecting players onto two dimensions,
the euclidean distance between points on the shown plot is
not entirely representative of player similarity.
The observed clusters also seem to suggest that as opposed
to fitting neatly within the traditional 5 positions of basket-
```

Fig. 2. NBA2Vec model architecture. We havev=1551 players, which are mapped toh=8 dimensional embeddings. After then=5 offensive and
defensive player embeddings are separately averaged and then concatenated, they are fed through ani=128 hidden layer with a ReLU activation. The
final output layer with a softmax activation predicts a probability distribution overo=23 play outcomes.


ball, players actually perform diverse roles that would place
them in multiple categories. As seen in figure 5, each of the
clusters comprises multiple groups of players—clusters 1 and
2 comprise positions 1-5, while cluster 2 corresponds mostly
to centers/forwards. Roughly, this may reflect that successful
players are rarely one trick ponies—centers must be able to
shoot, and point guards must be able to score. In general, we
can observe that the learned embeddings roughly correspond
to general basketball intuition. However, the embeddings are
also capturing player characteristics that may not be entirely
reflected in traditional metrics such as box score.
Exploring the structure of the embedded space by calcu-
lating the nearest neighbors by distance for various play-
ers further validates the learned player representations. For
example, Chris Paul, a canonical point guard, has nearest
neighbors including other point guards such as Steve Nash,
Jose Calderon, and Jason Terry. Shaquille O’Neal, a classic
big man, has nearest neighbors including centers such as
Dwight Howard, Roy Hibbert, Tiago Splitter, and Rudy
Gobert.
Next, we calculated Pearson’s correlation between the top
two PCA dimensions and player metrics including minute
adjusted rates for field goals made, three pointers, assists,
rebounds, and plus—minus, as shown in figure 7. Rudimen-
tary analysis revealed that PCA dimension 1 correlated at
a significant level (correctedα= 5 × 10 −^4 ) with rebounds,
assists, and three pointers, while PCA dimension 2 correlated
at a significant level with rebounds and assists (corrected
α= 5 × 10 −^4 ). For both dimensions, the noted correlations
remained significant even with Bonferroni adjustment.
Our exploratory analysis of the embeddings reveals that
the learned player representations encode meaningful infor-
mation that correspond roughly to our intuition about various
players. Through a rudimentary analysis, the embedded
dimensions seem to correspond to a complex combination
of player characteristics as well as real player performance
metrics.

B. Validation of NBA2Vec
Validation of the NBA2Vec network was performed on
plays in the final 25 games of the data set, and a mean K–L
divergence of 0. 301 ± 0 .162 was achieved. Some example
predicted vs. empirical distributions of play outcomes are
shown in Figure 9, showing that the model is able to closely
approximate the target distribution.
We also wanted to determine the minimum number of
plays needed to create an accurate empirical distribution
that can be modeled by the predictive network. Plotting the
K–L divergence vs. number of plays used in the empirical
distribution (Figure 8), we can estimate the minimum number
of plays to reach the minimum K–L divergence to be around
30.
```
```
C. Exploring Lineup Combinations
The results (Table 2) show that even with some crude
assumptions, the winner of any given 7-game series and
the average margin of victory can be approximated using
NBA2Vec embeddings and this neural network model. More
accurate game outcomes would require more precise se-
quence modeling of game-by-game dynamics instead of our
current play-by-play treatment; however, this demonstrates
```
```
Fig. 6. Different types of players comprising each identified cluster in Figure 5.
```
Fig. 7. Correlations and p-values of different metrics with the two top PCA
dimensions for each player. Each raw metric is summed for each player and
normalized by the total number of minutes played. dim1 = PCA dimension
1, dim2 = PCA dimension 2, fg = field goal. Reported p-values are not
Bonferroni corrected but, after correction, remain significant atα= 5 × 10 −^4
with 5 comparisons.

```
Fig. 8. K–L divergence vs. Number of plays used in empirical distribution.
The K–L divergence reaches a minimum plateau after about 30 plays.
```
```
the potential of NBA2Vec embeddings and the play outcome
predictive network.
While serviceable as a predictor of game and series
outcomes, there is also potential use for NBA2Vec as a
lineup optimizer. Given an opposing lineup, this model
facilitates selection of a corresponding optimally matched
lineup for both offense and defense. This optimization can
be accomplished by sampling the model’s predicted dis-
tribution many times for a given pair of lineups. As an
example, we optimize a lineup to face the Golden State
Warriors’ “death lineup” (Stephen Curry, Klay Thompson,
Andre Iguodala, Kevin Durant, and Draymond Green) for the
Houston Rockets, where we fix the first four players (James
Harden, Chris Paul, Eric Gordon, Clint Capela) and vary
the fifth. From this analysis, we can predict the Rockets’
best possible 5th man, and also compare his performance
```

Fig. 9. Empirical vs. predicted distributions for various lineup matchups. The outcome corresponding to each play index can be retrieved from Table 1.

```
TABLE 2
RESULTS OF 1000 SIMULATIONS OF7-GAME SERIES OF VARIOUS 2017 PLAYOFF MATCHUPS VS.GROUND TRUTH.
```
```
Team 1 Team 2 Series Score Margin Team 1 Game Win %
Simulation Cavaliers Warriors 1.69 vs. 4 − 8. 85 29.8%
Truth Cavaliers Warriors 1 vs. 4 − 6. 8 20.0%
Simulation Jazz Clippers 4 vs. 3.77 + 0. 61 51.5%
Truth Jazz Clippers 4 vs. 3 + 1. 14 57.1%
Simulation Rockets Thunder 4 vs. 2.05 + 6. 63 66.1%
Truth Rockets Thunder 4 vs. 1 + 8. 6 80.0%
Simulation Warriors Trailblazers 4 vs. 1.06 + 13. 3 79.0%
Truth Warriors Trailblazers 4 vs. 0 + 18 100%
Simulation Spurs Rockets 4 vs. 3.96 − 0. 04 50.3%
Truth Spurs Rockets 4 vs. 2 + 5 66.7%
Simulation Warriors Jazz 4 vs. 1.51 + 10. 0 72.5%
Truth Warriors Jazz 4 vs. 0 + 15 100%
Simulation Wizards Hawks 4 vs. 3.26 + 1. 92 55.1%
Truth Wizards Hawks 4 vs. 2 + 1. 17 66.7%
Simulation Celtics Wizards 4 vs. 3.53 + 1. 11 53.1%
Truth Celtics Wizards 4 vs. 3 + 1 57.1%
Simulation Celtics Cavaliers 3.49 vs. 4 − 1. 62 46.6%
Truth Celtics Cavaliers 1 vs. 4 − 20 20.0%
```
to that of previous starting forward Trevor Ariza. As the
simulated win percentages show (Table 3), the ideal 5th man
that is currently on the Rockets roster—and is also among
the data’s 500 most common players—for combating this
Warriors lineup is Nene. Compared to Trevor Ariza, only
Nene is predicted to add more value. Interestingly, offseason
acquisition and superstar Carmelo Anthony is projected to
add slightly less value for the Rockets when facing the
Warriors than Trevor Ariza (Table 3).

#### IV. CONCLUSIONS AND FUTURE WORK

In this study, we have demonstrated that NBA2Vec em-
beddings can be applied to micro-scale prediction tasks: in
particular, predicting the play-by-play outcome distribution

```
TABLE 3
ROCKETS’LINEUPS VS. WARRIORS FOR VARYING“5TH MAN”
Player Win % Avg. Margin of Victory (± 1 σ)
Trevor Ariza 34.4 % − 6. 69 ± 17
Nene 37.3% − 5. 3 ± 17
Carmelo Anthony 33.4% − 7. 37 ± 17. 1
Gerald Green 31.1% − 8. 47 ± 17. 1
Joe Johnson 30.7% − 8. 1 ± 16. 8
Brandon Knight 28.5% − 9. 22 ± 16. 7
Michael Carter-Williams 27.4% − 9. 58 ± 16. 6
```
```
given players on the court. Moreover, we have also demon-
strated that NBA2Vec also shows potential towards macro-
scale prediction tasks, such as in identification of optimal
lineups and prediction of game outcomes. In future applica-
```

tions, this could be extrapolated to predictive algorithms for
projections of each team’s win-loss record given the players
on its roster.
In addition to applications in predictive tasks, we have
shown that the generated NBA2Vec embeddings are able to
reveal underlying features of players without using aggregate
statistics such as points, FG%, and assists. Clustering on the
embeddings generally groups players in agreement with their
position and our priors about their play style/characteristics.
Furthermore, the embeddings in part reflect traditional per-
formance metrics, as we are able to show that they correlate
at a significant level with box score statistics including
rebounds, assists, and field goal rate. Given enough G-
League and NCAA training data, player embeddings for
potential recruits could also be generated. By examining the
nearest neighbor embeddings in the NBA player space, the
recruit’s “equivalent NBA player” representation could aid
scouts in characterizing him and how he would contribute to
a given NBA roster.
There are various improvements that can be made to
potentially extract better player embeddings. Instead of train-
ing to predict a singular outcome to every play, a more
complex model would train to predict a series of outcomes to
each play (e.g. missed shot followed by defensive rebound,
shooting foul followed by 2/2 free throws made). To further
increase the richness of the embeddings, the network could
also be modified to predict the player who commits each
action. Finally, with the appropriate player tracking data,
a recurrent neural network could be used to take as input
time series of player spatial positions and attempt to predict
play outcomes and later player spatial positions. Similar to
the embeddings generated in this study, these improvements
would use only raw data to capture each player’s features and
“identity.” Ultimately, we envision a future for basketball an-
alytics in which player embeddings allow for unprecedented
levels of player characterization, driving predictive models
that revolutionize the way front office and coaching decisions
are made.

V. ACKNOWLEDGEMENTS
We would like to thank the NBA for organizing the 2018
NBA Hackathon and providing the data for this analysis. We
would also like to extend our thanks to Caitlin Chen for her
generosity during our trip.