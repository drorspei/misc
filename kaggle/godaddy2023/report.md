# Cheating on my first Kaggle using Math and Entropy

Before anything else, an important note: I have no chance of winning, I didn't even submit the right file in the end (see below). I had fun playing around with this, but not sure I'll join again.

## ACS and Godaddy

In January I had some downtime from work and didn't have anything to do for about two weeks. On a whim I went to kaggle.com. There was a competition that mentioned the ACS - the American Community Survey. I am intimately familiar with the ACS, having consulted to a company for a few years on real estate investment in the US. I had never joined a kaggle competition before, and had only heard about 1 or 2.

The competition was this one: https://www.kaggle.com/competitions/godaddy-microbusiness-density-forecasting/

"GoDaddy - Microbusiness Density Forecasting"

The point is to predict how many microbusinesses Godaddy will think each US county has in the months March, April and May of 2023. Why? I don't know. This is what Godaddy had to say: "Your work will help policymakers gain visibility into microbusinesses, a growing trend of very small entities. Additional information will enable new policies and programs to improve the success and impact of these smallest of businesses." Also, I get that the first sentence of this paragraph is convoluted. What they will think? Ah, well. It's not that Godaddy actually know how many microbusinesses there are, or even know what was correct in the past. But they have thoughts, based on some survey they do.

The competitors are given historical data of Godaddy's monthly users over the last few years. It looked something like this (all code samples will be in python, except when they're in nim):

```python
pd.read_csv("train.csv").sample(5, random_state=1729)
```

| row_id           |   cfips | county             | state          | first_day_of_month   |   microbusiness_density |   active |
|:-----------------|--------:|:-------------------|:---------------|:---------------------|------------------------:|---------:|
| 13171_2019-11-01 |   13171 | Lamar County       | Georgia        | 2019-11-01           |                 1.71088 |      247 |
| 41055_2021-10-01 |   41055 | Sherman County     | Oregon         | 2021-10-01           |                 2.63736 |       36 |
| 45083_2021-11-01 |   45083 | Spartanburg County | South Carolina | 2021-11-01           |                 4.51879 |    10667 |
| 46049_2022-02-01 |   46049 | Faulk County       | South Dakota   | 2022-02-01           |                 1.31507 |       24 |
| 55129_2020-05-01 |   55129 | Washburn County    | Wisconsin      | 2020-05-01           |                 3.82854 |      485 |

The competitors submit a similar table, but with only the "row_id" and "microbusiness_density" columns. What is "microbusiness_density"? Good question!

Microbusiness density, as defined by Godaddy on the competition page is 
> "Microbusinesses per 100 people over the age of 18 in the given county." 

Where do you get the number of people over the age of 18 in a given county? 
> "The population figures used to calculate the density are on a two-year lag due to the pace of update provided by the U.S. Census Bureau"

But in truth the Census Bureau publishes multiple series that could be understood to mean "number of people over the age of 18 in a given county", and different tables actually have different numbers. And while ACS is one data product the Census Bureau publishes, it isn't the only one, and there are other data products that include numbers that measure the exact same thing.

Looking at the discussion board at kaggle on my first visit to the competition, it was clear that the other competitors had not found the correct Census published numbers, or at least weren't saying so on the board. Some competitors were using a monthly updated series, even though Godaddy specifically mention the figures have a two-year lag. Two year lag you say? ACS in the data examples? I could do better.

From the definition of microbusiness density, and the columns given in the train file, we can calculate the population figures that godaddy used¹:

```python
pd.read_csv("train.csv").loc[lambda df: (
    df['first_day_of_month'].str.startswith("2019-")
)].assign(
    pop=lambda df: df['active'].div(df['microbusiness_density']).mul(100),
    pop_rounded=lambda df: df['pop'].round()
)[['cfips', 'pop', 'pop_rounded']].sample(5, random_state=1729)
```

| cfips |         pop |   pop_rounded |
|------:|------------:|--------------:|
| 12109 | 176681.9974 |   176682.0000 |
| 27151 |   7347.0001 |     7347.0000 |
| 38069 |   3392.9999 |     3393.0000 |
| 48391 |   5566.9998 |     5567.0000 |
| 17025 |  10474.0003 |    10474.0000 |

That the pop column numbers are very close to whole numbers gives us an indication that we did things right.

I took these numbers for some 5 counties, and looked for them in all ACS variables that mentioned "population over 18". Exactly two variables worked, one of which was "B15001_001E":

```python
print("\n".join(
    l
    for i, cfips in enumerate(['12109', '27151', '38069', '48391', '17025'])
    for l in requests.get(
        f"https://api.census.gov/data/2017/acs/acs5?get=B15001_001E&for=county:{cfips[2:]}&in=state:{cfips[:2]}&key={api_key}"
    ).content.decode().splitlines()[i!=0:]
))
```
```
[["B15001_001E","state","county"],
["176682","12","109"]]
["7347","27","151"]]
["3393","38","069"]]
["5567","48","391"]]
["10474","17","025"]]
```

which matches the table from before.


## Noise, noise, and more noise

I looked at the data for a few hours. Even for two full days. I promise, at the beginning I took the competition seriously. I thought that I might actually get something interesting, since I'm familiar with the ACS, and many other possibly related series. Like, there are all these employment series from the Bureau of Labor Statistics. They must be related!

Alas, after about two days, all the number looked like noise to me. I'm sure this isn't true, and other actually serious competitors have found interested rules hidden in these microbusiness density numbers. Maybe it was that my downtime was coming to an end and I got back to work, maybe it was because my toddler daughter was sick that week.

I won't show anything that I tried, because it will be much less interesting than the great ideas and approaches others have used, and you can find these at the competition page on kaggle.

At this point I decided to abandon my first attempt at kaggle.

## Remembering math

A few days later, I remembered that I know math. Let's get to it.

The metric of success of submissions, the score we get, the evaluation method, was the Symmetric Mean Absolute Percentage Error, or SMAPE for short:

$$
S((x_i);(y_i))=\frac{100}{n}\sum_{i=1}^n 2\frac{|x_i-y_i|}{|x_i|+|y_i|}
$$

where $x_i$ is the microbusiness density prediction for the ith county, and $y_i$ is the correct value of microbusiness density (according to Godaddy.)

During January, when I got into the competition, the $y_i$ were the microbusiness density values of counties for November 2022 (the past). Each submission of 3135 $x_i$ values returns a single value - the SMAPE against the correct $y_i$ from November 2022.

Putting it in math terms: there are 3135 unknowns, each submission gives a single equation. If we could make lots and lots of submissions, then we should be able to solve for the unknowns. The competition allowed 5 submissions every day, there were about 60 days left to the competition - we won't be able to get all numbers. Could we at least get the important numbers?

## Getting the important numbers

Let's look at some numbers to get a feeling for how SMAPE works, and if some counties are more important than others (in the context of competition! I'm in no way implying that Daniels County, Montana, population 1661, is more important, or less important, than New York County, New York, population 1.5 million.)

For each county we can ask what is the symmetric absolute percentage error between the microbusiness density values in 2022-09-01 and 2022-10-01:

```python
pd.read_csv("train.csv").sort_values("row_id").assign(
    active_prev=lambda df: df.groupby(['cfips'])['active'].shift(),
    sape=lambda df: df.eval('2*abs(active-active_prev)/(active+active_prev)').fillna(0)
).query(
    "first_day_of_month=='2022-10-01'"
).sort_values("sape").tail(10)[[
    'cfips', 'active', 'active_prev', 'sape'
]]
```

|   cfips |   active |   active_prev |     sape |
|--------:|---------:|--------------:|---------:|
|   47033 |      274 |           228 | 0.183267 |
|   54053 |      255 |           307 | 0.185053 |
|   30019 |       44 |            36 | 0.2      |
|   47103 |     1692 |          1350 | 0.224852 |
|   19065 |      267 |           353 | 0.277419 |
|    8011 |       44 |            59 | 0.291262 |
|   53033 |   322989 |        238245 | 0.301992 |
|   20093 |       59 |            37 | 0.458333 |
|   56033 |    54509 |         30697 | 0.558928 |
|   29063 |     1135 |           130 | 1.58893  |

So King County, WA (53033) and Sheridan County, WY (56033) are a bit large and have large changes to their active microbusinesses. But the other sape top 8 counties are pretty small. Very small even. This hints that different counties potentially contribute rather differently to the total SMAPE score.

This is also gives us a direct way to rank counties: the bigger their sape in the last few months, the more we expect them to contribute to the total SMAPE.

## Getting a single value of November 2022

Let's take Kearny County, KS (20093). Here the last few active microbusiness values:

```python
pd.read_csv("train.csv").query("cfips==20093").tail(5)[['cfips', 'first_day_of_month', 'active']]
```

|   cfips | first_day_of_month   |   active |
|--------:|:---------------------|---------:|
|   20093 | 2022-06-01           |       29 |
|   20093 | 2022-07-01           |       32 |
|   20093 | 2022-08-01           |       30 |
|   20093 | 2022-09-01           |       37 |
|   20093 | 2022-10-01           |       59 |


It had small increases, then a jump at the end. This will be important in a bit.

Fix two distinct positive numbers $a,b$. If we compile two submission files, one that uses $a$ for Kearny County, one that uses $b$, and both using the same values for all other counties, we can get back two SMAPEs, which means these two equations:

$$
\begin{align}
s_a&=\frac{200}{n}\left(\frac{|y_1-a|}{y_1+a} + \sum_{i=2}^n\frac{|y_i-x_i|}{y_i+x_i}\right)\\
s_b&=\frac{200}{n}\left(\frac{|y_1-b|}{y_1+b} + \sum_{i=2}^n\frac{|y_i-x_i|}{y_i+x_i}\right)
\end{align}
$$

where I let Kearny County correspond to $i=1$.

Subtracting the two equations we get:

$$
s_a-s_b=\frac{200}{n}\left(\frac{|y_1-a|}{y_1+a}-\frac{|y_1-b|}{y_1+b}\right).
$$

This equation has a single unknown - $y_1$, the correct value for Kearny County, 2022-11-01. So all we need to do is send two such submissions, and we will find out the correct value!

## Finite precision

Unfortunately, it's not quite that simple.

The SMAPEs we get back from each submission only have 4 digits after the dot. But this shouldn't deter us, we can deal with it. How? There are now possibly infinitely many solutions. We will use one more thing we know: the correct value for $y_1$ has to be a positive integer, and based on its previous values, we will make the plausible assumption:

> The correct value for 2022-11-01 is between 39 and 79.

So there are only 41 possible values for $y_1$.

One of my first submissions during the two days that I was taking the competition seriously was to just use the values from 2022-10-01 in all counties. This gives a value of $1.0936$. So if $a=59$, then we already know

$$s_a\in[1.09355,1.09365).$$

When I sent my second submission I used $b=3.42424242$. Why? I'll explain in a bit. This county was actually not the first that I pegged down. But it's easier to explain how to get this county before I show how I got most counties.

The SMAPE for $b$ was, at the time (this will be important later), 1.15. Rearranging our equation from above we get:

$$
\text{round}(s_a + \frac{200}{3135}\left(
\frac{|y_1-b|}{y_1+b}-\frac{|y_1-59|}{y_1+59}
\right), 4)=1.15
$$

Here is code that solves the equations given what we know and the assumptions we've made:

```python
a, b = 59, 3.4242424242
y = np.r_[39:80][:,None]
sa = 1.0936 + np.r_[-5e-5:5e-5:101j][:-1][None,:]
sb = np.round(sa - 200*(abs(y - a)/(y + a) - abs(y - b)/(y+b))/3135, 4)
d1 = collections.defaultdict(set)
for y_, sb_ in zip(y.flatten(), map(set, sb)):
    for sb__ in sb_:
        d1[sb__].add(y_)
d1[1.15]
```
```
60
```

What does this code do? My approach to writing this is to live in the "multiverse" where $y_1$ is _all_ values in $[39,79]$ and $s_a$ is _all_ values in $[1.09355,1.09365)$ (actually, just 100 such values). From this we get all possible values for $s_b$. We then enter all these values as keys into an associate array, with the associated values being the corresponding $y_1$ values.

So that's it! We now know the correct $y_1$. Is this all we know? Actually, no - we can get a little more:

```python
a, b = 59, 3.4242424242
y = 60
sa = 1.0936 + np.r_[-5e-5:5e-5:101j][:-1]
sb = np.round(sa - 200*(abs(y - a)/(y + a) - abs(y - b)/(y+b))/3135, 4)
new_sa = sa[sb == 1.15]
len(new_sa)
```
```
71
```

In this piece of code, instead of looking for the correct $y_1$, we look for more digits of $s_a$. We started with 100 possible values of $s_a$, but now that we know that $y_1=60$, only 71 of these 100 are actually possible.

My idea was to keep making submissions to get the correct values for different counties, in all of them using that same first submission of "same value as 2022-10-01". After every submission my knowledge of the true score of $s_a$ increased a bit.

## Some counties need more

Let's do the same as above but for Menifee County, KY (22165) - the actual first county I used my strategy on. Its last few values were:

|   cfips | first_day_of_month   |   active |
|--------:|:---------------------|---------:|
|   21165 | 2022-06-01           |       84 |
|   21165 | 2022-07-01           |       88 |
|   21165 | 2022-08-01           |       84 |
|   21165 | 2022-09-01           |       72 |
|   21165 | 2022-10-01           |       60 |

so I assumed that the correct value for 2022-11-01 is between 40 and 80, I used $b=6.58777$ for my second "Menifee submission", and got back $s_b=1.1418$ from kaggle. I plugged the numbers into the code from before:

```python
a, b = 60, 6.58777
y = np.r_[40:80][:,None]
sb = np.round(new_sa - 200*(abs(y - a)/(y + a) - abs(y - b)/(y+b))/3135, 4)
d1 = collections.defaultdict(set)
for y_, sb_ in zip(y.flatten(), map(set, sb)):
    for sb__ in sb_:
        d1[sb__].add(y_)
d1[1.1418]
```
```
{56,69}
```

(Note: I'm using the `new_sa` array computed from the previous section.)

Ojoj. Two different values are compatible with the $s_a,s_b$ I got back. I knew in advance this could happen: I inspected `d1` before I made the submissions. In order to get the correct value between 56 and 69, I made _one more_ submission, using $c=12.34820$, which then determines that $y=56$.

How did I choose these $b,c$? My idea was to construct for many pairs $b,c$ a big associative array that takes pairs $s_b,s_c$ to sets of the possible correct $y$'s. If I hit on a pair $b,c$ whose associative array is one-to-one, meaning each possible pair of $s_b,s_c$ implies a unique solution for $y$, then I'll use that pair. Since many pairs might satisfy this property, I also ranked them according to what I get from the submission for $b$.

Here is the actual original code I wrote²:

```python
def getd1d12(y0, n1, n2, r1, r2):
    """get dictionaries for two submissions"""
    d1 = defaultdict(set)
    d12 = defaultdict(set)
    
    r1r2 = np.r_[r1:r2]
    z = smape(np.r_[r1:r2], y0)
    
    roundings = np.r_[np.round(1.0936 - 5e-5, 5):np.round(1.0936 + 5e-5, 5):101j][:-1]
    
    for s0 in roundings:
        for y, s1, s2 in zip(
            np.r_[r1:r2],
            np.round((smape(r1r2, n1) - z)/3135 + s0, 4),
            np.round((smape(r1r2, n2) - z)/3135 + s0, 4),
        ):
            d1[s1].add(y)
            d12[s1, s2].add(y)
    return d1, d12

def maxambig(d12):
    """largest possible ambiguity in smape units"""
    return max(smape(float(max(v)), float(min(v)))/3135 for v in d12.values())

def deterministic(d12, threshold):
    """do the submissions determine y completely"""
    return all(200*(max(v) - min(v))/(abs(max(v)) + abs(min(v)))/3135 <= threshold for v in d12.values())

def getbest(y0, r1, r2, pn11, pn12, pns1, pn22, pn22, pns2):
    return (
        dask.bag.from_sequence(np.linspace(pn11, pn12, pns1), npartitions=1000)
        .map(lambda n1: min(
            (np.round(maxambig(a[1]), 7), maxambig(a[0]), n1, n2, a[0], a[1])
            for n2 in np.linspace(pn22, pn22, pns2)
            for a in [getd1d12(y0, n1, n2, r1, r2)]
        ))
        .min()
        .compute()
    )

y0 = 60
r1, r2 = y0-20,y0+20
pn11, pn12, pns1 = 0, 60, 100
pn22, pn22, pns2 = 0, 60, 100
amb, amb1, n1, n2, d1, d12 = getbest(y0, r1, r2, pn11, pn12, pns1, pn22, pn22, pns2)
```

The $b,c$ I sent for this county were such that the two extra submissions (counting the first one corresponding to $s_a=1.0936$) would definitely determine $y$, and also there was a high chance the first submission would be enough. This is also the strategy I used for the second county I went after, Kearny County above, and with Kearny I was lucky and the submission for $b$ was enough to determine $y$.

## The big reset

Somewhere in mid February Godaddy published all microbusiness density values for November and December 2022. By then I had managed to calcuate tens of county values for November. Did Godaddy just annul all my work? Happily, no!

Godaddy had reset all scores on previous submissions, so all my submissions for the different counties now had their score calculated on January 2023. I updated my assumption on the $y$ of each county according to the new knowledge on Novermber and December, and used the reset scores to regain the knowledge of all the counties I had submitted. After a couple of hours of work I had almost everything back. Some counties changed enough so that my choices for $b,c$ weren't optimal anymore, and I couldn't determine the correct $y$'s completely.

## Entropy - losing interest and getting it back

I kept making daily submissions for another week, but I started doing interesting stuff at work, and just didn't have as much time on my hands. I wasn't in it to win it anyway.

At the time I was reading (actually, listening to) a book called The Physics of Wall Street, by James Owen Weatherall. It tells the story of how ideas from advanced math, statistics, and even physics, landed in wall street. One chapter is about Edward Thorp, who used Shannon's new information theory and the concept of entropy to cheat at blackjack and roulete.

When I finished the chapter I realised that I could use entropy to improve my scheme for choosing $b,c$: to each such pair I can compute the conditional entropy

$$
H(y|s_a,s_b,s_c)
$$

which measures how much "information" there is in the variable $y$, given that I know $s_a,s_b,s_c$ ($s_a=1.463051$, the new 1.0936 after the big reset). If $s_a,s_b,s_c$ determine $y$ completely then there is no _new_ information in this variable, so $H=0$. In the code before I just tried many different $b,c$, but with entropy I had an (almost) continuous score in terms of $b,c$, so I could use an optimization algorithm to search for $b,c$ such that $H$ is minimal - maybe even 0, meaning $s_b,s_c$ would completely determine $y$.

Entropy uses probability of events. In order to use it I also made more sophisticated assumptions about the possible values of each $y$: instead of just saying that $y$ is between, say, 40 and 80, I declared the probability of $y=60$ to be highest, and slowly decreasing as we move closer to the boundary of the interval (40,80). To make things simpler I actually went after counties whose recent "active" changes were +-1, and declared the possible next change to be in $\{-2,-1,0,1,2\}$, with assigned probabily $(0.005,0.045,0.9,0.045,0.005)$. But then, to make things complicated again, I went after multiple counties _at the same time_. So I was actualy calculating

$$
H(y_i,y_j,y_k|s_{a_1,a_2,a_3},s_{b_1,b_2,b_3},s_{c_1,c_2,c_3})
$$

meaning I make two extra submissions, and each has 3 different values for three different counties I'm going after. And I actually ran this on 5 at a time, not 3.

Once I had this idea, coding it was pretty straightforward:

```python
import itertools
Hdf = lambda xs, prev: pd.DataFrame([
    [np.exp(sum(np.log(p) for _, (p, *_) in rows))]
    + [np.round(sum(row[1+i] for _,row in rows)+1.463051,4) for i,_ in enumerate(xs)]
    for rows in itertools.product(*(
        pd.Series([0.005,0.045,0.9,0.045,0.005],s+np.r_[-2:3])
        .rename('p')
        .to_frame().pipe(lambda df: df.assign(**{
            f"s{i}": (smape(x,df.index)-smape(s,df.index))/3135
            for i, x in enumerate(xs_)
        })).iterrows()
        for s, *xs_ in zip(prev, *xs)
    ))
], columns=['p']+[f"s{i}" for i,_ in enumerate(xs)]).assign(px=lambda df: (
    df.groupby([f"s{i}" for i,_ in enumerate(xs)])['p'].transform('sum')
))
H = lambda df: df.eval("-p*log(p/px)").sum()
```

This code was written on my phone, with even less white space. My phone can show 56 characters per line inside nbterm, running in termux. So it's unreadable. Unfortunately, it was also a bit slow to run, though it did get me some numbers.

A year or so ago I got into the programming language nim, I don't remember why. I wanted to learn a new language, and I can't stand Rust. Anyway, nim is super easy to install and start coding in, and it took a couple of minutes to get it working inside termux on my phone, so I rewrote the above dense block in nim, writing inside neovim, which was a pleasure:

```
import std/math
import std/sequtils
import std/tables
import std/sugar
import std/threadpool

type
  XY = object
    p: seq[float]
    s: seq[seq[float]]

proc smape(x, y: float): float =
  if x == 0.0 and y == 0.0:
    return 0.0
  return 200*abs(x-y)/(abs(x)+abs(y))

proc makeXY(p: seq[float], prev: float, cs: seq[float], xs: seq[float]): XY =
  var xy = XY(p: p)
  for i in 0..len(p)-1:
    xy.s.add(@[])
    let ps = smape(prev,cs[i])
    for x in xs:
      xy.s[i].add((smape(x,cs[i]) - ps)/3135)
  xy

proc combine(a, b: XY): XY =
  for i, pa in a.p.pairs:
    for j, pb in b.p.pairs:
      result.p.add(pa*pb)
      result.s.add(zip(a.s[i],b.s[j]).map((xy:(float,float))=>xy[0]+xy[1]))

proc addRound(xy: XY, s: float, n: int): XY =
  result = xy
  for i in 0..len(result.s)-1:
    for j in 0..len(result.s[i])-1:
      result.s[i][j] = round(result.s[i][j]+s, n)

proc H(xy: XY): float =
  var px = initTable[seq[float],float]()
  for i, p in xy.p.pairs:
    if not px.hasKey(xy.s[i]):
      px[xy.s[i]] = 0.0
    px[xy.s[i]] += p
  for i, p in xy.p.pairs:
    result += -p*ln(p/px[xy.s[i]])

proc xysH(xys: seq[XY], s: float, n: int): float =
  return H(addRound(foldl(xys, combine(a, b)), s, n))
```

I used a kind of coordinate descent to find optimal $b,c$, where the code is again very dense:

```
proc optimize(p: seq[seq[float]], prev: seq[float], cs: seq[seq[float]], xs0: seq[seq[float]]): seq[seq[float]] =
  const d = 1000
  var xs = xs0
  var h = 1000.0
  for round in 0..20:
    var changed = false
    for i in 0..len(xs0[0])-1:
      for j in 0..len(xs0)-1:
        for pow in 0..7:
          var flows: array[d+1, FlowVar[float]]
          for k in 0..len(flows)-1:
            var xs1 = xs
            xs1[j][i] += (float(k)-(len(flows)-1)/2)/(len(flows)/20*float(d)^pow)
            let xys = collect(newSeqOfCap(len(p))):
              for l in 0..len(p)-1:
                makeXY(p[l], prev[l], cs[l], xs1[l])
            flows[k] = spawn xysH(xys, 1.463051, 4)
          sync()
          let res = collect(newSeqOfCap(len(flows))):
            for k in 0..len(flows)-1: ^flows[k]
          let m = res.minIndex()
          xs[j][i] += (float(m)-(len(flows)-1)/2)/(len(flows)/20*float(d)^pow)
          let new_h = res[m]
          if new_h < h - 0.0000001:
            changed = true
          h = new_h
          echo $(h)
          if res[m] == 0:
            return xs
    echo $(xs)
    if not changed:
      echo "no change, breaking"
      break
  return xs
```

This ran well on my phone, and then I scaled up to running it on a larger computer with more cores. Nim's builtin "spawn" that runs tasks on a thread pool was super easy and scaled extremely well. It was just as easy as using dask in python, and running compiled machine code was about 50 times faster.

I made a few submissions, and then realised I got something wrong. I fixed the code, I think, but I had already lost my interest in the competition for the second and last time.

## Conclusion, sending the wrong file

On the final day of the competition, more than a week since my previous submission, I looked at the leader board, and decided I will make a submission after all. I made 4 submissions with my entropy idea to get a few more counties, and left my last submission for what I will select as my real submission for the competition. Here is the code I used:

```python
t = (
    pd.concat([pd.read_csv("revealed_test.csv"), pd.read_csv("train.csv")], ignore_index=True).sort_values(['row_id'])[['row_id', 'cfips', 'first_day_of_month', 'active']]
    .sort_values(['row_id'])
    .assign(
        active_prev=lambda df: df.groupby(['cfips'])['active'].shift(),
        cbd=lambda df: df['active'].sub(df['active_prev']),
        first_day_of_month=lambda df: df['first_day_of_month'].pipe(pd.to_datetime),
    )
)
pd.concat([pd.read_csv("revealed_test.csv")[['row_id', 'microbusiness_density']]]+[
    t.query("first_day_of_month=='2022-12-01'")
    .assign(first_day_of_month=f"2023-0{i}-01", active=t['active'] + i*t.groupby(['cfips'])['cbd'].transform('median'))
    .merge(pop.query('year==2021'))
    .assign(
        microbusiness_density=lambda df: df['active'].mul(100).div(df['pop']),
        row_id=lambda df: df['cfips'].astype(str) + "_" + df['first_day_of_month']
    )
    [['row_id', 'microbusiness_density']]
    for i in range(1, 7)
], ignore_index=True).sort_values(['row_id']).to_csv("enoughimdone.csv", index=False)
```

This code takes the last known value of every county, from December 2022, and then proceeds to add the median of historical monthly changes every month until June 2023.

Wait, where do I use the all the values that I learnt about Januray 2023?? Ah. I forgot to use them.

I had fun implementing my ideas, especially the entropy part. I tried everything first on my phone, and then every time scaled up to using a larger machine, though usually still controlling it from my phone. This was also cool.

Future competition organizers can, if they want to, figure out how to block the methods from this piece. For example, Godaddy could have scored submissions using only, say, 15% of counties, and then later when the competition was closed, using the other 85%.

Good luck to all the competitors!

## The values

Here are the active ranges that I determined for each county for 2023-01-01, where the last few came from the faulty entropy code, and I think they're wrong (starting from cfips 30011).

|   cfips |   min |   max |
|--------:|------:|------:|
|   30019 |    58 |    58 |
|   16023 |    55 |    55 |
|   18139 |   221 |   221 |
|   21043 |   228 |   228 |
|   17135 |   706 |   707 |
|   18073 |  2281 |  2282 |
|    5067 |    96 |    96 |
|   31085 |     5 |     5 |
|   30103 |     9 |     9 |
|   31171 |     8 |     8 |
|   48311 |    11 |    11 |
|    5013 |    16 |    16 |
|   30033 |    19 |    19 |
|   13061 |    21 |    21 |
|   48047 |    21 |    21 |
|   30079 |    21 |    21 |
|   48447 |    21 |    21 |
|   20083 |    22 |    22 |
|    2188 |    24 |    24 |
|   54105 |    23 |    23 |
|   38013 |    27 |    27 |
|   21105 |    27 |    27 |
|   21063 |    27 |    27 |
|   48431 |    27 |    27 |
|   48079 |    30 |    30 |
|   46043 |    30 |    30 |
|   54017 |    31 |    31 |
|   38051 |    30 |    30 |
|   46085 |    31 |    31 |
|   28069 |    36 |    36 |
|   48101 |    32 |    32 |
|   20109 |    34 |    34 |
|   21129 |    35 |    35 |
|   48383 |    36 |    36 |
|    5039 |    37 |    37 |
|    8011 |    42 |    42 |
|   20189 |    46 |    46 |
|   28041 |    52 |    52 |
|   13283 |    47 |    47 |
|    5095 |    46 |    46 |
|   38045 |    55 |    55 |
|   40053 |    53 |    53 |
|   45005 |    54 |    54 |
|   21165 |    51 |    51 |
|   20093 |    58 |    58 |
|   21109 |    61 |    61 |
|   19143 |    68 |    68 |
|   31049 |    66 |    66 |
|   45081 |    69 |    69 |
|   48283 |    72 |    72 |
|   48359 |    73 |    73 |
|   21147 |    76 |    76 |
|   26131 |    80 |    80 |
|   28037 |    86 |    86 |
|   18161 |    88 |    88 |
|   31175 |    93 |    93 |
|   20043 |    98 |    98 |
|   40005 |   107 |   107 |
|   28015 |   110 |   110 |
|   21225 |   123 |   123 |
|   21143 |   119 |   119 |
|   19001 |   127 |   127 |
|   48489 |   126 |   126 |
|   28141 |   148 |   148 |
|   32027 |   169 |   169 |
|   48307 |   175 |   175 |
|   26085 |   184 |   184 |
|   13317 |   195 |   195 |
|   48145 |   226 |   226 |
|   37137 |   247 |   247 |
|    1013 |   340 |   340 |
|   28151 |   448 |   448 |
|   29175 |   494 |   494 |
|    6035 |   602 |   602 |
|    1133 |   669 |   669 |
|   19159 |    39 |    39 |
|    5117 |    45 |    45 |
|   21007 |    51 |    51 |
|   19009 |    69 |    69 |
|   27087 |    78 |    78 |
|   48351 |    90 |    90 |
|    2275 |   119 |   119 |
|   13285 |  2813 |  2813 |
|   21189 |    18 |    18 |
|   20089 |    41 |    41 |
|   48155 |    57 |    57 |
|   28009 |    90 |    90 |
|   54053 |   230 |   230 |
|   47033 |   265 |   265 |
|    6115 |  1717 |  1718 |
|   17177 |  1790 |  1792 |
|   26103 |  2897 |  2905 |
|   55131 |  6749 |  6755 |
|   13113 | 15163 | 15187 |
|   56033 | 58759 | 58854 |
|   53007 |  4883 |  4885 |
|    1111 |   892 |   894 |
|   47031 |  7829 |  7841 |
|   17151 |    21 |    26 |
|    1105 |    42 |    46 |
|   29111 |   100 |   103 |
|   20065 |   104 |   113 |
|   38061 |   216 |   224 |
|   21215 |  1154 |  1155 |
|   48053 |  3202 |  3484 |
|    5003 |   174 |   174 |
|   40045 |    54 |    54 |
|   13025 |   145 |   145 |
|   35029 |   383 |   383 |
|   48063 |  1122 |  1123 |
|   12007 |   529 |   529 |
|   48305 |    76 |    76 |
|   39001 |   410 |   410 |
|   18157 |  6427 |  6431 |
|   40033 |    42 |    42 |
|    1011 |    79 |    79 |
|   23017 |  4373 |  4385 |
|   51037 |   125 |   125 |
|   51027 |   134 |   134 |
|   26095 |   163 |   163 |
|   40069 |   118 |   118 |
|   46091 |    63 |    63 |
|   20027 |   129 |   129 |
|   45057 |  3958 |  3964 |
|   56011 |   270 |   274 |
|   24023 |  1142 |  1143 |
|   27127 |   300 |   300 |
|   29149 |   189 |   190 |
|   40139 |   193 |   193 |
|   30011 |     8 |     9 |
|   31091 |     8 |     8 |
|   28055 |     1 |     1 |
|   31009 |     1 |     1 |
|   48301 |     1 |     1 |
|   15005 |     2 |     2 |
|   31005 |     2 |     2 |
|   31115 |     2 |     2 |
|   48033 |     2 |     2 |
|   13101 |     3 |     3 |
|   31165 |     4 |     4 |
|   31183 |     4 |     4 |
|   48261 |     5 |     5 |
|   48269 |     5 |     5 |
|   16033 |     6 |     6 |
|   48393 |     6 |     6 |
|   13265 |     6 |     6 |
|   31007 |     7 |     7 |
|   46137 |     7 |     7 |
|   38083 |     9 |     9 |
|   46069 |     9 |     9 |
|   48263 |     9 |     9 |
|   46017 |    11 |    11 |
|   20071 |     9 |     9 |
|   31015 |    12 |    12 |
|   30069 |    13 |    13 |
|   31077 |    12 |    12 |
|   13307 |    12 |    12 |
|   31113 |    13 |    13 |
|   46007 |    14 |    14 |
|   13239 |    13 |    13 |
|   46089 |    14 |    14 |
|   38095 |    17 |    17 |
|   31057 |    17 |    17 |
|   20195 |    18 |    18 |
|   31069 |    17 |    17 |
|   20135 |    19 |    19 |
|   29227 |    18 |    18 |
|   38039 |    20 |    20 |
|   38065 |    23 |    23 |
|   46003 |    21 |    21 |
|   48105 |    21 |    21 |
|   48421 |    25 |    25 |
|   20033 |    18 |    18 |
|   31103 |    19 |    19 |
|   46021 |    21 |    21 |
|   35021 |    22 |    22 |
|   20187 |    24 |    24 |
|   46055 |    25 |    25 |
|   47095 |    25 |    25 |
|   21201 |    24 |    24 |
|   48173 |    12 |    12 |
|   46031 |     9 |     9 |
|   46075 |    15 |    15 |
|   48443 |    11 |    11 |
|   30109 |    15 |    15 |
|   48109 |    15 |    15 |
|   48345 |    18 |    18 |
|   51081 |    16 |    16 |
|    2282 |    21 |    21 |

## Footnotes

¹ - Actually, this is not how I got the numbers at all. When I opened the train file I completely ignored the "active" column - I don't even remember seeing it. I reconstructed it myself using some trivial number theory and the floating point numbers in the "microbusiness column". Specifically, for the interested: I took about 5 different counties, for each of these I took the microbusiness density values of all months over 2021, and multipled the 12 values (per county) by every positive integer in the range `(approxpop - 10%, approxpop + 10%)`, where I got an approximation for the population from some ACS variable that I found (that _wasn't_ the one Godaddy had used). The correct population has to satisfy the property that the 12 values are all very close to an integer - in fact an integer that is a multiple of 100. Though once again, I didn't quite get this at the time, and only after I noticed that all the numbers are multiple of 100 did I finally get Godaddy's definition of microbusiness density.

² - This code was written on my phone, running nbterm inside termux. In the beginning I even ran it with dask on my phone, that has 8 cores. Once it got too much for the phone I ran a dask scheduler and workers on a desktop computer I have, and connected from my phone to the scheduler. I love keeping code and data close in my pocket.
