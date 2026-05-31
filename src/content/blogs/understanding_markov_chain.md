---
title: "Understanding Markov Chains"
excerpt: "This a blog post which covers basics of Markov Chain and how it shapes our daily life and modern machine learning."
date: "2024-03-15"
readTime: "58 min read"
category: "Machine Learning"
---
## Contents

## 1. Markov Chain

### 1.1 Motivation

In probability theory, an independent and identically distributed (iid) process is defined as a sequence of random variables that are mutually independent and share the same probability distribution. However, many real-world processes exhibit some form of dependence between observations. In these cases, the Markov chain offers a natural, preliminary model for capturing such dependencies. By assuming that the future state depends solely on the present state (and not on the sequence of past states), a Markov chain provides a fundamental framework to model and analyze the first level of dependency in a stochastic process. They are instrumental in addressing a variety of complex problems, such as:

* **Card Shuffling**: Determining how many shuffles are required to sufficiently randomize a deck of cards.
* **Queue Dynamics**: Estimating the probability that a queue's buffer will overflow, given random arrival and service times.
* **Web Page Popularity**: Analyzing the structure of hyperlinks between web pages to infer their relative importance or popularity, which is a key concept behind algorithms such as PageRank.

### 1.2 Definition

Given a probability space $(\Omega, \mathcal{F}, \mathbb{P})$:

**Definition (Markov Chain)**:
A stochastic process $\{X_{t}:t\in\mathcal{T}\}$ is called a Markov chain if for every $t\in\mathcal{T}$, the past and future are conditionally independent given the present.
i.e., for any $m,n\in\mathbb{N},$

$$ 
(X_{s}:s<t) \perp (X_{s}:s>t) | X_{t}. 
$$

This means for any $s_{1}<\cdot\cdot\cdot<s_{m}<t, t<t_{1}<\cdot\cdot\cdot<t_{n}, x_{1},...,x_{m}\in\mathbb{R}, y_{1},...,y_{n}\in\mathbb{R},$ and $x\in\mathbb{R}.$  
$$ 
\mathbb{P}(X_{s_{1}}\le x_{1},...,X_{s_{m}}\le x_{m}, X_{t}\le x, X_{t_{1}}\le y_{1},...,X_{t_{n}}\le y_{n}) 
$$

$$ 
= \mathbb{P}(X_{s_{1}}\le x_{1},...,X_{s_{m}}\le x_{m}, X_{t}\le x) \cdot \mathbb{P}(X_{t_{1}}\le y_{1},...,X_{t_{n}}\le y_{n}|X_{t}=x).
$$

In words, the future state depends solely on the present state and not on the sequence of past states.

### 1.3 Discrete-Time Markov Chain (DTMC)

A discrete-time stochastic process is a sequence of random variables $\{X_{n}:n\ge1\}$ indexed by nonnegative integers. The set of possible values that each $X_{n}$ can take is called the state space, denoted by S, which is assumed to be discrete (for example, $\mathbb{Z}$ or a finite set). A discrete-time Markov chain (DTMC) is a discrete-time stochastic process $\{X_{n}:n\ge1\}$ taking values in S that satisfies the Markov property.
i.e, for any $n\in\mathbb{N}$
$$ 
(X_{1},...,X_{n-1}) \perp (X_{n+1},X_{n+2},...) | X_{n} 
$$
Now if the state space S is finite we call this DTMC taking Finitely Many Values.

#### 1.3.1 Definition (DTMC taking Finitely Many Values)

Given a probability space $(\Omega,\mathcal{F},\mathbb{P})$:

**Definition (DTMC taking Finitely Many Values)**:
Consider a process $\{X_{n}\}_{n=1}^{\infty}$ taking values in a finite set $\mathcal{X}$. Then, $\{X_{n}\}_{n=1}^{\infty}$ is called a discrete-time Markov chain (DTMC) on $\mathcal{X}$ if for any $n\in\mathbb{N},$
$$ (X_{1},...,X_{n-1}) \perp\hspace{-.7em}\perp (X_{n+1},X_{n+2},...) | X_{n} $$
i.e., for any $n,L\in\mathbb{N}, n<t_{1}<\cdot\cdot\cdot<t_{L}, x_{1},...,x_{n-1}\in\mathcal{X}, y_{1},...,y_{L}\in\mathcal{X},$ and $x\in\mathcal{X}$
$$ \mathbb{P}(X_{1}=x_{1},...,X_{n-1}=x_{n-1}, X_{t_{1}}=y_{1},...,X_{t_{L}}=y_{L} | X_{n}=x) $$
$$ = \mathbb{P}(X_{1}=x_{1},...,X_{n-1}=x_{n-1} | X_{n}=x) \cdot \mathbb{P}(X_{t_{1}}=y_{1},...,X_{t_{L}}=y_{L} | X_{n}=x). $$

#### 1.3.2 Alternate Viewpoint of DTMC

From the Markov property, we know that for any $n\in\mathbb{N},$
$$ 
(X_{1},...,X_{n-1}) \perp\hspace{-.7em}\perp (X_{n+1},X_{n+2},...) | X_{n}. 
$$
We wish to show that
$$ 
\mathbb{P}(X_{t_{1}}=y_{1},...,X_{t_{L}}=y_{L} | X_{n}=x, X_{1}=x_{1},...,X_{n-1}=x_{n-1}) \\= \mathbb{P}(X_{t_{1}}=y_{1},...,X_{t_{L}}=y_{L} | X_{n}=x). 
$$
**Proof**:
By the definition of conditional probability and the Markov property, we have
$$ 
\mathbb{P}(X_{t_{1}}=y_{1},...,X_{t_{L}}=y_{L}, X_{1}=x_{1},...,X_{n-1}=x_{n-1} | X_{n}=x) 
$$
$$ 
= \mathbb{P}(X_{1}=x_{1},...,X_{n-1}=x_{n-1} | X_{n}=x) \times \mathbb{P}(X_{t_{1}}=y_{1},...,X_{t_{L}}=y_{L} | X_{n}=x). 
$$
Using this factorization in the conditional probability of interest,
$$ 
\mathbb{P}(X_{t_{1}}=y_{1},...,X_{t_{L}}=y_{L} | X_{n}=x, X_{1}=x_{1},...,X_{n-1}=x_{n-1}) 
$$

$$ 
= \frac{\mathbb{P}(X_{t_{1}}=y_{1},...,X_{t_{L}}=y_{L}, X_{1}=x_{1},...,X_{n-1}=x_{n-1} | X_{n}=x)}{\mathbb{P}(X_{1}=x_{1},...,X_{n-1}=x_{n-1} | X_{n}=x)} 
$$
$$ 
= \frac{\mathbb{P}(X_{1}=x_{1},...,X_{n-1}=x_{n-1} | X_{n}=x) \times \mathbb{P}(X_{t_{1}}=y_{1},...,X_{t_{L}}=y_{L} | X_{n}=x)}{\mathbb{P}(X_{1}=x_{1},...,X_{n-1}=x_{n-1} | X_{n}=x)} 
$$
$$ 
= \mathbb{P}(X_{t_{1}}=y_{1},...,X_{t_{L}}=y_{L} | X_{n}=x). 
$$
Hence, once $X_{n}$ is known, additional information about the past $\{X_{1},...,X_{n-1}\}$ does not affect the distribution of future states $\{X_{t_{1}},...,X_{t_{L}}\}$. This completes the derivation of the alternate viewpoint of a DTMC.

## 2. Examples of Discrete-Time Markov Chains

### 2.1 Snake and Ladder Game

The snake and ladder game serves as a classic illustration of a discrete-time Markov chain (DTMC). The game is played on a board with 100 squares, numbered from 1 to 100. Players begin at square 1 and roll a fair six-sided die to determine the number of squares to move forward. The board includes ladders, which allow players to advance to higher-numbered squares, and snakes, which force players to retreat to lower-numbered squares.

Let $X_{n}$ represent the position of a player's pawn at time n, where n denotes the number of moves made. The state space is defined as $\mathcal{X}=\{1,2,...,100\}$, reflecting the finite set of possible positions on the board. The Markov property is evident in the game because the probability of transitioning to a particular square at time $n+1$ depends solely on the current position $X_{n}$, not on the sequence of previous positions.

For example, suppose a player is on square 6 at time n (i.e., $X_{n}=6$). Rolling the die yields one of six possible outcomes (1 through 6), leading to potential next positions of 7, 8, 9, 10, 11, or 12, each with a probability of $\frac{1}{6}$, assuming no ladders or snakes are present at these squares. If a ladder or snake is encountered, the player is immediately transported to the corresponding square, and the transition probability is adjusted accordingly.

Formally, the transition probability from state i to state j is given by:

$$
\mathbb{P}(X_{n+1}=j | X_{n}=i) = \begin{cases}
  \frac{1}{6} &  \substack{\small\text{If j is reachable from i by a die roll,} \\ \small \text{accounting for ladders or snakes}} \\
  0 & \text{otherwise}
\end{cases}
$$


This dependence on the current state alone, independent of past states, confirms that $\{X_{n}\}$ is a DTMC with a finite state space.

`[Image: Figure 1: Snake and ladder board illustrating possible transitions.]`

### 2.2 Random Walk

The random walk is another fundamental example of a DTMC, widely used to model processes with incremental changes. Consider a sequence of independent and identically distributed (IID) random variables $\{X_{i}\}_{i=1}^{\infty}$, where each $X_{i}$ takes values in $\{-1,1\}$ with probabilities $1-p$ and $p$, respectively. The random walk $S_{n}$ is defined as the cumulative sum of these variables:
$$ 
S_{n} = \sum_{i=1}^{n} X_{i} 
$$
with the initial condition $S_{0}=0$. The state space $\mathcal{X}=\mathbb{Z}$ (the set of all integers) is countably infinite, as the walk can reach any integer through successive steps.

The Markov property holds because the next state $S_{n+1}$ is determined by:
$$ 
S_{n+1} = S_{n} + X_{n+1} 
$$
Here, $X_{n+1}$ is independent of all previous $X_{i}$, and thus the distribution of $S_{n+1}$ depends only on the current state $S_{n}$, not on the past states $S_{1},S_{2},...,S_{n-1}.$ The transition probabilities are:
$$ 
\mathbb{P}(S_{n+1}=k | S_{n}=j, S_{n-1}=i_{n-1},...,S_{1}=i_{1}) = \mathbb{P}(X_{n+1}=k-j) 
$$
$$ 
= \begin{cases} p & \text{if } k=j+1 \\ 1-p & \text{if } k=j-1 \\ 0 & \text{otherwise} \end{cases} 
$$
Since this probability depends only on $j=S_{n}$, the process $\{S_{n}\}$ satisfies the Markov property, confirming it as a DTMC.

`[Image: Figure 2: Random walk trajectory illustrating state transitions.]`

## 3. Transition Probability Matrix

For a DTMC $\{X_{n}\}$ with a discrete state space $\mathcal{X}$, the transition probability matrix at time n, denoted $P(n) = [P_{ij}(n)]_{i,j\in\mathcal{X}}$, is defined such that:
$$ 
P_{ij}(n) = \mathbb{P}(X_{n+1}=j | X_{n}=i) 
$$
Each row of $P(n)$ represents the conditional probability mass function of the next state given the current state, satisfying:
$$ 
\sum_{j\in\mathcal{X}} P_{ij}(n) = 1 \quad \text{for all } i\in\mathcal{X}, n\ge1 
$$
The matrix $P(n)$ is square if $\mathcal{X}$ is finite, or infinite-dimensional if $\mathcal{X}$ is countably infinite. A matrix satisfying the above property is called row stochastic, as each row sums to 1 and all entries are non-negative ($P_{ij}(n)\ge0$).

For the snake and ladder game, the transition matrix P is a $100 \times 100$ matrix, where the row corresponding to state i has non-zero entries (typically $\frac{1}{6}$) for states reachable by a die roll, adjusted for ladders or snakes. For example, the row for state 6 might have non-zero entries for states 7, 8, 9, 10, 11, and possibly a state reached via a ladder (e.g., 31), with zeros elsewhere.

For the random walk, the transition matrix is infinite, with a tridiagonal structure. For state i, the non-zero entries are:

$$ 
P_{i,i-1} = 1-p 
$$
$$ 
P_{i,i+1} = p 
$$
$$ 
P_{i,j} = 0 \quad \text{for } j \ne i-1, i+1 
$$
To illustrate, consider a random walk restricted to a finite state space, say $\mathcal{X}=\{-2,-1,0,1,2\}$, with reflecting boundaries (i.e., from -2, the walk moves to -1 with probability 1, and from 2, to 1 with probability 1). Assuming $p=0.5$ for interior states, the transition matrix is:
$$ 
P = \begin{pmatrix} 0 & 1 & 0 & 0 & 0 \\ 0.5 & 0 & 0.5 & 0 & 0 \\ 0 & 0.5 & 0 & 0.5 & 0 \\ 0 & 0 & 0.5 & 0 & 0.5 \\ 0 & 0 & 0 & 1 & 0 \end{pmatrix} 
$$

### 3.1 Properties of Transition Probability Matrices

A transition probability matrix P has several important properties:

1. All entries are non-negative: $P_{ij}\ge0$ for all $i, j\in\mathcal{X}.$
2. Each row sums to 1: $\sum_{j\in\mathcal{X}}P_{ij}=1$ for all $i\in\mathcal{X}.$
3. The vector $\mathbf{1}$ (all ones) is a right eigenvector of P with eigenvalue 1: $P\mathbf{1}=\mathbf{1}.$
4. All eigenvalues $\lambda$ of P satisfy $|\lambda|\le1.$
5. If P has strictly positive entries, then $\lambda=1$ is a simple eigenvalue, and all other eigenvalues satisfy $|\lambda|<1$.

### 3.2 Interpretation of TPM Rows

Each row of the transition probability matrix represents a conditional probability mass function. Specifically, row i gives the distribution of $X_{n+1}$ given that $X_{n}=i.$ This interpretation directly connects to the Markov property: to determine the distribution of the next state, we only need to know the current state. For example, in the Snake and Ladder game, if we are currently at square 6, the corresponding row in the transition matrix would have non-zero entries only for the possible next positions (e.g., 7, 8, an up-ladder position like 31, 10, 11, 12), each with its respective probability (assuming a fair die, some would be $\frac{1}{6}$).

## 4. n-Step Transition Probabilities

### 4.1 Definition

For a time-homogeneous DTMC with transition matrix P, the n-step transition probability from state i to state j is defined as:
$$ 
P_{ij}^{(n)} = \mathbb{P}(X_{m+n}=j | X_{m}=i) 
$$
for any $m\ge0.$ This is the probability that the chain, starting from state i, will be in state j after exactly n steps.

### 4.2 Matrix Powers

For a time-homogeneous DTMC, the n-step transition matrix $P^{(n)}$ is equal to the n-th power of the one-step transition matrix P:
$$ 
P^{(n)} = P^{n} 
$$
This follows from repeated application of the Chapman-Kolmogorov equation. The $(i,j)$-entry of $P^{n}$ gives the probability of transitioning from state i to state j in exactly n steps.

### 4.3 Example

Consider a two-state Markov chain with transition matrix:
$$ 
P = \begin{pmatrix} 1-p & p \\ q & 1-q \end{pmatrix} 
$$
where $p,q\in(0,1)$.
To find the probability of being in state 0 after 4 steps, starting from state 0, we compute $P^{4}$ and look at the (0,0)-entry. For specific values, say $p=0.3$ and $q=0.4$, we can calculate this explicitly.

## 5. Time-Homogeneous Markov Chains

**Definition (Time-Homogeneous DTMC)**:
A DTMC is time-homogeneous if the transition probability matrix is constant over time, i.e., $P(n)=P$ for all $n\ge1,$ where P is a fixed row stochastic matrix.

In a time-homogeneous DTMC, the transition probabilities $\mathbb{P}(X_{n+1}=j|X_{n}=i)$ are independent of n, simplifying the analysis of the chain's behavior over multiple steps. Both the snake and ladder game (with a fixed board and die) and the random walk (with IID steps) are time-homogeneous, as their transition rules do not vary with time.

For a time-homogeneous DTMC, the probability of transitioning from state i to state j in exactly n steps, denoted $P_{ij}^{(n)}$, is given by the (i, j)-entry of the matrix power $P^{n}$:
$$ 
P_{ij}^{(n)} = [P^{n}]_{ij} = \mathbb{P}(X_{n+m}=j | X_{m}=i) 
$$
This follows because the probability of a sequence of transitions over n steps is the product of the individual transition probabilities, computed via matrix multiplication.

## 6. Chapman-Kolmogorov Equation

The n-step transition probabilities satisfy the Chapman-Kolmogorov equation, which describes how transitions over multiple steps can be composed.

**Theorem 1 (Chapman-Kolmogorov Equation)**:
For a time-homogeneous DTMC with transition matrix P, and for any $m,n\ge1$, the n-step transition probabilities satisfy:
$$ 
P^{m+n} = P^{m}P^{n} 
$$
Equivalently, for all states $i, j\in\mathcal{X}$:
$$ 
P_{ij}^{(m+n)} = \sum_{k\in\mathcal{X}} P_{ik}^{(m)} P_{kj}^{(n)} 
$$
**Proof**:
To go from state i to state j in $m+n$ steps, the chain must pass through some intermediate state k after m steps, then proceed from k to j in n steps. The probability of this path (for a specific k) is $P_{ik}^{(m)}P_{kj}^{(n)}$. The probability of first going from i to k in m steps is $P_{ik}^{(m)}$, and then from k to j in n steps is $P_{kj}^{(n)}$. Summing over all possible intermediate states k:
$$ 
P_{ij}^{(m+n)} = \sum_{k\in\mathcal{X}} P_{ik}^{(m)} P_{kj}^{(n)} 
$$
This is exactly the $(i,j)$-entry of the matrix product $P^{m}P^{n},$ since matrix multiplication is defined as:
$$ 
[P^{m}P^{n}]_{ij} = \sum_{k\in\mathcal{X}} [P^{m}]_{ik} [P^{n}]_{kj} 
$$
Thus, $P^{m+n}=P^{m}P^{n}$, proving the equation. 

The Chapman-Kolmogorov equation is fundamental for analyzing the long-term behavior of Markov chains, as it allows the computation of multi-step transition probabilities through matrix powers.

### 6.1 Example: Computing Multi-Step Transitions

Consider a simple random walk on $\{0, 1, 2\}$ with transition matrix:
$$ 
P = \begin{pmatrix} 0.5 & 0.5 & 0 \\ 0.4 & 0.2 & 0.4 \\ 0 & 0.5 & 0.5 \end{pmatrix} 
$$
To find the 2-step transition probabilities, we compute $P^{2}=P \cdot P:$
$$ 
P^{2} = \begin{pmatrix} 0.5 & 0.5 & 0 \\ 0.4 & 0.2 & 0.4 \\ 0 & 0.5 & 0.5 \end{pmatrix} \begin{pmatrix} 0.5 & 0.5 & 0 \\ 0.4 & 0.2 & 0.4 \\ 0 & 0.5 & 0.5 \end{pmatrix} = \begin{pmatrix} 0.45 & 0.35 & 0.2 \\ 0.36 & 0.38 & 0.26 \\ 0.2 & 0.35 & 0.45 \end{pmatrix} 
$$
For example, $P_{0,2}^{(2)}=0.2$ means that starting from state 0, the probability of being in state 2 after exactly 2 steps is 0.2.

**Question 1**: Consider a time-homogeneous DTMC with a finite state space. How can the Chapman-Kolmogorov equation be used to compute the probability of reaching a specific state after a large number of steps? What challenges arise if the state space is countably infinite?

**Solution**:
Consider a time-homogeneous Discrete-Time Markov Chain (DTMC) with a finite state space. The Chapman-Kolmogorov equation is a fundamental tool for understanding the evolution of the probabilities of being in different states over time. For a time-homogeneous DTMC with a transition probability matrix P, where $P_{ij} = \mathbb{P}(X_{n+1}=j|X_{n}=i)$ is constant for all n, the probability of being in state j after k steps, starting from state i, is given by the (i,j)-th entry of the matrix $P^{k}$, denoted as $P_{ij}^{(k)}$. This relationship is a direct consequence of the Chapman-Kolmogorov equation. Specifically, for any number of steps k and w, the $(k+w)$-step transition probabilities satisfy $P^{(k+w)} = P^{(k)}P^{(w)}$, and by induction, $P^{(k)}=P^{k}$.

To compute the probability of reaching a specific state j after a large number of steps n, we would calculate $P^{n}$ and examine the (i,j)-th entry, where i is the initial state. For some finite state Markov chains, as $n\rightarrow\infty,$ the distribution of states converges to a stationary distribution $\pi$. If such a distribution exists and is unique (e.g., for an irreducible, aperiodic, positive recurrent Markov chain), then the rows of $P^{n}$ will converge to $\pi$, meaning that $P_{ij}^{(n)}\rightarrow\pi_{j}$ as $n\rightarrow\infty,$ regardless of the initial state i. Thus, after a large number of steps, the probability of being in state j becomes approximately $\pi_{j}$.

However, if the state space is countably infinite, several challenges arise:

* **Infinite Dimensional Matrix**: The transition probability matrix P becomes infinitely dimensional. Standard matrix multiplication and calculating $P^{n}$ are no longer straightforward computational tasks. Analytical methods become necessary, and these can be significantly more complex.
* **Convergence of $P^{n}$**: The convergence of $P^{n}$ as $n\rightarrow\infty$ is not guaranteed. Even if it converges, the limit might not correspond to a stationary distribution in the same way as in the finite case.
* **Existence and Uniqueness of Stationary Distribution**: While a stationary distribution (a probability distribution satisfying $\pi P=\pi$) might exist for a countably infinite state space, its existence and uniqueness are not guaranteed. Finding such a distribution often involves solving an infinite system of linear equations, which can be difficult or impossible. Furthermore, even if a solution exists, it might not be a valid probability distribution (i.e., the components might not sum to 1).
* **Recurrence and Transience**: The concepts of recurrence (returning to a state infinitely often) and transience (visiting a state only finitely many times) become more nuanced in infinite state spaces. Determining whether a state is recurrent or transient is more challenging and crucial for understanding long-term behavior. For a transient state j, $\lim_{n\rightarrow\infty} P_{jj}^{(n)}=0.$ If all states are transient, the probability distribution over the state space might "drift to infinity".
* **Null Recurrence**: In addition to positive and transient states, countably infinite state spaces can have null recurrent states, which are recurrent (visited infinitely often) but have an expected return time of infinity. For such states, even if the chain returns to them, the long-run proportion of time spent in these states is zero.
* **Computational Issues**: Even when theoretical results about convergence exist, actually computing the long-term probabilities or analyzing the rate of convergence can be very difficult or require advanced analytical techniques.

In summary, while the Chapman-Kolmogorov equation still fundamentally governs the n-step transition probabilities in countably infinite DTMCs, the analysis of long-term behavior, including the existence and computation of limiting or stationary distributions, faces significant mathematical and computational hurdles compared to the finite state case. Concepts like recurrence, transience, and null recurrence play a more critical role in understanding the long-term dynamics.








