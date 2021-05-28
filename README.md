# Scheduling and Learning

## Research problems

Here we list some problem formulations:

* **Problem 0**:
   * Problem setting:
        * Multi-class, multi-server queueing control system 
        * Mean linear holding costs (c_i per unit time for class i)
        * Mean processing rates (mu_{i,j} for class i, server j)
        * Fixed set of servers, online job arrivals; may also consider having a dynamic set of servers
        * Unknown parameters c_i and/or mu_{i,j} and/or job class membership unknown
   * Note that there are two sources of information for learning
      * Information acquired per each time step about holding costs (e.g. through Bernoulli outcome observations)
      * Information acquired as jobs are assigned to servers (e.g. through Bernoulli success-failure outcomes as jobs are assigned to servers)
   * Special cases:
      * Knowlege about parameters:
         * c_i's unknown, mu_{i,j}'s known
         * c_i's known, mu_{i,j}'s unknown
         * both c_i's and mu_{i,j}'s unknown
      * Knowledge about class membership
         * Each job's class is unknown
         * Class of a job is unknown
         * Some prior information about job's class is known (e.g. random according to a prior distribution; see another example below)
   * We may think of w_{i,j} := c_i mu_{i,j} as a weight parameter for class-server pair i,j
   * Research question: algorithm design and establishing performance guarantees (e.g. expected payoff gap)
   * Methodology: Lyapunov drift method / regret type analysis
   * Existing work makes different assumptions about partial knowledge about unknown parameters
      * Each job's class is known but mu_{i,j} parameters are unknown (On Learning the cmu rule ...); c_i's are known
      * Partitioning of jobs is known such that each partition contains jobs of the same (unknown) class and sizes of these partitions are given as a parameter (Integrating online learning and adaptive control ...); product 
      * The general case, or other variants, seem open to be open problems
   * Additional remarks:
      * Paper "Integrating online learning and adaptive control ...) assumes each job processing takes a unit time and some (binary) payoff is observed; this can be interpretting allowing for task to be unsuccessfully processed; other uses cases are when there is some other notion of what it means for a job to be successfully resolved (e.g. for classification jobs)
      * Paper "Adaptive matching ..." assumes job classes are unknown, c_i's are equal to all 1 for all classes, and mu_{i,j} are assumed to be known 
      * Information acquired per time step may be complete or partial (e.g. because monitoring / testing requires resources) - we can be creative here
      * Interested in scalable algorithms that can scale well for large number of classes and/or servers

* **Problem 1**:
   * Problem setting:
        * Fixed set of arms, single server
        * Each arm state evolves according to a MC on a finite state space S of values in [0,1]
        * At each time step, the decision maker pulls an arm
        * At each time step
            * The decision maker observes a random cost for each arm, for arm a sampled from distribution p(.|s_a); e.g. assumed to be a Bernoulli random variable with mean s_a
            * The state of the pulled arm transitions to next state sampled from distribution \lambda (known to the decision maker)
            * The state of each other arm remains unchanged
        * The goal of the decision maker is to minimise the expected total cost of arms over a fixed time horizon; or, alternatively, a time discounted cost over an infinite time horizon.
   * Comments:
      * This is a POMDP problem
      * We may think of each arm to represent a job of unknown holding cost; upon completion of a job, this job gets replaced by a new job whose holding cost is sampled from S according to distribution \lambda
      * Variations:
         * Jobs have random processing times with means that may differ for different job classes
         * Open system - not fixing the number of jobs / arms
         * Multiple servers
      * Expected cost gap wrt a policy that knows all the parameters?
      * Intution: if job waiting time is sufficiently large (system load large enough) this can be exploited for learning (similar as in batch arrival model of Hsu et al - discussed below)
* **Problem 2**:
   * Problem setting:
      * Similar as in the scheduling and testing work but with more general assumptions about testing
      * Assume holding cost parameters to take one of two possible values {c_0, c_1}
      * Before a job is tested, the decision maker's belief about holding cost of this job is according to given prior distribution \lambda_0
      * After testing a job for the k-th time, the decision maker knows the posterior disribution \lambda_k of the holding cost of this job
      * Previously studied case is a special case where for k = 1, \lambda_k is fully concentrated on the true holding cost of a tested job
      * Assume that for each k, testing is for a fixed sample size, so for each k, there is a finite number of possible posterior distributions; assume Bernoulli experiment outcomes
      * What is the optimal policy in this case?
   * Comments:
      * If we assume that each job can be tested up to a fixed number of times, the information space (all possible posterior distributions) is a finite set
      * Note that the decision maker may never know the true value of the holding cost of a job
      * There is some similarity between this problem formulation and that in the paper by Shah et al (Adaptive matching ...) where the decision maker learns about job classes from observed _failures_ of servers to solve a job (one Bernoulli observation per server allocation), while in Problem 2 defined here, the decision maker observes a fixed number (one or more) Bernoulli outcomes per server allocation (for testing purposes)
* **Problem 3**:
   * Problem setting:
        * Multi-class, single-server queueing control system 
        * Mean linear holding costs (c_i per unit time for class i)
        * Unit processing time
        * Job arrival rate lambda
        * Unknown prior distribution of job classes
        * Unknown parameters c_i (Bernoulli random variables)
        * Unknown number of jobs in class i for i; one class corresponds to a queue, and therefore, queue lengths are unknown.
        * The total number of jobs in the system is unknown
        * If the server serves a class i job, it receives a realization of c_i.
        * If the server chooses class i but there is no class i job, it pays a penalty of p_i. 
   * Can we find a scheduling algorithm minimizing a regret?
   * Can we find a scheduling algorithm acheiving stability?
   * Comments
        * Serving class i jobs leads to learning the parameter c_i.
        * If serving a class i is unsuccessful because no class i job is present, then we use this information to learn the prior.
   * When we know the prior distribution and E[c_i] for all i exactly,
        * how does the cmu-rule work when queue lengths are unknown?
* ...

## Related work

* [On Learning the cmu Rule in Single and Parallel Server Networks](https://arxiv.org/abs/1802.06723), Krishnasaym, Aropstathis, Johari and Shakkottai, Allerton 2018.
   * Problem setting:
      * Standard multi-class job scheduling problem with linear holding cost functions, but unknown mean processing rate parameters
   * Summary of results:
      * Regret analysis with baseline cmu policy that knowns mean processing rate parameters
      * Shows that for single server case using empirical mean mu parameter estimators suffices
      * For multiple servers cases, shows that instability may occur, and proposes sufficient conditions for geometric ergodicity; morevoer, an almost greedy algorithm is proposed that explores only when the number of samples falls bellow a threshold
   * Comments:
      * Key assumption is that classes of jobs are known but some parameters of classes are unknown (mean processing rates)
* [Data-Driven Percentile Optimization for Multiclass Queueing Systems with Model Ambiguity: Theory and Application](https://pubsonline.informs.org/doi/abs/10.1287/ijoo.2018.0007), Bren and Saghafian, INFORMS Journal on Optimization, 2019.
   * Problem setting:
      * Multi-class job scheduling problem with unknown mean processing rate parameters
      * Studied as a POMDP problem
      * Robust control using an ambiguity set for prior distribution
   * Summary of results:
      * A percentile based policy
      * Needs a more careful reading to appreciate the results
* [Regret of Queueing Bandits](https://papers.nips.cc/paper/2016/file/430c3626b879b4005d41b8a46172e0c0-Paper.pdf), Krishnasamy, Sen, Johari and Shakkottai, NeurIPS 2016.
   * Problem setting:
      * Single queue, K servers, with unkown mean service rates \mu_k
      * At any time step, the queue can be served by at most one server
      * The problem is to understand how queueing affects the regret minimization in bandit algorithms
   * Summary of results:
      * tbd
* [Scheduling with testing](https://pubsonline.informs.org/doi/10.1287/mnsc.2017.2973), Levi, Magnanti and Shaposhnik, Management Science, 2018.
   * Problem setting:
      * Multi-class scheduling, fixed set of input jobs, all arriving at t = 0
      * Jobs have costant marginal holding costs, unknown
      * Jobs mean processing times, unknown
      * Single server
      * Server splits effort between two types of activities: (a) testing an unknown class job, (b) serving a job
      * Known prior distribution of job classes
      * Upon testing a job, the marginal holding cost and mean processing time of the job becomes known to the decision maker
      * The goal is to minimize the total holding cost (sum of products of holding cost parameters and completion times)
   * Summary of results:
      * Structural results about the optimal policy - not fully characterised
      * Dynamic programming formulation for state state of smaller dimension than an original dynamic programming formulation
      * PTAS algorithm for finding a solution
      * Sufficient conditions under which the optimal policy is fully characterised
   * Comments:
      * A POMDP problem, hidden state of each job i {unknown, known (c_i, \mu_i), served}, known prior
      * Special structure of MC transition probabilities
      * Special structure for partial observations: when state == unknown nothing is observed, when state == known (c_i, \mu_i) full observation of hidden state, and when state == served, marginal holding cost is zero
* [Patient Triage and Prioritization under Austere Conditions](https://pubsonline.informs.org/doi/10.1287/mnsc.2017.2855), Sun, Argon and Ziya, Management Science, 2017.
   * Problem setting:
        * Same as in the paper by Sun, Arghon and Ziya
   * Summary of results:
        * Complete characterizatio of optimal policy for the case of two job classes
* [Partially Observed Markov Decison Process Multiarmed Bandits--Structural Results](https://www.jstor.org/stable/40538383?seq=1#metadata_info_tab_contents), Krishnamurthy and Wahlberg, Mathematics of Operations Research, 2009.
   * Problem setting:
      * A fixed set of arms, arms' states evolve according to MCs such that at each decision time, one arm is pulled, the state of the pulled arm is updated according to the transition probability of this arm, and other arm's states remain unchanged
      * Partial bandit feedback - for the arm pulled, the decision maker observes a noisy observation of the state of this arm
      * Reward - random reward obtained for the arm pulled that is a function of the hidden state of the pulled arm
   * Summary of results:
      * POMDP problem formulation
      * Value iteration algorithm for computing Gittins index
      * Finite dimensional characterization of Gittins index
      * Sufficient conditions for optimality of an "opportunistic" policy - these conditions ensure an ordering of information states and the opportunistic policy choses the best information state with respect to this ordering
* [Regret Bounds for Restless Markov Bandits](https://www.sciencedirect.com/science/article/pii/S030439751400704X), Ortner, Ryabko, Auer, and Munos, Theoretical Computer Science, 2014.
   * Problem setting:
      * A fixed set of arms, each with state evolving according to an irreducible MC on a finite state space
      * For each state s of an arm i's MC, there is a reward distribution with mean r_i(s)
      * State transitions are unaffected by the action (which arm is pulled at a time step)
      * The decision maker knows neither the transition probabilities nor the mean rewards, but knows the cardinality of each MC state space
      * At each time step t, the decision maker pulls an arm a and observes the current state s of this arm and receives a random reward with mean r_a(s)
      * Note that perfect observation of the state of the pulled arm is observed
   * Summary of results:
      * Regret analysis with respect to optimal policy when the transition matrices and mean rewards are known
      * sqrt(T) regret bound with a T_mix^{3/2} factor where T_mix is the maximum mixing time of the MCs associated with arms
   * Comments:
      * Different from the scheduling POMDP we discussed in that states are not affected by actions
      * Accomodates a scheduling problem where an arm's state corresponds to the hidden state of a job associated with this arm; state of an arm changes upon job completion and replacement of this job with a new job associated with this arm; may think of this as a case where a single server is allocated to each job in the system
      * The goal is to pull high rewarding arms - may think of this as a top-1 selection in a time-changing environment
      
* [Learning to Control Renewal Processes with Bandith Feedback](https://semihcayci.github.io/SIGMETRICS19_p43.pdf), Cayci, Erylmaz, and Srikant, Sigmetrics 2019.
   * Problem setting:
      * A fixed set of arms, at each decision time one arm pulled
      * Upon pulling arm a, the decision maker needs to wait for random time X_a and observe reward R_a at the end
      * The waiting times and rewards are independent across arms and across pulls
      * Waiting for an arm to "complete" can be interrupted at the cost of not receiving reward for this arm pull
      * The goal is to maximize reward acrued over a fixed time horizon (knapsack-type problem)
   * Summary of results:
      * Algorithm design
      * Regret analysis
* [Adaptive Matching for Expert Systems with Uncertain Task Types](https://pubsonline.informs.org/doi/10.1287/opre.2019.1954), Shah, Gulikers, Massoulie, and Vojnovic, Operations Research 2020.
   * Problem setting:
      * Multi-class jobs, finite set of servers
      * Server s succeeds in solving task of class c with probability p_{c,s}
      * Each server s attempt to solve the allocated task takes a random time with Exp(1) distribution, independent across different allocations of jobs to servers 
      * Class of an incoming job is unknown to the decision maker
      * The decision maker knows matrix P = (p_{c,s})
      * At each instance at which a server becomes free, the decision maker needs to decide which job to assign to the server
      * The system may be seen as a system with pre-emption allowing to allocate a particular job to one or more server until the job is resolved
      * The learning of the job class is from observed failures of servers to resolve this job
   * Results:
      * Backpressure mechanism shown to be "stability optimal" - supports any vector of input arrival rates in the stability region
      * Standard Lyapunov function analysis with a novel bit in using a trancation of the information state space (posterior distribution of job classes) - this needed to apply the Lyapunov function arguments for a finite state space
   * Comments:
      * Open problem: job holding cost optimal policies?
* [Integrating Online Learning and Adaptive Control in Queueing Systems with Uncertain Payoffs](https://engineering.purdue.edu/~linx/paper/tech-queueing-learning17.pdf), Hsu, Xu, and Bell, Operations Research, to appear; conference version [here](https://ieeexplore.ieee.org/document/8503124)
   * Problem setting:
      * Discrete time, at each time step a batch of jobs arrive (belonging to a "client"), arrivals independent over time steps
      * Each batch arrival consists of a random number of jobs according to a geometric distribution (with mean N)
      * lambda = rate of job arrivals, lambda / N = prob of new client arrival in a time step
      * Each client belong to one of I classes, rho_i = probability that an arriving client is of class i
      * There are J servers
      * Payoffs: Ber(C^*_{i,j}) for a class-i job served by server j
      * Unknown parameters:
         * lambda
         * rho distr
         * N
         * each client's class
         * expected payoffs C^*_{i,j}
      * Objective: maximise long-term average payoff
      * Note: classes of jobs are unknown, but what is known is a partitioning of jobs (one partition per client) such that each job in a partition belongs to the same job class; the expected sizes of these partitions are equal to N; intuitively, for large N the decision maker can learn how to best assign jobs of a client from observed rewards when assigning jobs to servers
   * Summary of results:
      * Mean rate stability (Thm 1)
      * Algorithm that combines learning with optimisation
         * Basic idea: server j samples mu_j jobs in each time step, sampled job is of client l with probability p_j^l/mu_j where p_j^l is increasing in an UCB estimate of the cost of serving a job by this server (c_j^l(t) in [0,1]). More precisely, p_j^l(t) = (1/V) / (gamma-c_j^l(t) + delta) where gamma > 1 is a parameter and delta >=0 is the Lagrange multiplier with value such that \sum_l p_j^l(t) = mu_j.
         * The key is to learn a class-server specific parameter of a job by learning; this is allowed if the number of jobs per client is large enough (large mean N parameter
      * Payoff gap bound with respect to an optimal solution with known parameters (Thm 2): O(1/V + sqrt(log(N)/N)); the term V is common to Lyapunov-drift type methods 
      * The larger the value of V, the smaller the payoff gap (but this is at the expense of higher system load) - similar to the Lyapunov-drift method
      * The larger the value of N, the smaller the payoff gap (naturally so because more learning is allowed per client)
* [Matching while learning](https://arxiv.org/abs/1603.04549), Johari, Kamble and Kanoria
   * Problem setting:
      * A fixed unit mass of workers, q \in [0,1] represents a worker
      * Job classes are known (fixed finite set of job classes)
      * Worker classes are unknown (fixed finite set of worker classes)
      * Workers arrive and depart but the mass of workers remains finite at all times - each worker regenerates after every N time periods (N is the lifetime of a worker)
      * Each worker has the opportunity to do at most one job during each time period of their lifetime
      * Pyaoffs: matching a worker of type i with a job of type j results in a Bernoulli reward with success probability A(i,j)
      * Objective: maximize the rate of payoff generation
   * Summary of results:
      * Characterization of the structure of the optimal policy in the limit when each worker performs many jobs
      * Exploration vs exploitation: shadow price maintaining for each job class, and payoffs adjusted by these prices
   * Comments:
      * The role of N is similar to the role of N in Hsu et al
      * Workers correspond to clients in Hsu et al; not assumed to be a fixed mass in Hsu et al 
* Perhaps of interest [Proportional Allocation: Simple, Distributed, and Diverse Matching with High Entropy](http://proceedings.mlr.press/v80/agrawal18b.html), Agarwal et al, ICML 2018
* [Regret Bounds for Thomson Sampling in Episodic Restless Bandit Problems](https://papers.nips.cc/paper/2019/file/2edfeadfe636973b42d7b6ac315b896c-Paper.pdf), NeurIPS 2019.
   * Problem setting:
      * K arms, decision maker selects N arms in each time step
      * Each arm a has binary state in {0,1} that evolves as a MC with transition matrix P_a^x, x \in {0,1} where x indicates whether arm is pulled
      * At time step t pulling an arm a incurs a binary reward X_{t,a} which is the arm's current state
      * Bandit setting - the decision maker only observes the rewards of pulled arms, POMDP for this reason
      * Resets - episodes of L time steps (L is known to the decision maker); states of arms are reset at the beginning of each episode ro random states drawn from a known distribution
    * Summary of results:
      * Studies Thomson sampling algorithm
      * Bayesian regret bound
    * Comments:
       * Similar problem as in "Regret Bounds for Restless Markov Bandits" 
          * Some differences: (a) MC transition probabilities are allowed to be controlled by actions (arm pulls), (b) MC state restricted to {0,1}, (c) more general N>=1 case considered 
       * Relations with Problem 1: 
          * Think of N as the number of servers; each arm pulled corresponds to serving a job; arm state indicates whether the job represented by this arm is low or high reward; reward earned only by pulling a high-reward job
          * P_a^x can be chosen such that they model state change at new job entry; an arm retains the same job if not pulled, if an arm is pulled a new job is associated with this arm with randomly sampled job class
          * Resets change job classes
          * Unlike to Problem 1: reward accrued only for pulled arms - the goal is pull high-reward jobs
 * [Restless-UCB, an Efficient and Low-complexity Algorithm for Online Restless Bandits](https://proceedings.neurips.cc/paper/2020/file/89ae0fe22c47d374bc9350ef99e01685-Supplemental.pdf), NeurIPS 2020.
   * Problem setting:
      * A fixed set of arms, each with state evolving according to an ergodic MC.
      * All the Markov chains have the same finite state space S={1,2, ...., M}.
      * All the Markov chains have a birth-death structure.
      * The transition matrix of each Markov chain is fixed and independent of actions taken. 
      * The random reward of pulling arm i at state s is given by r_i(s). 
      * If s_1 > s_2, then r_i(s_1) > r_i(s_2).
   * Summary of results:
      * UCB type sampling algorithm, which consists of the exploration phase and the exploitation phase.
      * After the exploration phase, the decision maker estimates the transition matrices of the Markov chains and the state-dependent reward distributions.
      * For the exploitation phase, the decision maker solves the MDP obtained by the estimated distributions, which approximates the true problem. Then the algorithm commits to the optimal policy for the approximate MDP.
* [Solving convex optimization with side constraints in a multi-class queue by adaptive cmu rule](https://link.springer.com/article/10.1007/s11134-013-9377-3), Li and Neely, Queueing Systems, 2014
   * Problem setting:
      * M/G/1 queue with multple classes
      * Several performance objectives concerned with optimising per-class average queueing delays
      * "Second problem": optimising sum of convex functions of average delays across classes, subject to average delay of each class being within a specified bound, and the vector average delays across classes being in the stability region
   * Summary of results:
      * Adaptive cmu rule policies
      * For Second problem, serving jobs according to priority indices of the form (dynamic parameter) x mu 
      * The policy makes use of regeneration between busy cycles
      * Long-term performance bounds - optimal cost plus an additive term that scales with a parameter V
      * The larger the value of V the bound is closer to optimal, but this is at the expense of slower convergence
* [Robust Queueing Theory](https://pubsonline.informs.org/doi/10.1287/opre.2015.1367), Bandi, Berstimas and Youssef, Operations Research, 2015

* [Diagnostic Accuracy Under Congestion](https://pubsonline.informs.org/doi/10.1287/mnsc.1120.1576), Alizamir, de Véricourt and Sun, Management Science, 2013
  * Problem setting:
    * Two classes of jobs, but the classes of jobs are unknown.
    * Jobs arrive according to a Poisson process with rate 􏰒$\lambda$.
    * A single server tests a job, possibly multiple times, before it classifies the job and the job leaves the system.
    * Each test takes time exponentially distributed with rate $\mu$. 
    * The prior distribution of job classes is known, and the posterior distribution of the job being tested is updated after each test.
    * The single server can move on to the next job only after the current job is tested and classified. 
    * There exist the reward and misclassification cost associated associated with each class, 
      * e.g., when (1) the probability of a job in class 1 is $p$ and 
      * (2) the reward and misclassification cost associated with class 1 is $r_1$ and $c_2$, 
      * the expected profit from labelling the job as class 1 is $p r_1 - (1-p) c_2$.
    * There is a holding cost $c_w(x)$, where $x$ denotes the number of jobs in the system, per each time unit.
    * The objective is to maximize the long-run average profit.
  * Summary of results:
    * POMDP formulation.
    * The "structure" of the optimal policy: continue testing if the posterior probability $p$ is between two thresholds $\underline p(x, k)$ and $\bar p(x, k)$ where x is the number of jobs in the system and $k$ is the number of tests done for the job.
    * For the special case of identical tests, the thresholds are invariant for different $k$'s.
    * In either case, there is no characterization of the values $\underline p(x, k)$ and $\bar p(x, k)$.
    * Looked at the special case of no false negative (the one-sided case: any test result indicating class 2 has no error).
  * Connections to the scheduling and testing problem:
    * "Problem 2" is in similar spirit in this problem.
    * The objective of "Problem 2" is to minimize the total weighted holding cost. 
    * In "Problem 2", when a job is classified as high-priority class, the job is served and leaves the system. However, when a job is classified as low-priority class, the job remains in the system and incurs holding costs until it is served later.

