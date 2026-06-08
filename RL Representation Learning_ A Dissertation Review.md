## **Hierarchical Divergence, Plasticity Loss, and Representation Geometry in Deep Reinforcement Learning** 

## **1. Introduction** 

The development and evolution of internal representations within deep reinforcement learning (RL) agents constitute one of the most intricately complex domains in contemporary artificial intelligence research. The canonical architecture introduced by Mnih et al. in 2015 at the venue _Nature_ established a standard for processing high-dimensional visual environments such as the Arcade Learning Environment (ALE) Atari 2600 benchmark.[1] This architecture typically utilizes a shared convolutional backbone consisting of sequential convolutional layers (Conv1 → Conv2 → Conv3) that feed into a fully connected representation layer (fc_repr, often 512-dimensional), which subsequently maps to action-value (Q) estimates.[1] While this structural paradigm has proven remarkably successful in approximating optimal control policies directly from pixel data, the internal geometric topology, the optimization dynamics, and the multi-task adaptability of these networks exhibit profound vulnerabilities. 

Recent empirical evaluations of Deep Q-Networks (DQN) and Double DQN (DDQN) agents trained on dimensionally similar yet functionally distinct Atari games—specifically Pong and Breakout—reveal a confluence of pathological behaviors when subjected to specific training regimes. These behaviors manifest across several interconnected dimensions. First, there is a severe hierarchical divergence in representational similarity across the network depth. Second, agents exhibit catastrophic forgetting under sequential fine-tuning paradigms, wherein newly acquired policies destructively overwrite established ones. Third, an aggressive capacity loss characterized by high fractions of dormant or "dead" neurons emerges, leading to complete task failure during interleaved multi-task training. Finally, structural discrepancies arise based on the specific algorithmic variant utilized, with DDQN demonstrating significantly superior representational geometry compared to vanilla DQN. 

This comprehensive research report provides an exhaustive, mechanistic analysis of these phenomena. It synthesizes the extant literature across five core domains: (1) representation similarity and hierarchical divergence in deep neural networks, comparing RL dynamics to supervised learning benchmarks; (2) catastrophic forgetting in RL agents and the efficacy of continual learning mitigations such as continual backpropagation; (3) capacity loss, the dormant neuron phenomenon, and the geometric constraints of feature superposition; (4) destructive gradient interference in multi-task optimization and the failure of shared representations; and (5) the structural impact of value overestimation on representation geometry. By bridging empirical observations on the Atari benchmark with theoretical frameworks—ranging from transient non-stationarity to Neural Tangent Kernel (NTK) rank collapse—this analysis offers a highly nuanced understanding of how deep RL agents construct, maintain, and frequently destroy their internal spatial representations. 

## **2. Representation Similarity and Hierarchical Divergence** 

Empirical analyses utilizing Centered Kernel Alignment (CKA) to measure cross-game representational similarity across identically architected agents trained independently on Pong and Breakout yield a striking structural pattern. Early convolutional layers exhibit near-perfect similarity, with CKA values approximating 0.9 at Conv1 and Conv2. However, the terminal fully connected representation layer (fc_repr) diverges almost entirely, registering a CKA of approximately 0.09. This stark hierarchical divergence necessitates a precise mechanistic explanation, rooted in both the foundational supervised deep learning literature and the unique, value-driven representation geometry specific to reinforcement learning. 

## **2.1 Consistency with Supervised Deep Learning Paradigms** 

The observation that early layers of a deep neural network remain highly generalizable while deeper layers become increasingly task-specific is a foundational tenet of deep learning. This phenomenon was first rigorously quantified in supervised settings by Yosinski et al. in their seminal 2014 paper, _How Transferable are Features in Deep Neural Networks?_ , presented at _NIPS_ .[4] In their exhaustive study on the transferability of features, Yosinski et al. demonstrated that the first layer of a deep convolutional network almost universally learns general visual primitives, such as Gabor filters and color blobs, regardless of the target dataset.[6 ] 

As the network depth increases, the features undergo a phase transition from general to specialized. The authors identified a phenomenon termed "fragile co-adaptation," wherein neurons in middle and higher layers co-adapt to process the specific semantic structures of the training data.[4] If these layers are frozen and transferred to a new task, the network suffers a precipitous drop in performance because the higher-layer features have specialized to the original task at the expense of general utility.[5 ] 

In the context of the ALE Atari benchmark, the games Pong and Breakout share highly similar low-level visual semantics. Both games feature a static black background, high-contrast moving geometric shapes, and distinct paddle-and-ball physics dynamics. Consequently, the Conv1 and Conv2 layers learn a generalized physics and rendering engine for 2D Atari environments—specializing in edge detection, motion tracking, and background subtraction. Because these low-level visual requirements are nearly identical across both games, the resulting feature spaces are highly congruent, producing the observed CKA of 0.9. However, the severe divergence observed at the fc_repr layer directly mirrors the findings of Yosinski et al. regarding deep task specialization.[4] The terminal layers are tasked with mapping these general visual primitives into a specific semantic space that facilitates decision-making. Because the high-level semantic objectives of Pong (calculating the trajectory to bypass an opponent's paddle) and Breakout (calculating angles to chip away specific sections of a brick wall) are fundamentally distinct, the resulting feature spaces must be mathematically orthogonal, yielding the observed CKA collapse to 0.09. 

## **2.2 RL-Specific Representation Geometry and Value-Driven State** 

## **Aggregation** 

While the supervised learning literature explains the transition from visual primitives to semantic features, the extreme severity of the divergence at the fc_repr layer is uniquely exacerbated by the reinforcement learning paradigm. In supervised image classification, the penultimate layer must linearly separate classes based on morphological traits. In deep RL, the fc_repr layer is the penultimate stage of the value function approximation; it must warp the state space to linearly separate expected future rewards. 

Zahavy et al. provided a definitive analysis of this phenomenon in their 2016 paper, _Graying the Black Box: Understanding DQNs_ , presented at _ICML_ .[8] By extracting the high-dimensional activations of the fc_repr layer from agents trained on Atari games and mapping them to a lower-dimensional space using the t-SNE algorithm, Zahavy et al. demonstrated that DQNs aggregate the state space in a strictly hierarchical, value-driven fashion.[8 ] 

In their empirical evaluations of games such as Breakout and Pacman, the t-SNE clusters formed by the DQN corresponded to distinct sub-manifolds of the state space governed by expected reward and specific temporal termination rules.[11] For example, in Breakout, states were clustered not merely by the visual location of the ball, but by whether the agent had successfully carved a tunnel through the bricks, as this event drastically alters the expected future reward trajectory.[12 ] 

Therefore, the geometry of the fc_repr layer is not merely a visual summary; it is directly molded by the topology of the underlying Markov Decision Process (MDP) and the agent's expected future return (Q-values).[13] Because Pong and Breakout possess mutually exclusive reward structures, disparate transition dynamics, and unique optimal policies, their underlying value landscapes are fundamentally incompatible. The fc_repr layer must spatially warp the raw visual features into a geometry that allows the final linear layer to output accurate Q-values for that specific MDP. This value-driven spatial warping guarantees that the final representations will be geometrically orthogonal across disparate environments, fully explaining why the hierarchical divergence in RL is so absolute. 

## **3. Catastrophic Forgetting and Transient Non-Stationarity in RL** 

The fragility of these specialized representations is starkly illustrated during sequential training. When a Pong-trained agent is fine-tuned on Breakout, Pong performance collapses rapidly (e.g., from an optimal +9 to a failing -21 within 200,000 steps). Empirical ablations demonstrate that freezing the entire network (both the convolutional layers and the fc_repr layer) prevents this forgetting, whereas freezing only the convolutional layers still causes catastrophic forgetting. This indicates that the fc_repr layer aggressively adapts to the Breakout value landscape, inherently and destructively rewiring the Pong policy in the process. 

## **3.1 The Mechanics of RL Forgetting** 

Catastrophic forgetting occurs when a neural network overwrites parameters that are highly relevant to a previously learned task in order to minimize the loss gradients associated with a 

new task. In deep reinforcement learning, this phenomenon is severely exacerbated by the non-stationary nature of the data distribution. Unlike supervised learning, where the dataset is fixed and identically distributed, an RL agent's experience replay buffer shifts continuously as its policy evolves.[15] Furthermore, value-based methods suffer from target non-stationarity due to the use of bootstrapping against a shifting target network.[16 ] 

The observation that freezing the convolutional backbone is insufficient to prevent forgetting aligns perfectly with the hierarchical divergence discussed in Section 2. Because the fc_repr layer is uniquely responsible for mapping abstract, general visual features into a highly specific action-value topography, overwriting its parameters fundamentally scrambles the policy mapping for the original task. The new Breakout gradients violently alter the weights of the fc_repr layer to form new t-SNE sub-manifolds relevant to brick destruction, completely destroying the geometric clusters previously organized around paddle defense. 

## **3.2 Continual Learning Mitigations** 

The continual learning literature has proposed several architectural and regularization-based mitigations to preserve knowledge across sequential tasks. However, their efficacy in the highly volatile domain of reinforcement learning varies significantly. 

|**Mitigation Strategy**|**Mechanism of Action**|**Efficacy and Limitations in**<br>**RL Context**|
|---|---|---|
|**Elastic Weight**<br>**Consolidation (EWC)**|Computes the Fisher<br>Information Matrix after<br>training on a task to identify<br>weights crucial for that<br>task. It then applies an<br>penalty to constrain<br>modifications to those<br>specific parameters during<br>subsequent training.17<br>L,|Moderate. EWC slows<br>forgetting but frequently<br>fails in RL due to the highly<br>volatile gradient landscape.<br>The shifting scale of<br>Temporal Difference (TD)<br>errors can misalign the<br>Fisher estimates, leading to<br>inadequate parameter<br>protection.19|
|**PackNet**|Iteratively trains the<br>network, prunes a specific<br>percentage of less<br>important weights, and<br>freezes the remaining<br>highly active subset for<br>each task. It uses a binary<br>mask to allocate network<br>capacity to specific tasks.17|High. PackNet effectively<br>eliminates catastrophic<br>forgetting by strictly<br>partitioning the network<br>structurally. However, it<br>permanently restricts the<br>capacity available for<br>subsequent tasks,<br>eventually halting learning|



|||when the network flls.18|
|---|---|---|
|**Progressive Networks**|Completely freezes the<br>base network afer a task is<br>learned and spawns lateral<br>connections to a newly<br>initialized set of parameters<br>for the new task.17|High. Progressive networks<br>guarantee perfect retention<br>by never altering old<br>weights. However, they are<br>computationally expensive,<br>as inference cost and<br>parameter count scale<br>linearly with the number of<br>tasks, making them<br>unscalable for long<br>continual RL sequences.18|



## **3.3 Transient Non-Stationarity and Continual Backpropagation** 

Beyond the simple overwriting of parameters, deep RL networks suffer from unique optimization pathologies. Igl et al., in their 2021 paper _Transient Non-stationarity and Generalisation in Deep Reinforcement Learning_ at _ICLR_ , identified that RL networks suffer from a persistent "memory effect".[21] During early exploration, the agent encounters highly skewed, non-stationary data distributions. Igl et al. demonstrated that even after the data distribution stabilizes, the network's latent representation remains permanently warped by these early, noisy gradient updates, which permanently degrades generalization performance.[22] To counter this, they proposed Iterated Relearning (ITER), an algorithm that repeatedly distills the current policy into a freshly initialized network to shed this accumulated geometric distortion.[21 ] More recently, research has shifted toward algorithmic interventions that maintain plasticity continuously. Elsayed and Mahmood's 2024 paper at _ICLR_ , _Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning_ , alongside the 2024 _Nature_ publication by Dohare et al., _Loss of plasticity in deep continual learning_ , introduced Continual Backpropagation as a profound mitigation strategy.[24 ] 

Traditional gradient descent naturally causes units to become rigid over sequential tasks, leading to both catastrophic forgetting of old tasks and an inability to learn new ones.[27] Continual Backpropagation addresses this by tracking the historical utility of individual neurons.[28] Elsayed and Mahmood utilize Utility-based Perturbed Gradient Descent (UPGD), which applies smaller gradient modifications to highly useful units—thereby protecting them from catastrophic forgetting—while simultaneously injecting larger random perturbations into low-utility units, effectively rejuvenating their plasticity.[24] As Dohare et al. articulate, sustained deep learning in streaming and continual RL settings requires a persistent, random, non-gradient component to maintain structural variability and prevent the network from becoming permanently trapped in task-specific local optima.[26 ] 

## **4. Dead Neurons, Capacity Loss, and Superposition** 

Empirical tracking of network health reveals a chronic, debilitating rise in dormant (or "dead") neurons. A standard DQN agent trained on a single Atari task typically exhibits an inherently high dead neuron fraction of approximately 87%. Under sequential training across multiple tasks, this fraction edges upward to 88%. Most alarmingly, when an agent is subjected to interleaved, simultaneous multi-task training on both Pong and Breakout, the dead neuron fraction skyrockets to 92%, resulting in complete task failure where neither game is learned. 

## **4.1 The Dormant Neuron Phenomenon in DQN** 

The high baseline of dead neurons in DQN agents was formally characterized by Sokar et al. in their 2023 _ICML_ paper, _The Dormant Neuron Phenomenon in Deep Reinforcement Learning_ .[30] A neuron is classified as dormant if its normalized activation score—defined as the absolute value of its activation relative to the average absolute activation of its respective layer—falls 

below a specific threshold .[30 ] Sokar et al. rigorously isolated the root cause of this phenomenon: target non-stationarity induced by the RL bootstrapping process.[15] In supervised learning with fixed classification targets, the number of dormant neurons generally decreases over the course of training.[16] However, in value-based RL algorithms like DQN, the target values are constantly moving due to Bellman updates and periodic target network synchronization.[16 ] 

This relentless non-stationarity causes the feature representations to churn violently. Neurons that initially learned useful features are repeatedly subjected to conflicting gradient signals as the value estimates shift, driving their pre-activation distributions into the negative regime. Because the DQN architecture relies heavily on Rectified Linear Unit (ReLU) activations, these neurons fall below zero, output zero activation, cease passing gradients backward, and become permanently dormant.[16] To combat this, Sokar et al. proposed "ReDo" (Recycling Dormant neurons), an algorithm that periodically identifies and reinitializes dormant units to maintain network expressivity throughout training.[15 ] 

## **4.2 Disentangling Plasticity Loss: A Theoretical Contradiction** 

A critical finding in the presented empirical setup posits a direct causal chain: sequential and interleaved training causes ReLU saturation (dead neurons), which in turn causes capacity loss and task failure. However, an extensive body of literature by Lyle et al., published across _ICLR_ (2022), _ICML_ (2023), and _ArXiv_ (2024), presents a highly nuanced contradiction to this assumption.[34 ] 

**Contradiction Flag:** While the empirical observation of a 92% dead neuron fraction during multi-task failure is accurate, the literature explicitly refutes the idea that dead neurons are the _sole or primary_ driver of capacity loss. Lyle et al. systematically demonstrate that **loss of plasticity occurs even in the complete absence of saturated units** .[34 ] 

Instead of attributing plasticity loss solely to the dormant neuron phenomenon, Lyle et al. (2023, 2024) attribute it to the rank collapse of the Neural Tangent Kernel (NTK) Gram matrix 

and the severe decay of gradient magnitudes.[39] Training on non-stationary targets 

flattens the curvature of the loss landscape, trapping parameters in sharp, suboptimal local minima where gradients cannot propagate effectively, entirely independent of whether the ReLUs are active or dead.[34 ] 

Therefore, while the 92% dead neuron fraction observed during interleaved training is undoubtedly fatal to the agent's performance, it must be viewed mechanistically as a _symptom_ of a broader pathological learning dynamic rather than the fundamental root cause. To mitigate this deep geometric collapse, Lyle et al. (2024) demonstrated that intervening on multiple mechanisms simultaneously is necessary. Specifically, combining Layer Normalization (to prevent feature norms from growing unboundedly) with Weight Decay (to preserve parameter mobility) is highly effective at maintaining network trainability without relying on architectural resets or neuron recycling.[36] Additionally, Lyle et al. (2022) proposed Initial Feature Regularization (InFeR), an auxiliary loss that regresses a subspace of features toward their values at initialization, thereby tethering the representation to a high-rank state and preventing capacity collapse.[35 ] 

## **4.3 Toy Models of Superposition and Multi-Task Collapse** 

To thoroughly understand why the interleaved multi-task training paradigm forces the dead neuron fraction to an astronomical 92%, one must consult the theoretical framework established by Elhage et al. in their 2022 paper, _Toy Models of Superposition_ , published in _Transformer Circuits_ .[43 ] 

Elhage et al. demonstrated that neural networks frequently pack a multitude of unrelated concepts into a single neuron—a phenomenon known as polysemanticity—by storing sparse features in geometric superposition.[44] Whether a network stores features orthogonally (monosemantically) or in superposition is governed by a strict mathematical "phase change" dictated by feature sparsity and feature importance.[44 ] 

In a single-task setting (e.g., training exclusively on Pong), the network can comfortably allocate its limited 512 dimensions in the fc_repr layer to orthogonal, monosemantic representations of the most critical state features (e.g., paddle height, ball velocity).[44] However, under the interleaved training paradigm (alternating between Pong and Breakout every 1,000 steps using a shared backbone), the network is bombarded with two entirely distinct, high-importance feature sets. 

The network is rapidly forced past the phase change threshold, attempting to cram an excessive number of conflicting features into the limited 512-dimensional space using superposition.[44] Because Pong and Breakout features are dense and not mutually sparse, this forced superposition creates massive activation interference. The ReLU activation function, which normally acts as a non-linear filter to resolve minor superposition interference, is completely overwhelmed.[44] To minimize the chaotic, destructive interference cascading into the Q-value output heads, gradient descent takes the path of least resistance: it drives the incoming weights of the interfering neurons to zero, heavily and permanently saturating the ReLUs. Thus, the 92% dead neuron fraction is not merely an accident; it is the network aggressively pruning itself to silence the unresolvable interference caused by forced multi-task superposition. 

## **5. Destructive Gradient Interference in Multi-Task RL** 

The complete failure of the interleaved training setup—wherein neither task is learned despite the utilization of equal gradient signals, separate output heads, and separate replay buffers—highlights a critical limitation in multi-task representation learning: gradient interference. 

## **5.1 The Failure of Shared Representations** 

The hypothesis that a shared convolutional backbone will naturally discover universally applicable visual features (as it does in multi-task supervised image classification) fails catastrophically in reinforcement learning due to the high variance and conflicting directionality of the gradients. 

When the agent executes a training step on a batch of Pong experiences, the gradient vector 

pushes the shared parameters to extract features relevant to vertical paddle movement and horizontal ball velocity. Conversely, on the subsequent Breakout batch, 

pushes the exact same parameters to extract features relevant to grid destruction and angular ricochets. Because the optimal feature manifolds for these two MDPs are distinct, the cosine similarity 

between their gradient vectors is frequently negative ( ). This results in destructive gradient interference.[47] The network oscillates endlessly between the two tasks, perpetually overwriting the progress made in the previous 1,000 steps. The parameters become trapped in a steep, narrow valley of the loss landscape; the gradient of one task overshadows the other, causing the optimizer to jump back and forth between the walls of the valley without making any forward progress along the floor.[48] Providing separate output heads is insufficient because the shared fc_repr and convolutional layers beneath them are constantly being torn in opposing geometric directions. 

## **5.2 Principled Optimization Approaches** 

To overcome the catastrophic failure of naive shared representations, researchers have proposed advanced gradient surgery and multi-objective optimization techniques designed to enforce Pareto optimal parameter updates: 

|**Algorithm**|**Mechanism for**<br>**Deconflicting Gradients**|**Outcome and Efficacy in**<br>**RL**|
|---|---|---|
|**PCGrad**(Projecting<br>Conflicting Gradients)|Introduced by Yu et al.<br>(2020) at_NeurIPS_. If the<br>cosine similarity between<br>two task gradients is|Effectively removes the<br>destructive, conflicting<br>component of the update,<br>ensuring that optimizing|



||negative, PCGrad projects<br>the gradient of each task<br>onto the normal plane of<br>the other task's gradient<br>before computing the fnal<br>update.47|one task does not<br>mathematically degrade<br>the representation of the<br>other. Highly efective in<br>multi-task RL.47|
|---|---|---|
|**MGDA**(Multiple Gradient<br>Descent Algorithm)|Frames multi-task learning<br>strictly as a multi-objective<br>optimization problem,<br>seeking a common descent<br>direction that guarantees<br>local improvement for all<br>tasks simultaneously.48|Converges to a Pareto<br>stationary point. While it<br>mathematically guarantees<br>no negative transfer, it is<br>computationally heavy and<br>ofers no explicit control<br>over which point on the<br>Pareto front it converges<br>to.48|
|**CAGrad**(Confict-Averse<br>Gradient Descent)|Enhances optimization by<br>fnding the best update<br>vector within a specifed<br>neighborhood of the<br>average gradient that<br>maximizes the worst-case<br>local improvement across<br>all tasks.48|Regulates the minimum<br>decrease across tasks,<br>ofen outperforming<br>PCGrad in highly misaligned<br>RL environments by<br>explicitly preventing one<br>task from dominating the<br>optimization trajectory.51|



While these optimizers can mathematically resolve gradient conflicts, relying solely on optimization surgery may be insufficient if the network lacks the fundamental structural capacity to route the conflicting information. Introducing task-conditioned networks—such as appending a task-specific one-hot vector directly to the fc_repr layer, or utilizing Feature-wise Linear Modulation (FiLM) layers within the convolutional backbone—allows the network to functionally decouple the shared representation based on the current task without physically splitting the parameters. This structural flexibility can potentially avoid the superposition collapse entirely by dynamically altering the active sub-network for each game. 

## **6. DQN vs. DDQN: The Impact of Value Overestimation on Representation Geometry** 

An architectural comparison between vanilla DQN and Double DQN (DDQN) agents yields a fascinating observation regarding internal representation geometry: despite possessing identical layer architectures, DDQN produces tighter, more structured t-SNE clusters in the fc_repr layer and maintains a significantly lower fraction of dead neurons. Because the forward-pass architectures are identical, this profound discrepancy is rooted entirely in how their respective loss functions sculpt the optimization landscape. 

## **6.1 The Mechanics of Overestimation Bias** 

The vanilla DQN algorithm, as introduced by Mnih et al. (2015), utilizes a standard Q-learning 

target derived from the Bellman equation: .[1] Because the neural network's Q-value estimates are inherently noisy and imperfect during early training, 

the operator systematically selects the highest noisy estimate.[54] As rigorously demonstrated by van Hasselt et al. in their 2016 _AAAI_ paper, _Deep Reinforcement Learning with Double Q-learning_ , this maximization bias compounds iteratively over time, leading the network to learn unrealistically high and highly unstable action values.[54 ] 

DDQN mitigates this critical flaw by decoupling action selection from action evaluation. It uses the online network to select the maximizing action, but relies on the target network to evaluate 

the value of that action: .[53] This simple mathematical modification drastically reduces overestimation bias, leading to more conservative, mathematically stable value estimates.[57 ] 

## **6.2 Representational Consequences of the Double Q-Learning Fix** 

The downstream effect of correcting this overestimation bias profoundly alters the internal representation geometry. In vanilla DQN, the compounding overestimation creates massive, sudden spikes in the Temporal Difference (TD) error.[57] These spikes translate into explosive, high-variance gradients that propagate backward from the Q-value output heads, through the fc_repr layer, and into the convolutional backbone. 

This high-variance gradient environment has two highly destructive effects. First, it completely shatters the geometric coherence of the latent space. As the network violently and rapidly adjusts its parameters to fit the artificially inflated Q-targets, the points in the latent space are scattered chaotically, leading to the loose, disorganized t-SNE clusters observed empirically.[58] Second, these explosive gradients push the weights of the network into extreme values. In a network utilizing ReLU activations, sudden, massive negative weight updates drive the pre-activation values deeply into the negative domain. As previously established by Sokar et al., these units become permanently dormant.[15] Thus, the high variance and instability of vanilla DQN gradients actively accelerate the dormant neuron phenomenon and capacity loss. Conversely, DDQN's decoupled evaluation provides a vastly smoother, more consistent gradient signal.[57] With stable, conservative TD errors, the optimization process is gradual and measured. The network can smoothly aggregate the state space into the hierarchical, value-driven sub-manifolds described by Zahavy et al.[8] , resulting in the tight, distinct t-SNE clusters observed in the DDQN models.[60] Furthermore, because the weights are not subjected to sudden, violent negative gradient updates, the pre-activation distributions remain stable, preventing the ReLUs from being forced into permanent saturation. Consequently, DDQN natively maintains a lower dead neuron fraction and higher overall plasticity, powerfully demonstrating how algorithmic loss stability directly governs the structural health and 

geometric integrity of a neural representation. 

## **7. Conclusions** 

The comprehensive empirical observations of DQN and DDQN agents trained on Atari environments underscore the extreme fragility of internal representations in deep reinforcement learning. Synthesizing the experimental data with the theoretical literature yields several definitive conclusions regarding the optimization and geometry of RL agents: 

1. **Hierarchical Divergence is Value-Driven:** The high CKA observed in early convolutional layers and the extreme divergence in the fully connected layer is not an architectural anomaly; it reflects the necessary transition from general visual primitives (consistent with supervised learning) to highly specialized, MDP-specific value topographies. The representation space is forcibly warped to linearly separate expected future rewards. 

2. **Forgetting is a Geometrical Overwrite:** Catastrophic forgetting upon fine-tuning occurs because the terminal representation layer must be completely rewired to accommodate a new value landscape. Architectural freezing is insufficient unless the representation layer itself is locked. Methods like Continual Backpropagation (UPGD) are required to actively inject diversity and prevent the irreversible rigidification of these critical layers. 

3. **Dead Neurons Are a Symptom of Interference:** The catastrophic surge in dead neurons during interleaved multi-task training is the result of the network collapsing under the weight of feature superposition. To silence the chaotic gradient interference of mapping two disjoint MDPs into a single 512-dimensional space, the network aggressively saturates its ReLUs. Furthermore, plasticity loss is fundamentally tied to NTK rank collapse and curvature loss, which the literature proves occurs even without the presence of dead neurons. 

4. **Algorithmic Stability Dictates Latent Geometry:** The structural superiority of DDQN's representations—evidenced by tighter t-SNE clusters and fewer dead neurons—proves that target stability is paramount. Eliminating overestimation bias prevents the explosive, high-variance gradients that shatter latent geometry and induce premature neuron dormancy. 

To successfully scale multi-task and continual RL agents, future architectures must move beyond naive shared backbones. Implementing gradient surgery techniques (such as PCGrad or CAGrad), enforcing optimization stability (via LayerNorm and Weight Decay), and injecting stochastic utility-based resets (such as ReDo or Continual Backprop) are essential theoretical and practical steps toward building agents capable of maintaining robust, highly plastic internal representations. 

## **Works cited** 

1. LEARNING CONTINUALLY AT PEAK PERFORMANCE - OpenReview, accessed June 8, 2026, htps://openreview.net/pdf?id=UJqXhFFzKu 

2. Quantum entanglement provides a competitive advantage in adversarial games - arXiv, accessed June 8, 2026, htps://arxiv.org/html/2603.10289 

3. Task-Induced Representational Invariances Depend on Learning Objective in 

Deep RL, accessed June 8, 2026, htps://arxiv.org/html/2606.01868v1 

4. How transferable are features in deep neural networks? - NIPS, accessed June 8, 2026, - - - - - - 

htps://proceedings.neurips.cc/paper/5347 how transferable are features in de - - 

ep neural networks.pdf 

5. [1411.1792] How transferable are features in deep neural networks? - arXiv, accessed June 8, 2026, htps://arxiv.org/abs/1411.1792 

6. machine-learning-papers-summary/understanding-generalization-transfer/howtransferable-are-features-in-deep-neural-networks.md at master - GitHub, accessed June 8, 2026, - - - 

htps://github.com/GitYCC/machine learning papers summary/blob/master/unde - - - - - - - - 

rstanding generalization transfer/how transferable are features in deep neural - networks.md 

7. (PDF) How transferable are features in deep neural networks? - ResearchGate, accessed June 8, 2026, htps://www.researchgate.net/publication/268079628_How_transferable_are_feat ures_in_deep_neural_networks 

8. Graying the black box: Understanding DQNs - Proceedings of Machine Learning Research, accessed June 8, 2026, htps://proceedings.mlr.press/v48/zahavy16.html 

9. [1602.02658] Graying the black box: Understanding DQNs - arXiv, accessed June 

   - 8, 2026, htps://arxiv.org/abs/1602.02658 

10. Paper Summary - Graying the black box: Understanding DQNs - Abhijeet Krishnan, accessed June 8, 2026, 

- - htps://abhijeetkrishnan.me/technical/paper review zahavy/ 

11. Graying the black box: Understanding DQNs [Quick Review] - Liner, accessed 

   - - - - 

   - June 8, 2026, htps://liner.com/review/graying black box understanding dqns 

12. Graying the black box: Understanding DQNs - Department of Statistical Sciences, accessed June 8, 2026, htps://utstat.toronto.edu/droy/icml16/publish/zahavy16.pdf 

13. Graying the black box: Understanding DQNs - arXiv, accessed June 8, 2026, ------------------------ 

htps://arxiv.org/pdf/1602.02658v3.pdf?source=post_page 

   - --- 

14. Graying the Black Box: Understanding DQNs - The Morning Paper, accessed June 8, 2026, - - - - - 

htps://blog.acolyer.org/2016/03/02/graying the black box understanding dqns/ 

15. The Dormant Neuron Phenomenon in Deep Reinforcement Learning, accessed June 8, 2026, htps://proceedings.mlr.press/v202/sokar23a/sokar23a.pdf 

16. The Dormant Neuron Phenomenon in Deep Reinforcement Learning - arXiv, accessed June 8, 2026, htps://arxiv.org/html/2302.12902v2 

17. Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem, accessed June 8, 2026, htps://arxiv.org/html/2402.02868v2 

18. A Robotic Benchmark For Continual Reinforcement Learning - OpenReview, accessed June 8, 2026, htps://openreview.net/forum?id=5qsptDcsdEj 

19. Advancements and Challenges in Continual Reinforcement Learning: A 

   - Comprehensive Review - arXiv, accessed June 8, 2026, htps://arxiv.org/html/2506.21899v1 

20. A Robotic Benchmark For Continual Reinforcement Learning - deepsense.ai, accessed June 8, 2026, - - - - - 

htps://deepsense.ai/wp content/uploads/2025/01/Continual World A Robotic B - - - - 

enchmark For Continual Reinforcement Learning.pdf 

21. ICLR Poster Transient Non-stationarity and Generalisation in Deep Reinforcement Learning, accessed June 8, 2026, htps://iclr.cc/virtual/2021/poster/3156 

22. Transient Non-stationarity and Generalisation in Deep Reinforcement Learning, accessed June 8, 2026, htps://openreview.net/forum?id=Qun8fv4qSby 

23. NeurIPS Poster A Bayesian Fast-Slow Framework to Mitigate Interference in Non-Stationary Reinforcement Learning, accessed June 8, 2026, htps://neurips.cc/virtual/2025/poster/118436 

24. Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning, accessed June 8, 2026, htps://proceedings.iclr.cc/paper_fles/paper/2024/hash/8e5f0591943d8dae5702a - - 

f12dcdcd2f6 Abstract Conference.html 

25. Addressing Loss of Plasticity and Catastrophic Forgetting in Continual Learning, accessed June 8, 2026, htps://openreview.net/forum?id=sKPzAXoylB 

26. Loss of plasticity in deep continual learning - PubMed, accessed June 8, 2026, htps://pubmed.ncbi.nlm.nih.gov/39169245/ 

27. The Dual Nature of Plasticity Loss in Deep Continual Learning: Dissection and Mitigation, accessed June 8, 2026, htps://neurips.cc/virtual/2025/poster/115378 

28. Continual Backprop: Stochastic Gradient Descent with Persistent Randomness - Rich Sutton, accessed June 8, 2026, - - 

htp://incompleteideas.net/papers/RLDM22 DMS Continual_Backprop.pdf 

29. Loss of plasticity in deep continual learning - IDEAS/RePEc, accessed June 8, 2026, - - 

htps://ideas.repec.org/a/nat/nature/v632y2024i8026d10.1038_s41586 024 07711 - 7.html 

30. timoklein/redo: ReDo: The Dormant Neuron Phenomenon in Deep Reinforcement Learning (pytorch) - GitHub, accessed June 8, 2026, htps://github.com/timoklein/redo 

31. [2302.12902] The Dormant Neuron Phenomenon in Deep Reinforcement Learning - arXiv, accessed June 8, 2026, htps://arxiv.org/abs/2302.12902 

32. The Dormant Neuron Phenomenon in Deep Reinforcement Learning - Pablo Samuel Castro, accessed June 8, 2026, - 

htps://psc g.github.io/posts/research/rl/redo/ 

33. Plasticity Loss in Deep Reinforcement Learning: A Survey - ResearchGate, accessed June 8, 2026, htps://www.researchgate.net/publication/385630215_Plasticity_Loss_in_Deep_Re inforcement_Learning_A_Survey 

34. Understanding Plasticity in Neural Networks - Proceedings of Machine Learning Research, accessed June 8, 2026, 

htps://proceedings.mlr.press/v202/lyle23b/lyle23b.pdf 

35. ICLR Spotlight Understanding and Preventing Capacity Loss in Reinforcement Learning, accessed June 8, 2026, htps://iclr.cc/virtual/2022/spotlight/6598 

36. Disentangling the Causes of Plasticity Loss in Neural Networks - arXiv, accessed June 8, 2026, htps://arxiv.org/html/2402.18762v1 

37. [2303.01486] Understanding plasticity in neural networks - arXiv, accessed June 

   - 8, 2026, htps://arxiv.org/abs/2303.01486 

38. Understanding plasticity in neural networks - Semantic Scholar, accessed June 8, 2026, - - - - 

htps://www.semanticscholar.org/paper/Understanding plasticity in neural netw - - 

orks Lyle Zheng/542905f5fc96bce7572f6ded7f56aedfa62270c1 

39. Mitigating Plasticity Loss in Continual Reinforcement Learning by Reducing Churn, accessed June 8, 2026, htps://icml.cc/virtual/2025/poster/45929 

40. [2604.01913] The Rank and Gradient Lost in Non-stationarity: Sample Weight Decay for Mitigating Plasticity Loss in Reinforcement Learning - arXiv, accessed June 8, 2026, htps://arxiv.org/abs/2604.01913 

41. [2402.18762] Disentangling the Causes of Plasticity Loss in Neural Networks - arXiv, accessed June 8, 2026, htps://arxiv.org/abs/2402.18762 

42. Understanding and Preventing Capacity Loss in Reinforcement Learning - 

   - NeurIPS 2026, accessed June 8, 2026, htps://neurips.cc/virtual/2021/35684 

43. Toy Models of Superposition - Science Explorer Abstract, accessed June 8, 2026, htps://www.scixplorer.org/abs/2022arXiv220910652E/abstract 

44. Toy Models of Superposition - Transformer Circuits Thread, accessed June 8, - 

2026, htps://transformer circuits.pub/2022/toy_model/index.html 

45. [2209.10652] Toy Models of Superposition - arXiv, accessed June 8, 2026, htps://arxiv.org/abs/2209.10652 

46. Toy Models of Superposition, accessed June 8, 2026, - 

htps://www.mlmi.eng.cam.ac.uk/fles/2022_ _2023_advanced_machine_learning_ posters/toy_models_of_superposition_reduced.pdf 

47. Gradient Surgery for Multi-Task Learning, accessed June 8, 2026, htps://proceedings.neurips.cc/paper_fles/paper/2020/fle/3fe78a8acf5fda99de9 - 

5303940a2420c Paper.pdf 

48. Conflict-Averse Gradient Descent for Multi-task Learning - arXiv, accessed June 8, 2026, htps://arxiv.org/html/2110.14048v2 

49. Gradient Interference-Aware Graph Coloring for Multitask Learning - arXiv, accessed June 8, 2026, htps://arxiv.org/html/2509.16959v1 

50. Converging Pathways: Structured Visual Representations with Multi-Task Learning, accessed June 8, 2026, htps://www.doc.ic.ac.uk/~ajd/Publications/Liu-S-2024-PhD-Thesis.pdf 

51. Proactive Gradient Conflict Mitigation in Multi-Task Learning: A Sparse Training Perspective, accessed June 8, 2026, htps://arxiv.org/html/2411.18615v1 

52. Gradient Similarity Surgery in Multi-Task Deep Learning - AWS, accessed June 8, 2026, - - - 

htps://ecmlpkdd storage.s3.eu central 1.amazonaws.com/preprints/2025/resear ch/preprint_ecml_pkdd_2025_research_1013.pdf 

53. What exactly is the advantage of double DQN over DQN? - AI Stack Exchange, 

accessed June 8, 2026, 

   - - - - - 

   - htps://ai.stackexchange.com/questions/22776/what exactly is the advantage of - - - - double dqn over dqn 

54. Double DQN: Fixing Overestimation Bias - SATYAM MISHRA, accessed June 8, - - - 

2026, htps://satyamcser.substack.com/p/double dqn fxing overestimation 

55. Overestimation in Q-learning : r/reinforcementlearning - Reddit, accessed June 8, 2026, htps://www.reddit.com/r/reinforcementlearning/comments/dpitc9/overestimation 

   - _in_qlearning/ 

56. DDQN: Tackling Overestimation Bias in Deep Reinforcement Learning | by Dong-Keon Kim, accessed June 8, 2026, 

   - - - - - - 

   - htps://medium.com/@kdk199604/ddqn tackling overestimation bias in deep re - - 

   - inforcement learning b1b0d6fa72a4 

57. DDQN vs DQN: Who Balances Better? | by chloehyc - Medium, accessed June 8, 2026, - - - - - - 

htps://medium.com/@chloehyc.823/ddqn vs dqn who balances beter ba5beb 015f6b 

58. Explainable Artificial Intelligence for Reinforcement Learning Agents - Diva-Portal.org, accessed June 8, 2026, - 

htps://www.diva portal.org/smash/get/diva2:1553830/FULLTEXT01.pdf 

59. Deep Reinforcement Learning for Simulated Autonomous Vehicle Control - Stanford Computer Science, accessed June 8, 2026, htps://cs.stanford.edu/~rbedi/fles/cs231n_report.pdf 

60. Low Dimensional State Representation Learning with Reward-shaped Priors, accessed June 8, 2026, - 

htps://www.computer.org/csdl/proceedings article/icpr/2021/09412421/1tmiW9T hMB2 

61. Mimicking Human Intuition: Cognitive Belief-Driven Q-Learning - arXiv, accessed June 8, 2026, htps://arxiv.org/html/2410.01739v2 

