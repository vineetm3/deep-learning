\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{cite}
\usepackage{booktabs}
\usepackage{enumitem}


\title{\textbf{Spatio-Temporal Player Trajectory Prediction in the NFL using Graph Neural Networks}}
\date{}

\begin{document}

\maketitle

\begin{abstract}
This project proposes a novel deep learning framework for predicting the $x$ and $y$ coordinates of NFL players during the critical moments when the football is in the air. The complex, multi-agent nature of football necessitates modeling the explicit interactions and dependencies between players, a limitation of traditional sequence models like LSTMs. Our solution is a Graph Neural Network (GNN)-LSTM Encoder-Decoder architecture. The GNN serves as the encoder, dynamically creating a graph where players are nodes and edges represent their relative distances and influence. Additional nodes represent the known ball landing location and targeted receiver, providing crucial contextual information. This GNN encodes the spatio-temporal state of the field into rich player embeddings, which are then decoded by role-conditioned LSTMs to generate the precise sequence of future $x$ and $y$ positions for each player. We will evaluate the model's performance using the Root Mean Squared Error (RMSE) against established baseline models on the NFL Big Data Bowl 2026 dataset.
\end{abstract}

\section{Introduction and Related Work}

\subsection{Motivation}

The availability of high-resolution player tracking data from NFL Next Gen Stats presents a significant opportunity to advance sports analytics beyond simple descriptive statistics to complex predictive modeling. The 2026 NFL Big Data Bowl focuses on a fundamental challenge in football analytics: forecasting the trajectories of all 22 players on the field during the time the ball is in flight after a quarterback releases a pass. This is an inherently difficult spatio-temporal, multi-agent prediction task where each player's movement is influenced by their role, the ball's destination, and the actions of surrounding players.

Accurately predicting these trajectories has significant practical applications. For defensive coordinators, understanding likely coverage reactions can inform play-calling and pre-snap adjustments. For offensive coaches, modeling defender movements can optimize route concepts and spacing. For broadcast analytics, real-time trajectory predictions can enhance viewer engagement by visualizing expected vs. actual player movements.

\subsection{Related Work and Key Limitations}

Prior work in sports trajectory prediction and multi-agent modeling provides important foundations but exhibits key limitations for this specific problem.

\textbf{Recurrent Neural Networks (RNNs/LSTMs):} Standard sequence models like LSTMs~\cite{hochreiter1997long} have been widely applied to trajectory prediction tasks. These models excel at capturing temporal dependencies—how a player's previous movements influence their next position. However, when applied to multi-agent scenarios, they typically process each agent independently or concatenate all agent features into a single input vector. This approach fails to explicitly model the dynamic, non-local interactions between players. For instance, a defender's trajectory is a direct reaction to the targeted receiver's movement, creating a spatial dependency that sequential processing alone cannot naturally capture.

\textbf{Convolutional Neural Networks (CNNs):} Some sports analytics approaches encode the field as a spatial image grid and apply CNNs~\cite{le2017coordinated}. While CNNs can capture spatial patterns, they struggle with the irregular, non-grid-like positioning of players and the variable number of agents. The sparse nature of 22 players on a 120×53 yard field leads to inefficient representations.

\textbf{Graph Neural Networks for Sports:} Graph Neural Networks~\cite{kipf2017semi, veličković2018graph} provide a natural framework for modeling relational data. Recent applications in sports include team performance prediction and tactical analysis~\cite{decroos2019actions}, but their application to real-time trajectory prediction with NFL tracking data remains underexplored.

\subsection{Key Limitations Addressed}

Our proposed solution directly addresses three critical limitations:

\begin{enumerate}[leftmargin=*]
    \item \textbf{Lack of explicit spatial interaction modeling:} We use GNNs to create a dynamic graph where edges explicitly represent player-to-player and player-to-ball relationships, allowing the model to learn which interactions matter most.
    
    \item \textbf{Ignoring role-based behavioral differences:} We condition our trajectory decoder on player roles, acknowledging that targeted receivers, defensive coverage players, and other route runners exhibit fundamentally different movement patterns.
    
    \item \textbf{Underutilizing ball landing information:} Unlike general trajectory prediction problems, we have access to the ball's landing location at prediction time. We incorporate this as a special graph node, enabling all players to attend to this critical spatial anchor.
\end{enumerate}

\section{Proposed Contributions}

Our primary contribution is the development and validation of a \textbf{Spatio-Temporal Graph Attention Network (ST-GAT) with Role-Conditioned LSTM Decoders}, tailored specifically for the multi-agent prediction problem in American football. This architecture explicitly models player interactions through graph structures while respecting the heterogeneous nature of player roles.

\subsection{Proposed Architecture: GNN-LSTM Encoder-Decoder}

Our architecture consists of three main components: graph construction, a GNN encoder to model spatial interactions, and role-conditioned LSTM decoders to generate future trajectories.

\subsubsection{Graph Construction}

For each play, we represent the field state as a graph where players and the ball landing location are nodes, and edges represent spatial relationships.

\textbf{Graph Nodes:} We create 23 nodes per play:
\begin{itemize}
    \item \textbf{22 player nodes} (11 offense, 11 defense): Each player is represented with features including their position ($x, y$), motion (speed, acceleration, direction), and contextual information (role, distance to ball landing)
    \item \textbf{1 ball landing node}: A special node representing where the ball will land, with features including the landing coordinates and play context
\end{itemize}

\textbf{Graph Edges:} We connect nodes using three types of edges:
\begin{itemize}
    \item \textbf{Player-to-ball edges:} Every player connects to the ball landing node, allowing the model to learn how each player orients toward the ball's destination
    \item \textbf{Proximity edges:} Each player connects to their 5 nearest neighbors, capturing local spatial interactions
    \item \textbf{Role-specific edges:} Defensive coverage players have additional connections to the targeted receiver, emphasizing this critical interaction
\end{itemize}

Each edge includes features such as distance and relative velocity between connected players, providing the model with geometric context.

\subsubsection{GNN Encoder: Graph Attention Network}

We use a Graph Attention Network (GAT)~\cite{veličković2018graph} with 2-3 layers to encode the spatial relationships. The key advantage of GAT is its \textit{attention mechanism}, which learns to weight the importance of different connections. For example, a defender in coverage should pay more attention to the targeted receiver's movement than to a distant offensive lineman.

The attention mechanism works as follows: for each player node, the GNN examines all connected neighbors and computes attention weights that indicate how much influence each neighbor should have. These weights are learned during training and adapt to different game situations. The GNN then aggregates information from neighbors (weighted by attention) to update each player's representation, capturing both their individual state and their spatial context.

The GNN processes the last 10 input frames (1 second before the throw) sequentially, producing a rich, spatially-aware embedding for each player that encodes: (1) their own movement history, (2) their relationships to nearby players, and (3) their orientation to the ball landing location.

\subsubsection{LSTM Decoder: Role-Conditioned Trajectory Generation}

The decoder generates future trajectories using role-conditioned LSTMs. We use separate decoders for different player roles because a targeted receiver behaves fundamentally differently than a pass rusher or coverage defender.

For each player, the decoder receives:
\begin{itemize}
    \item \textbf{Context vector:} The player's final GNN embedding, summarizing their spatial situation at the time of the throw
    \item \textbf{Role embedding:} A learned representation of their role (Targeted Receiver, Defensive Coverage, etc.)
    \item \textbf{Ball landing location:} The known $(x, y)$ coordinates where the ball will arrive
\end{itemize}

The LSTM then auto-regressively generates the trajectory frame-by-frame. At each step, it predicts the next $(x, y)$ position, feeds that prediction back as input, and repeats until the sequence is complete (determined by the play-specific number of frames).

During training, we use \textit{scheduled sampling}~\cite{bengio2015scheduled}: early in training, we feed ground truth positions back to the decoder; later, we increasingly feed the model's own predictions. This helps the model learn to correct its own errors during deployment.

\subsection{Training Procedure}

\textbf{Loss Function:} We train the model by minimizing the Mean Squared Error (MSE) between predicted and true $(x, y)$ positions across all players and time steps. Importantly, we use \textit{role-based weighting}: predictions for targeted receivers and defensive coverage players receive 1.5× weight, while other players receive standard 1.0× weight. This focuses the model on accurately predicting the most critical interactions around the ball.

\textbf{Optimization:} We use the Adam optimizer with an initial learning rate of 0.001, reducing it by half when validation loss plateaus. Training uses mini-batches of 32 plays with gradient clipping (max norm 1.0) to ensure stable convergence. We train for approximately 50-100 epochs depending on convergence, monitoring validation RMSE to prevent overfitting.

\section{Evaluation Plan}

\subsection{Dataset}

We will use the \textbf{NFL Big Data Bowl 2026} dataset, which provides Next Gen Stats tracking data at 10 frames per second. The dataset contains:
\begin{itemize}
    \item \textbf{Training data:} Historical data from the 2023 NFL season (weeks 1-18), split into input (pre-pass) and output (ball in air) files
    \item \textbf{Test data:} Live evaluation on weeks 14-18 of the 2025 NFL season (unseen future data)
    \item \textbf{Play filtering:} Quick passes, deflections, and throwaways are excluded
\end{itemize}

We will split the 2023 training data into:
\begin{itemize}
    \item Training: Weeks 1-14 (78\%)
    \item Validation: Weeks 15-16 (11\%)
    \item Test (local): Weeks 17-18 (11\%)
\end{itemize}

This temporal split ensures the model generalizes to unseen game situations and prevents data leakage.

\subsection{Evaluation Metrics}

\textbf{Primary Metric:} Root Mean Squared Error (RMSE) as specified by the competition:
\begin{equation}
\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left[ (x_{\text{true}, i} - x_{\text{pred}, i})^2 + (y_{\text{true}, i} - y_{\text{pred}, i})^2 \right]}
\end{equation}

where $N$ is the total number of coordinate predictions across all players, plays, and frames.

\subsection{Baseline Models}

To demonstrate the value of the GNN architecture, we will compare against three baselines:

\begin{enumerate}
    \item \textbf{Constant Velocity (CV):} A physics-based baseline that extrapolates each player's trajectory using their velocity at the time of the throw:
    \begin{equation}
    \hat{x}_{t+\Delta t} = x_t + v_x \cdot \Delta t, \quad \hat{y}_{t+\Delta t} = y_t + v_y \cdot \Delta t
    \end{equation}
    This provides a simple, interpretable baseline.
    
    \item \textbf{Vanilla LSTM:} A standard LSTM that processes each player's trajectory independently, using only their own historical positions and velocities. This baseline tests whether temporal modeling alone is sufficient.
    
    \item \textbf{Concatenated LSTM:} An LSTM that takes the flattened state of all 22 players as input at each time step. This provides implicit interaction modeling through the shared hidden state but lacks explicit relational structure.
\end{enumerate}

\begin{thebibliography}{9}

\bibitem{kipf2017semi}
Kipf, T. N., \& Welling, M. (2017).
Semi-supervised classification with graph convolutional networks.
\textit{International Conference on Learning Representations (ICLR)}.

\bibitem{veličković2018graph}
Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., \& Bengio, Y. (2018).
Graph attention networks.
\textit{International Conference on Learning Representations (ICLR)}.

\bibitem{hochreiter1997long}
Hochreiter, S., \& Schmidhuber, J. (1997).
Long short-term memory.
\textit{Neural computation}, 9(8), 1735-1780.

\bibitem{alahi2016social}
Alahi, A., Goel, K., Ramanathan, V., Robicquet, A., Fei-Fei, L., \& Savarese, S. (2016).
Social lstm: Human trajectory prediction in crowded spaces.
\textit{Proceedings of the IEEE conference on computer vision and pattern recognition}, 961-971.

\bibitem{gupta2018social}
Gupta, A., Johnson, J., Fei-Fei, L., Savarese, S., \& Alahi, A. (2018).
Social gan: Socially acceptable trajectories with generative adversarial networks.
\textit{Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition}, 2255-2264.

\bibitem{salzmann2020trajectron++}
Salzmann, T., Ivanovic, B., Chakravarty, P., \& Pavone, M. (2020).
Trajectron++: Dynamically-feasible trajectory forecasting with heterogeneous data.
\textit{European Conference on Computer Vision}, 683-700.

\bibitem{le2017coordinated}
Le, H. M., Yue, Y., Carr, P., \& Lucey, P. (2017).
Coordinated multi-agent imitation learning.
\textit{International Conference on Machine Learning}, 1995-2003.

\bibitem{decroos2019actions}
Decroos, T., Bransen, L., Van Haaren, J., \& Davis, J. (2019).
Actions speak louder than goals: Valuing player actions in soccer.
\textit{Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining}, 1851-1861.

\bibitem{bengio2015scheduled}
Bengio, S., Vinyals, O., Jaitly, N., \& Shazeer, N. (2015).
Scheduled sampling for sequence prediction with recurrent neural networks.
\textit{Advances in Neural Information Processing Systems}, 28, 1171-1179.

\end{thebibliography}

\end{document}

