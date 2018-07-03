% Energy Management System based on Machine Learning theory (reinforcment
% learning) for a DC grid. The main source of power is a fuel-cell and a
% battery is connected to the grid to supply fast load changes (the slow
% dynamics of the FC cannot supply fast load changes).
% The aim of machine learning is to make the grid more efficient (by
% minimizing the losses in converters and battery for example) and to
% increase lifecycle of the system by reducing stress on conponents.


% DESCRIPTION:
% Goal 1: Ensure the power supply at any time (done by the hardware, not
% related to ML)
% Goal 2: Maintain the State of Charge of the battery in [SOCmin;SOCmax].
% Goal 3: Minimize fuel consumption of the grid.

% NOTE:
% The Q-learning script is written in per-unit. The outputs of the simulink
% model are in p.u.

% STATUS:
% 15/06/18: Body is working. Next step: setting up the Simulink model.

% TO DO:
% Plot the performance

% #########################################################################
% ################                STATES             ######################
% #########################################################################

% As a first set of states, is has been chosen:
% Fuel-cell and load powers, SOC of the battery




% THE STATES ARE:
P_FC_Q = single(linspace(0,1.5,31));
P_load_Q = single(linspace(0,1.5,31));
SOC_Q = single(linspace(0,10,11));
% The suffix _Q is added to emphasize that this is the state used in the ML
% calculation.
% Single precision is used to gain memory space
% To gain more memory space, a variable intervall size can be implemented
% for the states (not necessay yet).


% Generate a state list
states=zeros(length(P_FC_Q)*length(P_load_Q)*length(SOC_Q),3,'single'); % 3 Column matrix of all possible combinations of the discretized state.
% 'single' precision here
index=1;
for j=1:length(P_FC_Q)
    for k = 1:length(P_load_Q)
        for l = 1:length(SOC_Q)  
            states(index,1)=P_FC_Q(j);
            states(index,2)=P_load_Q(k);
            states(index,3)=SOC_Q(l);      
            index=index+1;
        end        
    end
end


% #########################################################################
% ###############                ACTIONS             ######################
% #########################################################################

% The only action on the grid from the EMS is on the FC current.
dQ_input=0.1; %p.u.
actions=[0 -dQ_input dQ_input];
% NB:Check consistency with the implementation of the Q-matrix below
% (number of parameters and number of actions possible for each parameter)

% #########################################################################
% ###############               SETTINGS             ######################
% #########################################################################

% Confidence in new trials?
learnRate = 0.99; 

% Exploration vs exploitation
epsilon = 0.5; % Initial value
epsilonDecay = 0.98; % Decay factor per iteration

% Future vs present value
discount = 0.9;

% Inject some noise?
successRate = 1; % No noise

% Starting point : to be defined

% How many episodes of testing ? (i.e. how many courses the system attend?) 
maxEpi = 10;
% How long are the episodes ? (i.e. how long are the courses?)
maxit = 100;
% The more courses the system attend the best will be the system. The
% same with the length of the courses.

% Time-step for running the model:
dt = 0.05;

% Q matrix:
% Lines: states
% Rows: actions
% Check consistency with the parameters and actions stated above ([1,3,3]
% means two parameters modifiable, and 3 actions for each parameter)

Q=repmat(zeros(size(states,1),1,'single'),[1,3]); 
% Q elements are stored on 32bits (allow Qfactors between 2^-126 and 2^127)



% #########################################################################
% #############              START LEARNING              ##################
% #########################################################################

% Number of episodes or reset
for episodes = 1:maxEpi
    
    % Start point to fill here...
    Q_state = [0.5,0.5,0.7];
    Q_input = 0;
    
    for g = 1:maxit
        
        % $$$$$$$$$$$$$$$$$$     Pick an action     $$$$$$$$$$$$$$$$$$$$$$$
        % Interpolate the state within our discretization (ONLY for
        % choosing the action. We do not actually change the state by doing
        % this!)
        [~,sIdx] = min(sum((states - repmat(Q_state,[size(states,1),1])).^2,2));
        % sIdx is the index of the state matrix corresponding the best to
        % the current_state.
        
        % $$$$$$$$$$$$$$$$$    Choose an action    $$$$$$$$$$$$$$$$$$$$$$$$
        % EITHER 1) pick the best action according the Q matrix (EXPLOITATION). OR
        % 2) Pick a random action (EXPLORATION)
        if (rand()>epsilon || episodes == maxEpi) && rand()<=successRate % Pick according to the Q-matrix it's the last episode or we succeed with the rand()>epsilon check. Fail the check if our action doesn't succeed (i.e. simulating noise)
            [~,aIdx_fc] = max(Q(sIdx,:)); % Pick the action (for the FC current) the Q matrix thinks is best
        else
            aIdx_fc = randi(size(actions,2),1); % Random action for FC!
        end
        
        % $$$$$$$$$$$$$$$$$    Run the model    $$$$$$$$$$$$$$$$$$$$$$$$$$$
        % New input for the model:
        dQ_input= actions(1,aIdx_fc);
        Q_input = Q_input + dQ_input;
        if Q_input(1)<0 % Current always flowing out of the FC
            Q_input(1)=0;
        end
        % Updating the state:
        %    run Simulink here for dt
        new_state = [0.5,0.5,0.7]; %update the state variables
        
        % $$$$$$$$$$$$$$$$    Calculate the reward     $$$$$$$$$$$$$$$$$$$$
        reward = 1; % rewardFunc(new_state);
        
        % $$$$$$$$$$$$$$$$   Update the Q-matrix    $$$$$$$$$$$$$$$$$$$$$$$
        % NB: no end condition of the episode here, because it is a
        % tracking problem.
        [~,snewIdx] = min(sum((states - repmat(new_state,[size(states,1),1])).^2,2)); % Interpolate again to find the new state the system is closest to.
        Q_state = new_state;
        
        if episodes ~= maxEpi % On the last iteration, stop learning and just execute. Otherwise...
            % Update Q
            Q(sIdx,aIdx_fc) = Q(sIdx,aIdx_fc) + learnRate * ( reward + discount*max(max(Q(snewIdx,:))) - Q(sIdx,aIdx_fc) );
            
            % Lets break this down:
            %
            % We want to update our estimate of the global value of being
            % at our previous state s and taking action a. We have just
            % tried this action, so we have some information. Here are the terms:
            %   1) Q(sIdx,aIdx) AND later -Q(sIdx,aIdx) -- Means we're
            %      doing a weighting of old and new (see 2). Rewritten:
            %      (1-alpha)*Qold + alpha*newStuff
            %   2) learnRate * ( ... ) -- Scaling factor for our update.
            %      High learnRate means that new information has great weight.
            %      Low learnRate means that old information is more important.
            %   3) R(snewIdx) -- the reward for getting to this new state
            %   4) discount * max(Q(snewIdx,:)) -- The estimated value of
            %      the best action at the new state. The discount means future
            %      value is worth less than present value
            %   5) Bonus - I choose to give a big boost of it's reached the
            %      goal state. Optional and not really conventional.
        end
        
        % Decay the odds of picking a random action vs picking the
        % estimated "best" action. I.e. we're becoming more confident in
        % our learned Q.
        epsilon = epsilon*epsilonDecay;
        
        
    end % end iterations counting for single episode
    
end % end episodes counting