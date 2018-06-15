% Use of Machine Learning to make an EMS for a load supplied by Fuel-Cell
% (FC) and Battery. The purpose is to ensure load-sharing as with a normal
% control strategy (i.e. no overload) and to make the sharing more
% efficient (what normal control strategies difficultly allow). 


% DESCRIPTION:
% Learn a control policy to ensure that the power available on the bus
% equals the power of the load at any time.
% The environment is the Simulink Model of the power grid (FC, Battery, DC
% to DC converters and Load).

% NOTE:
% This script is made for use 100% of a 8GB ram 

% STATUS:
% Body is working. Next step: setting up the Simulink model.

% TO DO:
% Plot the performance

% #########################################################################
% ################                STATES             ######################
% #########################################################################

% As a first set of states, is has been chosen:
% - Bus, FC & Battery currents
% - Bus, FC & Battery voltages
% Then, the power is known by P=UI
% NB: the choice of current + votage state has been chosen to make easier
% further developements.


% $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
% $$$$$$$    Values to fill depending on the model or hardware    $$$$$$$$$
% To make this script usable for both differents models and different 
% hardware, the state values are translated in p.u. with the load power as 
% the base.
% e.g. Iload_pu=Iload/Iload_nominal & Vload_pu=Vload/Vload_nominal

% The base current and voltage
Vload_nominal=1;
Iload_nominal=1;

% The nominal current and voltages for each component
Vbus_nominal=1;
Ibus_nominal=1;
Vfc_nominal=1;
Ifc_nominal=1;
Vbatt_nominal=1;
Ibatt_nominal=1;
% $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
% $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

% It is assumed that the values for the components cannot og over -3 and
% +3 p.u.
% In reality, if this intervall is:
% - too small => false calculation.
% - too big => explosion of the number of possible states.


% THE STATES ARE:
Pfc_pu = -0.5:0.05:3; % The current is not supposed to be negative
Pbatt_pu = -3:0.05:3;
Pbus_pu = -1:0.05:2;

% This must be optimized for memory (otherwise the Q-matrix reaches many
% GB...) and running time.

% Optimization 0: Certains ranges of values are irrelevant (e.g. negative
% current for the FC). Remove them with experience

% Optimization 1: Use of 32-bit format instead of the default 64 bits. The
% size in memory of a number is divided by 2. All integers with 6 or fewer 
% significant decimal digits can be converted into an IEEE 754 
% floating-point value without loss of precision. Then, here if it is
% chosen to have states between -10 and 10 p.u., the minimum discretisation
% step without loss of precision is 0.0001 (because 10.0001 have 6 
% significant digits)
Pfc_pu = single(Pfc_pu);
Pbatt_pu = single(Pbatt_pu);
Pbus_pu = single(Pbus_pu);


% Optimization 2: Variable step (i.e. increasing the step for high p.u.
% values). Not used here for the moment.

%Generate a state list
states=zeros(length(Pfc_pu)*length(Pbatt_pu)*length(Pbus_pu),3,'single'); % 3 Column matrix of all possible combinations of the discretized state.
% 'single' precision here
index=1;
for j=1:length(Pfc_pu)
    for k = 1:length(Pbatt_pu)
        for l = 1:length(Pbus_pu)  
            states(index,1)=Pfc_pu(j);
            states(index,2)=Pbatt_pu(k);
            states(index,3)=Pbus_pu(l);      
            index=index+1;
        end        
    end
end


% #########################################################################
% ###############                ACTIONS             ######################
% #########################################################################

% It is possible to act on the command of Ifc and Ibatt:
d_Ifc_pu=0.05; %p.u.
d_Ibatt_pu=0.05; %p.u.
actions=[0 -d_Ifc_pu d_Ifc_pu ; 0 -d_Ibatt_pu d_Ibatt_pu];
% Line1: actions for FC current
% Line2: actions for Battery current
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
% 1st dimension: states
% 2nd dimension: actions on FC current (3 possible)
% 3rd dimension: actions on Battery current (3 possible)
% Check consistency with the parameters and actions stated above ([1,3,3]
% means two parameters modifiable, and 3 actions for each parameter)

Q=repmat(zeros(size(states,1),1,'single'),[1,3,3]); 
% Q elements are stored on 32bits (allow Qfactors between 2^-126 and 2^127)
% Can also be ran in 'double' while not too much states... 'simple' makes
% computation faster


% #########################################################################
% #############              START LEARNING              ##################
% #########################################################################

% Number of episodes or reset
for episodes = 1:maxEpi
    
    % Start point to fill here...
    current_state = [0.2,0,0.2];
    set_point = [1,1];
    % Command and current state are the same at the moment. For more
    % accuracy, more states can be included later on
    
    for g = 1:maxit
        
        % $$$$$$$$$$$$$$$$$$     Pick an action     $$$$$$$$$$$$$$$$$$$$$$$
        % Interpolate the state within our discretization (ONLY for
        % choosing the action. We do not actually change the state by doing
        % this!)
        [~,sIdx] = min(sum((states - repmat(current_state,[size(states,1),1])).^2,2));
        % sIdx is the index of the state matrix corresponding the best to
        % the current_state.
        
        % $$$$$$$$$$$$$$$$$    Choose an action    $$$$$$$$$$$$$$$$$$$$$$$$
        % EITHER 1) pick the best action according the Q matrix (EXPLOITATION). OR
        % 2) Pick a random action (EXPLORATION)
        if (rand()>epsilon || episodes == maxEpi) && rand()<=successRate % Pick according to the Q-matrix it's the last episode or we succeed with the rand()>epsilon check. Fail the check if our action doesn't succeed (i.e. simulating noise)
            [~,aIdx_fc] = max(Q(sIdx,:,1)); % Pick the action (for the FC current) the Q matrix thinks is best!
            [~,aIdx_batt] = max(Q(sIdx,:,2)); % Pick the action (for the Battery current) the Q matrix thinks is best!
        else
            aIdx_fc = randi(size(actions,2),1); % Random action for FC!
            aIdx_batt = randi(size(actions,2),1); % Random action for Batt!
        end
        
        % $$$$$$$$$$$$$$$$$    Run the model    $$$$$$$$$$$$$$$$$$$$$$$$$$$
        % New input for the model:
        d_Ifc = actions(1,aIdx_fc)*Ifc_nominal;
        d_Ibatt = actions(2,aIdx_fc)*Ibatt_nominal; %Ibatt or Ibus?...
        set_point = set_point + [d_Ifc,d_Ibatt];
        if set_point(1)<0 % Current always flowing out of the FC
            set_point(1)=0;
        end
        % Updating the state:
        %    run Simulink here for dt
        new_state = [0.2,0,0.2]; %update the state variables
        
        % $$$$$$$$$$$$$$$$    Calculate the reward     $$$$$$$$$$$$$$$$$$$$
        reward = 1; % rewardFunc(new_state);
        
        % $$$$$$$$$$$$$$$$   Update the Q-matrix    $$$$$$$$$$$$$$$$$$$$$$$
        % NB: no end condition of the episode here, because it is a
        % tracking problem.
        [~,snewIdx] = min(sum((states - repmat(new_state,[size(states,1),1])).^2,2)); % Interpolate again to find the new state the system is closest to.
        current_state = new_state;
        
        if episodes ~= maxEpi % On the last iteration, stop learning and just execute. Otherwise...
            % Update Q
            Q(sIdx,aIdx_fc,aIdx_batt) = Q(sIdx,aIdx_fc,aIdx_batt) + learnRate * ( reward + discount*max(max(Q(snewIdx,:,:))) - Q(sIdx,aIdx_fc,aIdx_batt) );
            
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