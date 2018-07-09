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

clear all
close all
clc

% #########################################################################
% ################                STATES             ######################
% #########################################################################

P_FC_Q = single(linspace(0,0,1)); % Fuel-cell power (not considered)
SOC_Q = single(linspace(0.5,1,11)); % Battery state of charge
dP_batt_Q = single(linspace(-1,1,2)); % Is the battery willing to charge or discharge? 
% The suffix _Q is added to emphasize that this is the state used in the
% Q-learning calculation

% Generate a state list
% 3 Column matrix of all possible combinations of the discretized state.
Q_states=zeros(length(P_FC_Q)*length(SOC_Q)*length(dP_batt_Q),3,'single'); 
% This part doesn't need to be optimized for simulation time (executed only
% once)
index=1;
for j=1:length(P_FC_Q)
    for k = 1:length(SOC_Q)
        for l = 1:length(dP_batt_Q)
            Q_states(index,1)=P_FC_Q(j);
            Q_states(index,2)=SOC_Q(k);
            Q_states(index,3)=dP_batt_Q(l);
            index=index+1;
        end
    end
end


% #########################################################################
% ###############                ACTIONS             ######################
% #########################################################################

% The only action on the grid from the EMS is on the FC current.
dI_FC_Q=0.1; %p.u.
actions=[0 -dI_FC_Q dI_FC_Q];
% NB: Must be consistent with the number of rows in the Q-matrix

% #########################################################################
% ###############          Q-learning SETTINGS         ####################
% #########################################################################

% Confidence in new trials?
learnRate = 0.99;

% Exploration vs exploitation
epsilon = 0.5; % Initial value
epsilonDecay = 0.9996; % Decay factor per iteration

% Future vs present value
discount = 0.9;

% Inject some noise?
successRate = 1; % No noise

% How many episodes of testing ? (i.e. how many courses the system attend?)
maxEpi = 2;

% Q matrix:
% Lines: states | Rows: actions
Q = repmat(zeros(size(Q_states,1),1,'single'),[1,3]);

% Load the functions (polynoms) calculating the rewards
load('rewardCurveSOC.mat')

% #########################################################################
% ################        INITIALIZE THE MODEL        #####################
% #########################################################################

% Choose model
model = 'DC_grid_V2';

% Set the (approximate) duration of one episode:
totalTime = 20;
% Set the length of one iteration in the simulink model
iterationTime = 1.3;

% Calculates the nomber of iterations
maxit = floor(totalTime/iterationTime);

% Empty structure containing the datas for one iteration:
systemStatesTab = struct(...
    'time',transpose(0:iterationTime:maxit*iterationTime)...
    ,'P_FC',zeros(maxit+1,1)...
    ,'P_Batt',zeros(maxit+1,1)...
    ,'SOC_battery',zeros(maxit+1,1)...
    ,'Load_profile',zeros(maxit+1,1)...
    ,'Setpoint_I_FC',zeros(maxit+1,1)...
    ,'isExploitationAction',zeros(maxit+1,1)...
    ,'reward',zeros(maxit+1,1));
% NOTE: This structure is overwritten each iteration.
% Has one more line than the number of iteration to include the initial
% state, and then generate rate of changes.

% Initialize a .txt file containing relevant datas
delete results.txt
resultsReport = fopen('results.txt','w');
fprintf(resultsReport,[datestr(now) '\r\n']);
fprintf(resultsReport,['Model used: ' model '\r\n']);
fprintf(resultsReport,'Learning rate %2.2f \r\n', learnRate);
fprintf(resultsReport,'Epsilon start: %2.2f, epsilonDecay %2.7f\r\n',epsilon,epsilonDecay);
fprintf(resultsReport,'Discount: %3.3f\r\n', discount);
fprintf(resultsReport,'Number of episodes planned: %i\r\n', maxEpi);
fprintf(resultsReport,'Total time per episode: %5.1fs, Iteration time: %3.2fs\r\n',totalTime,iterationTime);
fprintf(resultsReport,'_______________\r\n\r\n');

% #########################################################################
% #############              START LEARNING              ##################
% #########################################################################

for episodes = 1:maxEpi
    
    % $$$$$$$$$$$$$$$     INITIALIZE THE EPISODE      $$$$$$$$$$$$$$$$$$$$$
    
    % Measure the simulation time
    t_SimulinkTotal = 0; % For the time running in Simulink
    t_LearningStart = cputime; % For the total time (Simulink + Learning)
    
    % Load the initial conditions and set the load profile
    % NB: At this level is decided the initial condition
    % Develop here a case or random selection of load profile
    load('initialState_1A.mat');
    inputArray(2) = 1; % Code for the load profile (see documentation)
    
    % Loading the SimState
    currentSimState = initialSimState;
    
    % Charge the input for initial time: inputArray
    % (the input cannot ba calculated for initial time)
    % Row 1: Command for the FC current at the bus interface (i.e. between
    %        DC/DC conveter and bus. Unit is p.u. (base is the load).
    % Row 2: Load profile (code for each profile)
    inputsFromWS = Simulink.Parameter(inputArray);
    inputsFromWS.StorageClass='ExportedGlobal';
    
    % Initialize the the model constants to ensure consistency with the
    % initialization phase
    % (model_constants loaded a few lines above)
    initialize_model_constants(model,model_constants);
    
    % Open the model and set the simulation modes
    % NB: At this point, the initial state must be known
    load_system(model);
    % set_param(model,'FastRestart','off'); % Supposed to be 'off' at the
    % end of "initialize_model_constants(model,model_constants)"
    set_param(model,'SaveFinalState','on','FinalStateName','myOperPoint',...
        'SaveCompleteFinalSimState','on','LoadInitialState','on');
    set_param(model,'SimulationMode','accelerator');
    set_param(model,'FastRestart','on'); % No recompllling of the model between iterations
    
    % Starting point
    Q_state_struct = struct(...
        'P_FC',0,... % Not used yet
        'SOC',initial_outputsToWS.SOC,...
        'dP_Batt',0);
    systemStatesTab.P_Batt(1) = initial_outputsToWS.P_batt;
    % NOTE: The init value for dPbatt doesn't matter. 
    
    % Convert the structure to array for use in the Q-learning calculation
    Q_state_array = transpose(cell2mat(struct2cell(Q_state_struct)));
    
    % Number of exploitation actions (non-random actions) for result
    % analysis
    nExploitation = 0;
    
    % Initialize boolean for the case SOC < 10% (causing crash in simulink)
    failure = 0;
    
    for h = 1:maxit
        g = h + 1; % Do not write the first line (initial values)
        fprintf('Episode n.%i, iteration n.%i/%i\n',episodes,h,maxit);
        
        % $$$$$$$$$$$$$$$$$$     Pick an action     $$$$$$$$$$$$$$$$$$$$$$$
        % Interpolate the state within our discretization (ONLY for
        % choosing the action. We do not actually change the state by doing
        % this!)
        [~,sIdx] = min(sum((Q_states - repmat(Q_state_array,[size(Q_states,1),1])).^2,2));
        % sIdx is the index of the state matrix corresponding the best to
        % the current_state.
        
        % $$$$$$$$$$$$$$$$$    Choose an action    $$$$$$$$$$$$$$$$$$$$$$$$
        
        % EITHER 1) pick the best action according the Q matrix (EXPLOITATION).    
        if rand()>epsilon... 
                && rand()<=successRate... % Fail the check if our action doesn't succeed (i.e. simulating noise)
                && ((Q(sIdx,1)~=Q(sIdx,2)) && (Q(sIdx,1)~=Q(sIdx,3)))   % Take a random action when all the coefficients are equals
            
            [~,aIdx_fc] = max(Q(sIdx,:)); % Pick the action (for the FC current) the Q matrix thinks is best
            systemStatesTab.isExploitationAction(g) = 0.2; % For displaying only
            nExploitation = nExploitation + 1;
            
        % OR 2) Pick a random action (EXPLORATION)
        else
            aIdx_fc = randi(size(actions,2),1); % Random action for FC!
            systemStatesTab.isExploitationAction(g) = 0; % For displaying only        
        end
        
        % $$$$$$$$$$$$$$$$$    Run the model    $$$$$$$$$$$$$$$$$$$$$$$$$$$
        
        % New input for the model:
        dI_FC_Q = actions(1,aIdx_fc);
        inputArray(1) = inputArray(1) + dI_FC_Q;
        % Keep the I_FC_Q in bounds (redundant with limiters in the
        % simulink model, but accelerates convergence)
        if inputArray(1)<0 
            inputArray(1)=0;
        elseif inputArray(1)>1.9
            inputArray(1)=1.9;
        end
        inputsFromWS.Value = inputArray;
        
        % Run ths Simulink model for iterationTime
        t_SimulinkIterationStart = cputime;
        [currentSimState,simOut] = run_simulation(model,currentSimState,iterationTime);
        t_SimulinkTotal = t_SimulinkTotal + cputime - t_SimulinkIterationStart;
        
        % Collect the results of the iteration (last value returned by the model):
        systemStatesTab.P_FC(g)  = simOut.outputsToWS.P_FC.Data(end); 
        systemStatesTab.P_Batt(g) = simOut.outputsToWS.P_batt.Data(end);
        systemStatesTab.SOC_battery(g) = simOut.outputsToWS.SOC.Data(end);
        systemStatesTab.Setpoint_I_FC(g) = inputArray(1);
        systemStatesTab.Load_profile(g) = simOut.outputsToWS.Load_profile.Data(end);
        
        % Fill the Q-learning state
        % Q_state_struct.P_FC = simOut.outputsToWS.P_FC.Data(end);
        Q_state_struct.SOC = simOut.outputsToWS.SOC.Data(end);
        if systemStatesTab.P_Batt(g) <= systemStatesTab.P_Batt(g-1) % The battery power is decreasing (willing to charge even more)
            Q_state_struct.dP_Batt = -1;
        else % The battery power is increasing
            Q_state_struct.dP_Batt = 1;
        end

        
        % Convert the structure to array for use in the Q-learning calculation
        Q_state_array = transpose(cell2mat(struct2cell(Q_state_struct)));
        
        % $$$$$$$$$$$$$$$$    Calculate the reward     $$$$$$$$$$$$$$$$$$$$
        reward = getReward(Q_state_struct,rewardCurveSOC);
        fprintf('SOC %3.3f\n',Q_state_struct.SOC);
        systemStatesTab.reward(g) = reward;
        
        % $$$$$$$$$$$$$$$$   Update the Q-matrix    $$$$$$$$$$$$$$$$$$$$$$$
        % Interpolate again to find the new state the system is closest to.
        [~,snewIdx] = min(sum((Q_states - repmat(Q_state_array,[size(Q_states,1),1])).^2,2)); % Interpolate again to find the new state the system is closest to.
        
        % Update Q
        Q(sIdx,aIdx_fc) = Q(sIdx,aIdx_fc) + learnRate * ( reward + discount*max(Q(snewIdx,:)) - Q(sIdx,aIdx_fc) ); % The line that makes everything !!!
        fprintf('State index %i\n',sIdx);
        fprintf('Reward %2.2f\n',reward);
        fprintf('Q(sIdx,aIdx_fc) %3.2f\n',Q(sIdx,aIdx_fc));
        
        % Decay the odds of picking a random action vs picking the
        % estimated "best" action. I.e. we're becoming more confident in
        % our learned Q.
        epsilon = epsilon*epsilonDecay;
        
        % Break the iteration if SOC < 10%
        if simOut.outputsToWS.SOC.Data(end) < 0.1
            failure = 1;
            break
        end        
        
    end % end iterations counting for single episode

    % Close the model without saving it
    set_param(model,'FastRestart','off');
    close_system(model,0); % Seem that the simulations are longer when restarting from an already opened model
    
    % Analysis of the episode performance
    if ~failure
        t_LearningTotal = cputime - t_LearningStart;
        fprintf(resultsReport,'Episode %i: \r\n',episodes);
        ratioExploitation = (nExploitation/maxit)*100;
        fprintf(resultsReport,'Exploitation actions: %3.2f%% \r\n',ratioExploitation);
        fprintf(resultsReport,'Epsilon (end of iteration): %2.3f \r\n',epsilon);
        fprintf(resultsReport,'Simulink time: %5.1fs \r\n',t_SimulinkTotal);
        fprintf(resultsReport,'Episode duration (Simulink + Q-process): %5.1fs \r\n',t_LearningTotal);
        ratioTime = (t_SimulinkTotal/t_LearningTotal)*100;
        fprintf(resultsReport,'Ratio Simulink/Total time for episode: %3.2f%% \r\n',ratioTime);
        fprintf(resultsReport,'_______________\r\n\r\n');
    else
        fprintf(resultsReport,'Episode %i: \r\n',episodes);
        fprintf(resultsReport,'Failure, SOC too close to 0\r\n');
        fprintf(resultsReport,'_______________\r\n\r\n');
    end
    
    % Plotting the result of the episode
    fig = figure(episodes);
    subplot(311);
    plot(systemStatesTab.time(2:end),systemStatesTab.SOC_battery(2:end),'.-');
    hold on
    bar(systemStatesTab.time(2:end),systemStatesTab.isExploitationAction(2:end));
    legend('SOC','Exploitation','Location','southwest');
    subplot(312)
    plot(systemStatesTab.time(2:end),systemStatesTab.reward(2:end),'.-');
    legend('Reward','Location','southwest');
    subplot(313);
    plot(systemStatesTab.time(2:end),systemStatesTab.Setpoint_I_FC(2:end),'.-');
    hold on
    plot(systemStatesTab.time(2:end),systemStatesTab.Load_profile(2:end),'.-');
    legend('I FC (p.u.)','Load profile (p.u.)','Location','southwest');
    drawnow
    saveas(fig,['episode' num2str(episodes) '.fig']);
    close(fig);
    
    % Save the Q-matrix
    save(['Q_matrix_episode' num2str(episodes) '.mat'],'Q');
    
end % end episodes counting

% Close the text file
fclose(resultsReport);