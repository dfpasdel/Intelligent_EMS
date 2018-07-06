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

clear all
close all
clc

% #########################################################################
% ################                STATES             ######################
% #########################################################################

P_FC_Q = single(linspace(0,1.5,16)); % Fuel-cell power
P_load_Q = single(linspace(0,1.5,16)); % Load power
SOC_Q = single(linspace(0.5,0.9,6)); % Battery state of charge
% The suffix _Q is added to emphasize that this is the state used in the
% Q-learning calculation

% Generate a state list
% 3 Column matrix of all possible combinations of the discretized state.
Q_states=zeros(length(P_FC_Q)*length(P_load_Q)*length(SOC_Q),3,'single'); 
% 'single' precision here
index=1;
for j=1:length(P_FC_Q)
    for k = 1:length(P_load_Q)
        for l = 1:length(SOC_Q)
            Q_states(index,1)=P_FC_Q(j);
            Q_states(index,2)=P_load_Q(k);
            Q_states(index,3)=SOC_Q(l);
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
% NB:Check consistency with the implementation of the Q-matrix below
% (number of parameters and number of actions possible for each parameter)

% #########################################################################
% ###############          Q-learning SETTINGS         ####################
% #########################################################################

% Confidence in new trials?
learnRate = 0.99;

% Exploration vs exploitation
epsilon = 0.5; % Initial value
epsilonDecay = 0.99992; % Decay factor per iteration

% Future vs present value
discount = 0.9;

% Inject some noise?
successRate = 1; % No noise

% How many episodes of testing ? (i.e. how many courses the system attend?)
maxEpi = 5;

% % How long are the episodes ? (i.e. how long are the courses?)
% maxit = 100;

% Q matrix:
% Lines: states | Rows: actions
Q=repmat(zeros(size(Q_states,1),1,'single'),[1,3]);
% Q elements are stored on 32bits (allow Qfactors between 2^-126 and 2^127)

% Load the functions (polynoms) calculating the rewards
load('rewardCurveSOC.mat')

% #########################################################################
% ################        INITIALIZE THE MODEL        #####################
% #########################################################################

% Choose model
model = 'DC_grid_V2';

% Set the (approximate) duration of one episode:
totalTime = 10;
% Set the length of one iteration in the simulink model
iterationTime = 1.3;

% Calculates the nomber of iterations
maxit = floor(totalTime/iterationTime);

% Empty structure containing the results of one iteration:
systemStatesTab = struct(...
    'time',zeros(maxit,1)...
    ,'P_FC',zeros(maxit,1)...
    ,'P_Batt',zeros(maxit,1)...
    ,'SOC_battery',zeros(maxit,1)...
    ,'Load_profile',zeros(maxit,1)...
    ,'Setpoint_I_FC',zeros(maxit,1)...
    ,'isExploitationAction',zeros(maxit,1));
% NOTE: This structure is overwritten each iteration.

% Initialize a .txt file containing relevant datas
delete results.txt
resultsReport = fopen('results.txt','w');

% Array containing the episode duration over simulation
epiDuration = [];


% #########################################################################
% #############              START LEARNING              ##################
% #########################################################################



for episodes = 1:maxEpi
    
    % $$$$$$$$$$$$$$$     INITIALIZE THE EPISODE      $$$$$$$$$$$$$$$$$$$$$
    
    % Measure the simulation time
    t_SimulinkTotal = 0;
    t_LearningStart = cputime;
    
    % Load the initial conditions
    load('initialState_1A.mat');
    inputArray(2) = 1;
    % Loading the SimState
    currentSimState = initialSimState;
    
    % Initial time (time when the iterations start)
    t_init = initialSimState.snapshotTime;
    
    
    % Charge the input for initial time: inputArray
    % (the input cannot ba calculated for initial time)
    % Row 1: Current command for the FC at the bus interface (i.e. between
    %        DC/DC conveter and bus. Unit is p.u. (base is the load).
    % Row 2: Load profile (1 for nominal power)
    inputsFromWS = Simulink.Parameter(inputArray);
    inputsFromWS.StorageClass='ExportedGlobal';
    
    % Initialize the the model constants to ensure consistency with the
    % initialization phase
    initialize_model_constants(model,model_constants);
    
    % Open the model and set the simulation modes
    initialize_model(model);
    
    % Starting point
    new_Q_state_struct = struct(...
        'P_FC',initial_outputsToWS.P_FC,...
        'P_load',initial_outputsToWS.Load_profile,...
        'SOC',initial_outputsToWS.SOC...
        );
    
    % Convert the structure to array for use in the Q-learning calculation
    current_Q_state_array = transpose(cell2mat(struct2cell(new_Q_state_struct)));
    
    % Number of exploitation actions (non-random actions):
    nExploitation = 0;
    
    % Initialize boolean for the case SOC < 5%
    failure = 0;
    
    for g = 1:maxit
        fprintf('Episode n.%i, iteration n.%i/%i\n',episodes,g,maxit);
        
        % $$$$$$$$$$$$$$$$$$     Pick an action     $$$$$$$$$$$$$$$$$$$$$$$
        % Interpolate the state within our discretization (ONLY for
        % choosing the action. We do not actually change the state by doing
        % this!)
        [~,sIdx] = min(sum((Q_states - repmat(current_Q_state_array,[size(Q_states,1),1])).^2,2));
        % sIdx is the index of the state matrix corresponding the best to
        % the current_state.
        
        % $$$$$$$$$$$$$$$$$    Choose an action    $$$$$$$$$$$$$$$$$$$$$$$$
        % EITHER 1) pick the best action according the Q matrix (EXPLOITATION).
        fprintf('epsilon = %2.2f\n',epsilon);
        if rand()>epsilon... % Exploit
                && rand()<=successRate... % Fail the check if our action doesn't succeed (i.e. simulating noise)
                && not(isequal(Q(sIdx,:),[0 0 0]))   % Take a random action when all the coefficients are equals
            
            [~,aIdx_fc] = max(Q(sIdx,:)); % Pick the action (for the FC current) the Q matrix thinks is best
            systemStatesTab.isExploitationAction(g) = 0.2;
            nExploitation = nExploitation + 1;
        % OR 2) Pick a random action (EXPLORATION)
        else
            aIdx_fc = randi(size(actions,2),1); % Random action for FC!
            systemStatesTab.isExploitationAction(g) = 0;         
        end
        
        % $$$$$$$$$$$$$$$$$    Run the model    $$$$$$$$$$$$$$$$$$$$$$$$$$$
        % New input for the model:
        
        dI_FC_Q = actions(1,aIdx_fc);
        inputArray(1) = inputArray(1) + dI_FC_Q;
        if inputArray(1)<0 % Current always flowing out of the FC
            inputArray(1)=0;
        end
        inputsFromWS.Value = inputArray;
        
        % Updating the state by running the model
        t_SimulinkIterationStart = cputime;
        [currentSimState,simOut] = run_simulation(model,currentSimState,iterationTime);
        t_SimulinkTotal = t_SimulinkTotal + cputime - t_SimulinkIterationStart;
        
        % Get the current time of the simulation:
        current_time = currentSimState.snapshotTime - t_init;
        
        % Fill the results of the iteration in the structure containing
        % results:
        systemStatesTab.time(g) = current_time;
        systemStatesTab.Fuel_Cell_power(g)  = simOut.outputsToWS.P_FC.Data(end); % Take the last value to see the impact of the input at the end of iteration time.
        systemStatesTab.Battery_power(g) = simOut.outputsToWS.P_batt.Data(end);
        systemStatesTab.SOC_battery(g) = simOut.outputsToWS.SOC.Data(end);
        systemStatesTab.Setpoint_I_FC(g) = inputArray(1);
        systemStatesTab.Load_profile(g) = simOut.outputsToWS.Load_profile.Data(end);
        
        % Fill the Q-learning state
        new_Q_state_struct.P_FC = simOut.outputsToWS.P_FC.Data(end);
        new_Q_state_struct.P_load = simOut.outputsToWS.Load_profile.Data(end);
        new_Q_state_struct.SOC = simOut.outputsToWS.SOC.Data(end);
        
        % Convert the structure to array for use in the Q-learning calculation
        new_Q_state_array = transpose(cell2mat(struct2cell(new_Q_state_struct)));
        
        % $$$$$$$$$$$$$$$$    Calculate the reward     $$$$$$$$$$$$$$$$$$$$
        reward = getReward(new_Q_state_struct);
        
        % $$$$$$$$$$$$$$$$   Update the Q-matrix    $$$$$$$$$$$$$$$$$$$$$$$
        % NB: no end condition of the episode here, because it is a
        % tracking problem.
        [~,snewIdx] = min(sum((Q_states - repmat(new_Q_state_array,[size(Q_states,1),1])).^2,2)); % Interpolate again to find the new state the system is closest to.
        current_Q_state_array = new_Q_state_array;
        
        % Update Q
        Q(sIdx,aIdx_fc) = Q(sIdx,aIdx_fc) + learnRate * ( reward + discount*max(Q(snewIdx,:)) - Q(sIdx,aIdx_fc) );
        
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

    
    % Analysis of the performance
    if ~failure
        t_LearningTotal = cputime - t_LearningStart;
        epiDuration = [epiDuration t_LearningTotal];
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
        fprintf(resultsReport,'Failure, SOC too close to 0');
    end
    
    % Plotting the result of the episode
    fig = figure(episodes);
    subplot(311)
    plot(systemStatesTab.time,systemStatesTab.Fuel_Cell_power,'.-');
    hold on
    plot(systemStatesTab.time,systemStatesTab.Battery_power,'.-');
    legend('P FC (p.u.)','P Batt (p.u.)','Location','southwest');
    subplot(312);
    plot(systemStatesTab.time,systemStatesTab.SOC_battery,'.-');
    hold on
    bar(systemStatesTab.time,systemStatesTab.isExploitationAction);
    legend('SOC','Exploitation','Location','southwest');
    subplot(313);
    plot(systemStatesTab.time,systemStatesTab.Setpoint_I_FC,'.-');
    hold on
    plot(systemStatesTab.time,systemStatesTab.Load_profile,'.-');
    %ylim([0,1.5]);
    legend('I FC (p.u.)','Load profile (p.u.)','Location','southwest');
    drawnow
    saveas(fig,['episode' num2str(episodes) '.jpg']);
    close(fig);
    
    % Save the Q-matrix
    save('Final_Q_matrix.mat','Q');
    
    
end % end episodes counting

% Plot the evolution of the duration episides duration:
fig = figure(maxEpi+1);
plot(epiDuration);
saveas(fig,'Episodes_duration.jpg');
close(fig);

% Close the text file
fclose(resultsReport);

%%
function initialize_model(model)
% DESCRIPTION:
% Function to be used before multiple simulations of the model.
% This function aim to reduce the time of execution of the simulation by
% setting 'FastRestart' i.e. no re-compilling of the model between the
% runs.
% NB: When the initialize function is called, the initial state must be
% known
% FREQUENCY OF EXECUTION:
% Once at the beginning of a multiple run simulation
% EXAMPLE OF USE:
% See example and test in the script SimState_testing_and_example

open_system(model);
set_param(model,'FastRestart','off');
set_param(model,'SaveFinalState','on','FinalStateName','myOperPoint',...
    'SaveCompleteFinalSimState','on','LoadInitialState','on');
set_param(model,'SimulationMode','accelerator');
set_param(model,'FastRestart','on');
end