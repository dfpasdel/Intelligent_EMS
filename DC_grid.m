%% DESCRIPTION:
% This script aim to demonstrate that the model can ensure over time the
% condition Pload=Pfc+Pbatt
%
% INPUTS:
% Initial state of the DC grid as a .mat file. %add me 
% The initial state contains the folowing data:
%   - General state of the DC grid: Matlab SimState file
%   - The model constants (e.g. nominal volatge, battery capacity...)
%   - The initial SOC of the battery
%   - The value of the inputs parameters for the time t=0
%   - The value of the outputs parameters for the time t=0
%
% NOTE #1: By convention, all currents, voltages and powers values 
% mentionned in the worksapce environment are called I,V and P, and are 
% exprimed in per-unit values. (the conversion p.u./real values is done in 
% the simulink model). The base for p.u. is the load: Ibase = 30A, Vbase = 
% 200V
%
% NOTE #2: The inputs in the simulink model can be bypassed for debugging
% by running simulations from the Simulink interface. To avoid errors 
% flags, run it in a new file without the Q-learning interface.

%% Choose model
model = 'DC_grid_V2';

%% Run simulations from initial state

% Load the initial conditions here. 
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

% Duration of the simulation:
iterationTime = 1.3;
totalTime = 30;
numberIterations = floor(totalTime/iterationTime);

% Open the model and set the simulation modes
initialize_model(model);

% Measure the simulation time
t_SimulinkTotal = 0;
t_LearningStart = cputime;

% Select the variables used for Q-learning
% The Q-learning code requires an array as input. Always follow this
% pattern (and check for consistency when changing the number of Q states)

new_Q_state_struct = struct(...
    'P_FC',initial_outputsToWS.P_FC,...
    'P_load',initial_outputsToWS.Load_profile,...
    'SOC',initial_outputsToWS.SOC...
    );
    
% Empty structure containing the results of one iteration:
systemStatesTab = struct(...
    'time',zeros(numberIterations,1)...
    ,'P_FC',zeros(numberIterations,1)...
    ,'P_Batt',zeros(numberIterations,1)...
    ,'SOC_battery',zeros(numberIterations,1)...
    ,'Load_profile',zeros(numberIterations,1)...
    ,'Setpoint_I_FC',zeros(numberIterations,1));

for i = 1:numberIterations
    fprintf('Iteration n.%i\n',i);
    
    % Simulate the model for dt=iterationTime
    t_SimulinkIterationStart = cputime; 
    [currentSimState,simOut] = run_simulation(model,currentSimState,iterationTime);
    t_SimulinkTotal = t_SimulinkTotal + cputime - t_SimulinkIterationStart; 
   
    % Get the current time of the simulation:
    current_time = currentSimState.snapshotTime - t_init;
    
    % HERE DO MACHINE LEARNING BASED ON THE OUTPUTS RESULTS AND CALCULATING
    % THE NEW INPUT
    
    % Update the input for next iteration step:
    %inputArray = [min(1.22,0.1*i),4];
    
    inputsFromWS.Value = inputArray;
    
    % Fill the results of the iteration in the structure containing
    % results:
    systemStatesTab.time(i) = current_time;
    systemStatesTab.Fuel_Cell_power(i)  = simOut.outputsToWS.P_FC.Data(end); % Take the last value to see the impact of the input at the end of iteration time.
    systemStatesTab.Battery_power(i) = simOut.outputsToWS.P_batt.Data(end);
    systemStatesTab.SOC_battery(i) = simOut.outputsToWS.SOC.Data(end);
    systemStatesTab.Setpoint_I_FC(i) = inputArray(1);
    systemStatesTab.Load_profile(i) = simOut.outputsToWS.Load_profile.Data(end);
    
    % Fill the Q-learning state
    new_Q_state_struct.P_FC = simOut.outputsToWS.P_FC.Data(end);
    new_Q_state_struct.P_load = simOut.outputsToWS.Load_profile.Data(end);
    new_Q_state_struct.SOC = simOut.outputsToWS.SOC.Data(end);
       
end

t_LearningTotal = cputime - t_LearningStart;

%%
fprintf('Simulink time: %5.1f\n',t_SimulinkTotal);
fprintf('Learning time (Simulink + Q-process): %5.1f\n',t_LearningTotal);
ratio = (t_SimulinkTotal/t_LearningTotal)*100;
fprintf('Ratio Simulink/Learning time (percent): %3.2f\n',ratio);

%%
figure(2)
subplot(311)
plot(systemStatesTab.time,systemStatesTab.Fuel_Cell_power,'o-');
hold on
plot(systemStatesTab.time,systemStatesTab.Battery_power,'.-');
legend('Power FC','Power Batt');
subplot(312);
plot(systemStatesTab.time,systemStatesTab.SOC_battery,'*-');
legend('SOC');
subplot(313);
plot(systemStatesTab.time,systemStatesTab.Setpoint_I_FC,'o-');
hold on
plot(systemStatesTab.time,systemStatesTab.Load_profile,'.-');
ylim([0,1.5]);
legend('I_FC','Load profile');

% Note:
% At x(n), the input plotted is the input for the next iteration, and the
% output plotted is the output resulting of the previous iteration (i.e.
% the result of the input at x(n-1)).

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
