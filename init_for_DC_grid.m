%% DESCRIPTION:
% This script is a prerequisty for simulation. 
% It generates the start point for the iteratives simulation as .mat files

% Fill the function parameters as follow to generate initial states:
% make_initialState(
%       'name of the simulink model'
%       nominal voltage of the battery,
%       rated capacity of the battery,
%       initial SOC of the battery (=SOC at the end of the initialization)
%       battery response time
%       initial current of the FC delivered to the bus (in p.u., between
%                                        the DC/DC converter and the bus
%       initial load ID (see below)
%       'name of the .mat file to be generated'

% Initial load ID:
% In the simunink model, a switch block allows to select a type of load
% (constant, square, sinus). The ID are the followings:
% 1- 50% load 
% 2- 100% load 
% 3- Pulse of period 1sec
% 4- Pulse of period 5sec
% 5- Pulse of period 60sec
% 6- Sinus of period 1sec
% 7- Sinus of period 5sec
% 8- Sinus of period 60sec
% 9- 0% load (default case)


% NOTE: The FC have internal resistance, then I_bus_FC is not proportionnal 
% to the power output.
% I_bus_FC = 16.18A => P_FC = 3kW
% I_bus_FC = 36.64A => P_FC = 6kW
% I_bus_FC = 56.31A => P_FC = 8kW
% Remember to divide by the nominal current.

model = 'DC_grid_V2';
make_initialState(model,100,1.5,70,2,16.18/30,1,'initialState_1A');
make_initialState(model,100,1.5,99,2,16.18/30,1,'initialState_1B');
make_initialState(model,100,1.5,30,2,16.18/30,1,'initialState_1C');
make_initialState(model,100,1.5,70,2,0,9,'initialState_2X'); % No load for initialization
make_initialState(model,100,1.5,70,2,0,9,'initialState_3X'); % No load for initialization

function make_initialState(simulinkModel,nom_vol,rat_cap,init_soc,...
    batt_response_time,I_bus_FC_0,ID_load_profile_0,name)
% DESCRIPTION:
% OUTPUTS:
% A .mat file containing:
%   - A SimState set of values (the current system parameters in the
%   Simulink point of view. Does not contain data of interest for the user.
%   Type: ModelSimState
%
%   - Model Constants
%   Some values of interest for the Mask of the FC and Battery models,
%   such as the initial SOC (%), the rated capacity (Ah) or the response
%   time (s) for the battery. This values stay the same all along the
%   simulation.
%   Note that other values for the FC and Battery models are passed 
%   graphically. For further work++, all the values (i.e. the complete 
%   mask) should be passed programatically (out of scope here). 
%   Type: struct
%
%   - The input parameters of the DC grid (see inputsFromWS in the simulink
%   model). This values are supposed to change during simulation. 
%   Type: Array
%
%   - The output parameters of the DC grid (see outputsToWS in the simulink
%   model). Type: struct
%% Setting the input parameters for initialization:

% ################          Model constants          ######################
model_constants = struct(...
    'nominal_voltage',nom_vol,...
    'rated_capacity',rat_cap,...
    'initial_SOC',init_soc,...
    'battery_response_time',batt_response_time);
% Note: It is misuse of language to say that the initial SOC is a constant,
% but it is percieved as it by the model.

% ################         Input parameters          ###################### 
% Converting the initial set points in an array form for the model:
inputArray = [I_bus_FC_0,ID_load_profile_0]; 
inputsFromWS = Simulink.Parameter(inputArray);
assignin('base','inputsFromWS',inputsFromWS);
inputsFromWS.StorageClass='ExportedGlobal';

% Call the function which initialize the user chosen model constants in the
% Simulink model:
initialize_model_constants(simulinkModel,model_constants)

% Generate the SimState and the Output structure of values
[initialSimState,initial_outputsToWS] = generate_start_point(simulinkModel);

% Make a .mat file containing the total state after initialization precedure
save([name,'.mat'],'initialSimState','model_constants',...
    'inputArray','initial_outputsToWS');
end

function [initialSimState,initial_outputsToWS] = generate_start_point(simulinkModel)
% DESCRIPTION:
% The goal of this function is to generate an initial state for a model which
% has no initial state already. The initial state outputed is the state
% after 10s of simulation (it is supposed that after 10s, the system is in 
% a steady state).
%
% INPUTS:
% The model for which the initial state has to be generated.
%
% OUTPUTS:
% Save an initial state in the current folder as .mat file
%
% FREQUENCY OF EXECUTION:
% To be ran for getting new initial states i.e. not so often. 
% NB: To generate multiple initial states, there is need to rename them
% manually in the folder.
% EXAMPLE OF USE:
% See example and test in the script SimState_testing_and_example

set_param(simulinkModel,'FastRestart','off');
set_param(simulinkModel,'SaveFinalState','on','FinalStateName','myOperPoint',...
    'SaveCompleteFinalSimState','on','LoadInitialState','off');
disp('You have been setting your initial conditions and model constants, the initial state is being generated');
simOut = sim(simulinkModel,'StopTime','10','SimulationMode','accelerator');

% The operation point of the model in the Simulink view
initialSimState = simOut.myOperPoint;   

% Variables of interest in the the user view:
initial_outputsToWS = struct(...
    'P_FC',simOut.outputsToWS.P_FC.Data(end),...
    'P_batt',simOut.outputsToWS.P_FC.Data(end),...
    'SOC',simOut.outputsToWS.SOC.Data(end),...
    'Fuel_flow',simOut.outputsToWS.Fuel_flow.Data(end),...
    'Stack_efficiency',simOut.outputsToWS.Stack_efficiency.Data(end),...
    'Load_profile',simOut.outputsToWS.Load_profile.Data(1)); 

set_param(simulinkModel,'LoadInitialState','on');  % Prevent of being off
set_param(simulinkModel,'FastRestart','on');
end

