function [finalState,simOut] = run_simulation(model,initState,simTime)
% DESCRIPTION:
% Run the simulink model between the instants [initTime;initTime+simTime]. 
% The state at the beginning of the simulation is given through the 
% argument 'initState'.
% The function returns the new state (finalState) at the end of the 
% simulation and the time corresponding (finalTime).
% FREQUENCY OF EXECUTION: 
% In a loop
% EXAMPLE OF USE:
% See example and test in the script SimState_testing_and_example

initTime  = initState.snapshotTime;
finalTime = initTime + simTime; % Calculate the final time of the simulation
assignin('base','finalTime',finalTime);
assignin('base','initState',initState);
simOut = sim(model,'StopTime','finalTime','InitialState','initState');
finalState = simOut.myOperPoint; % Update the model state at the end of the simulation

end