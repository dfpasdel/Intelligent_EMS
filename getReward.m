function [r] = getReward(S,SOCpolynom)
% DESCRIPTION:
% Takes the Q-state as input and returns a reward as output.
% NOTE: 
% The Q-state is passed as a structure
%
% INPUTS:
% - Structure containing the current Qstate
% - Polynom defining the reward depending on the value of the Qstate (the
%   polynom is generated in a separate script and loaded once in the main 
%   script)

soc = S.SOC;
r = polyval(SOCpolynom,soc);
end