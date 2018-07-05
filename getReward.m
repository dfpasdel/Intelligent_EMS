function [r] = getReward(S)
% DESCRIPTION:
% Takes the Q-state as input and returns a reward as output.
% NOTE: 
% The Q-state is passed as a structure

soc = S.SOC;
if (0.65 <= soc) && (soc <= 0.75)
    r = 1;
elseif (soc <= 0.25)
    r = -1;
else
    r = 0;
end 
end