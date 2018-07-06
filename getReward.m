function [r] = getReward(S)
% DESCRIPTION:
% Takes the Q-state as input and returns a reward as output.
% NOTE: 
% The Q-state is passed as a structure
%
% INPUTS:
% - Structure containing the current Qstate

soc = S.SOC; % State of charge
d_soc = S.rateSOC;  % Is the rate of change positive (+1) or negative (-1)
switch true
    case ((0.6 <= soc) && (soc <= 0.65)) || ((0.75 <= soc) && (soc <= 0.8))
        r = 1;
    case ((0.65 < soc) && (soc < 0.75))
        r = 3;
    case (soc < 0.6) && (d_soc == -1)
        r = -10;
    case (soc < 0.6) && (d_soc == 1)
        r = -5;
    case (soc > 0.8) && (d_soc == -1)
        r = -5;
    case (soc > 0.8) && (d_soc == 1)
        r = -10;
end
end