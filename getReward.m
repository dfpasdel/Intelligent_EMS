function [r] = getReward(S)
% DESCRIPTION:
% Takes the Q-state as input and returns a reward as output.
% NOTE: 
% The Q-state is passed as a structure

% switch true
%     case S.SOC < 0.6
%         r = 0;
%     case S.SOC > 0.8
%         r = 0;
%     case 0.6 <= S.SOC <= 0.8
%         r = 1;
% end
soc = S.SOC;
if (0.68 <= soc) && (soc <= 0.72)
    r = 1;
else
    r = 0;
    disp('fail');
end 
end