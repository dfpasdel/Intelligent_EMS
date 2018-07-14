% DESCRIPTION:
% Fill values in x,y and check if the reward curve suits the reward
% strategy desired.

x = [-0.2  0  0.2 0.35 0.6 1  1.5];
y = [0     1  0   1   0   0  0];

rewardCurvePFC = polyfit(x,y,6);

X = 0:0.001:1;
Y = polyval(rewardCurvePFC,X);

figure(100)
plot(X,Y)

save('rewardCurvePFC.mat','rewardCurvePFC');

clear x y X Y