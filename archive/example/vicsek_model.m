% This code is made to give a primer for those who want to enjoy the Vicsek model simulation.
% Reference: 1995, Vicsek, T. et al., PRL

%%
clear all
close all

%%
TMAX=1000;

L=7; % domain size
noise=2.0; % noise
N=300; % particle number
r=1; % interacting radius
vel=0.03; % absolute velocity 
dt=1;

%% Boundary condition
PERIODIC=1; % 1: periodic boundary condition, 0: unlimited

%% Initial condition
x=L*rand(1,N);
y=L*rand(1,N);
theta=2*pi*(rand(1,N)-0.5); % randomly distributed

%plot(x,y,'.','MarkerSize',15)

for time=1:TMAX
   
    %% Calculation of average angle in the interacting circle
    D = pdist([x' y'],'euclidean');
    
    % Periodic boundary %
    if PERIODIC==1
        tmp_x(x<r) = L + x(x<r);
        tmp_x(x>L-r) = x(x>L-r)-L;
        tmp_x(r<=x & x<=L-r) = x(r<=x & x<=L-r);
        tmp_y(y<r) = L + y(y<r);
        tmp_y(y>L-r) = y(y>L-r)-L;
        tmp_y(r<=y & y<=L-r) = y(r<=y & y<=L-r);
        
        tmp_D = pdist([tmp_x' tmp_y'],'euclidean');        
        D = min([D; tmp_D]);
    end
    
    M = squareform(D); %@Matrix representation for the distance between particles
    [l1,l2]=find(0<M & M<r);
    
    for i = 1:N
        list = l1(l2==i);
        if ~isempty(list)
            ave_theta(i) = atan2(mean(sin(theta(list))),mean(cos(theta(list))));
        else
            ave_theta(i) = theta(i);
        end
    end
    
    %% Update
    x = x + vel*cos(theta)*dt;
    y = y + vel*sin(theta)*dt;
    
    if PERIODIC==1
        x(x<0) = L + x(x<0);
        x(L<x) = x(L<x) - L;
        y(y<0) = L + y(y<0);
        y(L<y) = y(L<y) - L;
    end

    theta = ave_theta + noise*(rand(1,N) - 0.5);
    
    %% Figure
    plot(x,y,'.','MarkerSize',15)
    xlim([0 L]);
    ylim([0 L]);
    axis square

    pause(0.1)
end

