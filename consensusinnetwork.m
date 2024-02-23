clc; clear all; close all;
T = 50;
sharedprior = [0.5,0.5,0.5,0.5];
beliefs = [0.9,0.4,0.5,0.6];
V = zeros(T,4);
consensus = min(beliefs)/(min(beliefs) + min(1-beliefs));

for t = 1:T    
    gg = sharedprior(1);
    dummyvar = [beliefs(1),sharedprior(2:4)];
    sharedprior(1) = min(dummyvar)/(min(dummyvar) + min(1-dummyvar));
    dummyvar = [beliefs(2),gg];
    sharedprior(2) = min(dummyvar)/(min(dummyvar) + min(1-dummyvar));
    dummyvar = [beliefs(3),gg];
    sharedprior(3) = min(dummyvar)/(min(dummyvar) + min(1-dummyvar));
    dummyvar = [beliefs(4),gg];
    sharedprior(4) = min(dummyvar)/(min(dummyvar) + min(1-dummyvar));
    V(t,:) = sharedprior;
end

figure(1)
plot(V(:,1))
 hold on
plot(V(:,2))
hold on
plot(V(:,3))
 hold on
 plot(V(:,4))