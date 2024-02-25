clc; clear all; close all;
l1w1t1 = [0.51,0.5,0.5,0.5];
l1w1t2 = [0.50,0.5,0.5,0.5];
logl1w1t1 = log([0.501,0.5,0.5,0.5]);
logl1w1t2 = log([0.50,0.5,0.5,0.5]);
T = 5000*3;
nature = rand>0.5
prior = [0.5,0.5,0.5,0.5];
sharedprior = [0.5,0.5,0.5,0.5];
sharedprior2 = [0.5,0.5,0.5,0.5];
sharedprior3 = [0.5,0.5,0.5,0.5];

U = zeros(T,4);
V = zeros(T,4);
W = zeros(T,4);
X = zeros(T,4);

for t = 1:T
    if nature==0
        signal = rand([1,4]) > l1w1t1;
    else
        signal = rand([1,4]) > l1w1t2;
    end
    for n = 1:4
        if signal(n) == 0
            
            prior(n) = prior(n)*l1w1t1(n)/(prior(n)*l1w1t1(n) + (1-prior(n))*l1w1t2(n));
            
        else 
            prior(n) = prior(n)*(1-l1w1t1(n))/(prior(n)*(1-l1w1t1(n)) + (1-prior(n))*(1-l1w1t2(n)));
        end
        U(t,:) = prior;
    end
    gg = sharedprior(1);
    dummyvar = [prior(1),sharedprior(2:4)];
    sharedprior(1) = min(dummyvar)/(min(dummyvar) + min(1-dummyvar));
    dummyvar = [prior(2),gg];
    sharedprior(2) = min(dummyvar)/(min(dummyvar) + min(1-dummyvar));

    dummyvar = [prior(3),gg];
    sharedprior(3) = min(dummyvar)/(min(dummyvar) + min(1-dummyvar));
    dummyvar = [prior(4),gg];
    sharedprior(4) = min(dummyvar)/(min(dummyvar) + min(1-dummyvar));
    V(t,:) = sharedprior;
%%%% 
    oldsharedprior = sharedprior2;
    dummyvar = [prior(1),oldsharedprior(2:4)];
    sharedprior2(1) = min(dummyvar )*l1w1t1(1)/(min(dummyvar )*l1w1t1(1) + min(1-dummyvar )*l1w1t2(1));
    dummyvar = [prior(2),oldsharedprior(1)];
    sharedprior2(2) = min(dummyvar)*l1w1t1(2)/(min(dummyvar)*l1w1t1(2) + min(1-dummyvar)*l1w1t2(2));

    dummyvar = [prior(3),oldsharedprior(1)];
    sharedprior2(3) = min(dummyvar)*l1w1t1(3)/(min(dummyvar)*l1w1t1(3) + min(1-dummyvar)*l1w1t2(3));
    dummyvar = [prior(4),oldsharedprior(1)];
    sharedprior2(4) = min(dummyvar)*l1w1t1(4)/(min(dummyvar)*l1w1t1(4) + min(1-dummyvar)*l1w1t2(4));
    W(t,:) = sharedprior2;
%%%%
    oldsharedprior = sharedprior3;
    dummyvar = oldsharedprior(1:4);
    if signal(1) == 0
        sharedprior3(1) = min(dummyvar )*l1w1t1(1)/(min(dummyvar )*l1w1t1(1) + min(1-dummyvar )*l1w1t2(1));
    else 
        sharedprior3(1) = min(dummyvar )*(1-l1w1t1(1))/(min(dummyvar )*(1-l1w1t1(1)) + min(1-dummyvar )*(1-l1w1t2(1)));
    end

    dummyvar = [oldsharedprior(1),oldsharedprior(2)];
    if signal(2) == 0
        sharedprior3(2) = min(dummyvar)*l1w1t1(2)/(min(dummyvar )*l1w1t1(2) + min(1-dummyvar )*l1w1t2(2));
    else 
        sharedprior3(2) = min(dummyvar)*(1-l1w1t1(2))/(min(dummyvar )*(1-l1w1t1(2)) + min(1-dummyvar )*(1-l1w1t2(2)));
    end

    dummyvar = [oldsharedprior(1),oldsharedprior(3)];
    if signal(3) == 0
        sharedprior3(3) = min(dummyvar)*l1w1t1(2)/(min(dummyvar )*l1w1t1(2) + min(1-dummyvar )*l1w1t2(2));
    else 
        sharedprior3(3) = min(dummyvar)*(1-l1w1t1(2))/(min(dummyvar )*(1-l1w1t1(2)) + min(1-dummyvar )*(1-l1w1t2(2)));
    end
    dummyvar = [oldsharedprior(1),oldsharedprior(4)];
    if signal(4) == 0
        sharedprior3(4) = min(dummyvar)*l1w1t1(2)/(min(dummyvar )*l1w1t1(2) + min(1-dummyvar )*l1w1t2(2));
    else 
        sharedprior3(4) = min(dummyvar)*(1-l1w1t1(2))/(min(dummyvar )*(1-l1w1t1(2)) + min(1-dummyvar )*(1-l1w1t2(2)));
    end
    X(t,:) = sharedprior3; 
end
figure(1)
plot(U(:,1))
hold on
plot(U(:,2))
figure(2)
plot(V(:,1))
hold on
plot(V(:,2))
figure(3)
plot(W(:,1))
hold on
plot(W(:,2))
figure(4)
plot(X(:,1))
hold on
plot(X(:,2))