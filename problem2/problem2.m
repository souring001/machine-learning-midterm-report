A = [3,0.5;0.5,1];
m = [1;2];
w = [3;-1];

w_true2 = [0.82;1.09];
w_true4 = [0.64;0.18];
w_true6 = [0.33;0];

lambda = 2;
gamma = max(eig(2*A));
q = lambda/gamma;

figure
semilogy(1,norm(w-w_true2),'ro')
hold on


for n=1:50
    grad=2*A*(w-m)
    m1 = w-grad/gamma;
    for i=1:2
        if m1(i)>q
            w(i) = m1(i)-q;
        elseif m1(i)<-q
            w(i) = m1(i)+q;
        else
            w(i) = 0;
        end
    end
    w
    m
    semilogy(n,norm(w-w_true2),'ro')
    hold on
end

xlabel('iteration')
ylabel('log loss')
title('\lambda = 2')
